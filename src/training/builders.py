from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, WeightedRandomSampler

import global_config
from src.config import ExperimentConfig, ModelConfig, TrainConfig
from src.data.augmentation.pipeline import AugmentationPipeline
from src.data.collation import ValidationCollator
from src.data.data_workspace import DataWorkspace
from src.data.dataset import AIIJCDataset
from src.training.sampling import (
    DistributedBatchSampler,
    DistributedValidationSampler,
    FinalFullTrainSampler,
    FocusCoverageSampler,
    UniformMaskAreaSampler,
)

if TYPE_CHECKING:
    from src.modules.segmenter import Segmenter


class DataLoaderThreadLimits:
    """Keep CPU libraries single-threaded in the parent and loader workers."""

    @staticmethod
    def apply(worker_id: int | None = None) -> None:
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[variable] = "1"
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        # Also limit libraries already imported by a notebook or forked worker.
        threadpool_limits(limits=1)


@dataclass(frozen=True)
class AmpContext:
    device: torch.device
    dtype: torch.dtype
    enabled: bool
    scaler_enabled: bool

    def autocast(self) -> Any:
        return torch.amp.autocast(
            self.device.type,
            dtype=self.dtype,
            enabled=self.enabled,
        )

    def scaler(self) -> torch.amp.GradScaler:
        return torch.amp.GradScaler(
            self.device.type,
            enabled=self.scaler_enabled,
        )


def build_model(config: ModelConfig, *, pretrained: bool = True) -> Segmenter:
    from src.modules.segmenter import Segmenter

    model = Segmenter(
        encoder_name=config.encoder_name,
        forensic_channels=config.forensic_channels,
        norm=config.norm,
        aux_weight=config.aux_weight,
        use_forensics=config.use_forensics,
        forensic_mode=config.forensic_mode,
        jpeg_variant=config.jpeg_variant,
        fusion_variant=config.fusion_variant,
        dct_aux_weight=config.dct_aux_weight,
        forensic_contrastive_dim=config.forensic_contrastive_dim,
        pretrained=pretrained,
        decoder_kwargs=config.decoder_kwargs,
        local_image_size=config.local_image_size,
        luma_image_size=config.luma_image_size,
        wavelet_image_size=config.wavelet_image_size,
        wavelet_fusion=config.wavelet_fusion,
        wavelet_aux_source=config.wavelet_aux_source,
        strided_resize=config.strided_resize,
        resize_variant=config.resize_variant,
        noise_encoder_name=config.noise_encoder_name,
        guided_radius=config.guided_radius,
        guided_epsilon=config.guided_epsilon,
        guided_scale=config.guided_scale,
        dual_fusion_width=config.dual_fusion_width,
    )
    if pretrained and config.jpeg_pretrained is not None:
        path = Path(config.jpeg_pretrained)
        if not path.is_absolute():
            path = global_config.PROJECT_ROOT / path
        model.forensic_fusion.branch.artifact.load_pretrained(path)
    if config.sync_batchnorm:
        from src.modules.sync_batchnorm import SynchronizedBatchNorm

        model = SynchronizedBatchNorm.apply(model)
    return model


def configure_memory_format(model):
    """Use NHWC for fixed RGB shapes, NCHW for the variable-size JPEG stream.

    On RTX 3070, new native shapes in the NHWC JPEG stream incurred seconds
    of cold convolution overhead. NCHW avoids this without resampling inputs.
    Apply before constructing the optimizer or EMA.
    """
    model.to(memory_format=torch.channels_last)
    if getattr(model, 'forensic_mode', 'maps') == 'jpeg':
        model.forensic_fusion.branch.to(memory_format=torch.contiguous_format)
    return model


def build_datasets(
    config: ExperimentConfig,
    data_workspace: DataWorkspace,
    train_df,
    val_df,
) -> tuple[AIIJCDataset, AIIJCDataset]:
    augmentations = AugmentationPipeline(config.augmentation, total_epochs=config.train.epochs)
    amp = build_amp(config.train)
    local_dtype = amp.dtype if amp.enabled else torch.float32

    train_ds = AIIJCDataset(
        data_workspace=data_workspace,
        folded_df=train_df,
        train=True,
        image_size=config.dataset.image_size,
        seed=config.seed,
        augmentations=augmentations,
        boundary_targets=config.loss.boundary_weight > 0,
        fmap_channels=None,
        use_forensics=config.model.use_forensics,
        forensic_mode=config.model.forensic_mode,
        jpeg_variant=config.model.jpeg_variant,
        mode="train",
        resize_mode=config.dataset.resize_mode,
        local_image_size=config.model.local_image_size,
        luma_image_size=config.model.luma_image_size,
        wavelet_image_size=config.model.wavelet_image_size,
        strided_resize=config.model.strided_resize,
        local_dtype=local_dtype,
    )

    val_ds = AIIJCDataset(
        data_workspace=data_workspace,
        folded_df=val_df,
        train=False,
        image_size=config.dataset.image_size,
        seed=config.seed,
        augmentations=None,
        boundary_targets=config.loss.boundary_weight > 0,
        fmap_channels=None,
        use_forensics=config.model.use_forensics,
        forensic_mode=config.model.forensic_mode,
        jpeg_variant=config.model.jpeg_variant,
        mode="val",
        resize_mode=config.dataset.resize_mode,
        original_targets=True,
        local_image_size=config.model.local_image_size,
        luma_image_size=config.model.luma_image_size,
        wavelet_image_size=config.model.wavelet_image_size,
        strided_resize=config.model.strided_resize,
        local_dtype=local_dtype,
    )

    return train_ds, val_ds


def build_sampler(
    config: TrainConfig,
    train_ds: AIIJCDataset,
) -> WeightedRandomSampler:
    if config.sampling_strategy == 'uniform_mask_area':
        if 'mask_area' not in train_ds.df:
            raise ValueError('uniform_mask_area requires mask_area in the train dataset')
        return UniformMaskAreaSampler(train_ds.df['mask_area'], config.epoch_size)
    if config.sampling_strategy == 'focus_coverage':
        manifest = Path(config.focus_manifest)
        if not manifest.is_absolute():
            manifest = global_config.PROJECT_ROOT / manifest
        paths = pd.read_parquet(manifest)['chng_img_path']
        train_paths = train_ds.df['chng_img_path']
        if paths.duplicated().any() or not paths.isin(train_paths).all():
            raise ValueError('focus manifest must contain unique paths from train only')
        return FocusCoverageSampler(train_paths.isin(paths), train_ds.is_negative,
                                    config.epoch_size, config.focus_fraction, config.negative_fraction)
    neg = train_ds.is_negative
    if neg is None:
        raise ValueError("train dataset must expose is_negative for weighted sampling")

    n_neg = int(neg.sum())
    n_pos = int((~neg).sum())
    if n_neg == 0 or n_pos == 0:
        raise ValueError(f"weighted sampling requires both classes, got pos={n_pos}, neg={n_neg}")

    neg_fr = config.negative_fraction
    weights = np.where(neg, neg_fr / n_neg, (1 - neg_fr) / n_pos)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=config.epoch_size,
        replacement=True,
    )


def build_loaders(
    config: TrainConfig,
    train_ds: AIIJCDataset,
    val_ds: AIIJCDataset,
    *,
    runtime=None,
) -> tuple[DataLoader, DataLoader]:
    DataLoaderThreadLimits.apply()
    pin_memory = torch.device(config.device).type == "cuda"
    local = (getattr(train_ds, 'local_preprocessor', None) is not None
             or bool(getattr(train_ds, 'strided_resize', False))
             or bool(getattr(train_ds, 'luma_image_size', 0))
             or bool(getattr(train_ds, 'wavelet_image_size', 0))
             or getattr(train_ds, 'forensic_mode', 'maps') == 'jpeg')
    # Local views are 60 MiB each; avoid buffering two huge batches per worker.
    prefetch = 1 if local else 2
    sampler = build_sampler(config, train_ds)
    if config.full_train_epochs:
        sampler = FinalFullTrainSampler(sampler, len(train_ds), config.epochs - config.full_train_epochs)
    batching = dict(batch_size=config.batch_size, sampler=sampler,
                    drop_last=not bool(config.full_train_epochs))
    if runtime is not None and runtime.distributed:
        batching = dict(batch_sampler=DistributedBatchSampler(
            sampler, config.batch_size, runtime.rank, runtime.world_size,
            drop_last=not bool(config.full_train_epochs)))
    train_loader = DataLoader(
        train_ds,
        **batching,
        collate_fn=ValidationCollator() if (getattr(train_ds, 'luma_image_size', 0)
                                           or getattr(train_ds, 'wavelet_image_size', 0)
                                           or getattr(train_ds, 'forensic_mode', 'maps') == 'jpeg') else None,
        num_workers=config.workers,
        pin_memory=pin_memory,
        persistent_workers=config.workers > 0,
        worker_init_fn=DataLoaderThreadLimits.apply,
        prefetch_factor=prefetch if config.workers else None,
    )
    val_batch_size = config.batch_size if local else config.batch_size * 2
    val_sampler = (DistributedValidationSampler(val_ds, val_batch_size, runtime.rank, runtime.world_size)
                   if runtime is not None and runtime.distributed else None)
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=pin_memory,
        persistent_workers=config.workers > 0,
        worker_init_fn=DataLoaderThreadLimits.apply,
        collate_fn=ValidationCollator(),
        prefetch_factor=prefetch if config.workers else None,
    )
    return train_loader, val_loader


def build_optimizer(
    config: TrainConfig,
    model,
) -> torch.optim.AdamW:
    """AdamW with separate LR/weight decay for encoder, forensic branch and decoder."""
    groups = {"enc": [], "enc_nd": [], "fmap": [], "fmap_nd": [], "dec": [], "dec_nd": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = parameter.ndim <= 1 or name.endswith(("channel_gate", "gamma"))
        if name.startswith(("forensic_fusion.", "branch.", "fuse.")):
            groups["fmap_nd" if no_decay else "fmap"].append(parameter)
        elif name.startswith("encoder.") and not name.startswith('encoder.fusions.'):
            groups["enc_nd" if no_decay else "enc"].append(parameter)
        else:
            groups["dec_nd" if no_decay else "dec"].append(parameter)

    param_groups = [
        {"params": groups["enc"], "lr": config.encoder_lr, "weight_decay": config.weight_decay},
        {"params": groups["enc_nd"], "lr": config.encoder_lr, "weight_decay": 0.0},
        {"params": groups["fmap"], "lr": config.fmap_lr, "weight_decay": config.weight_decay},
        {"params": groups["fmap_nd"], "lr": config.fmap_lr, "weight_decay": 0.0},
        {"params": groups["dec"], "lr": config.lr, "weight_decay": config.weight_decay},
        {"params": groups["dec_nd"], "lr": config.lr, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW([group for group in param_groups if group["params"]])


def build_scheduler(
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    *,
    full_steps_per_epoch: int | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    steps_per_epoch = max(1, int(steps_per_epoch))
    if config.full_train_epochs:
        if full_steps_per_epoch is None or full_steps_per_epoch <= 0:
            raise ValueError('full_steps_per_epoch is required for full train finetuning')
        main_steps = steps_per_epoch * (config.epochs - config.full_train_epochs)
        final_steps = full_steps_per_epoch * config.full_train_epochs
        warmup = int(config.warmup_frac * main_steps)

        def phase_lr(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = min(1.0, max(0.0, (step - main_steps) / final_steps))
            return config.min_lr_factor + (1 - config.min_lr_factor) * .5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, phase_lr)
    total_steps = max(1, steps_per_epoch * config.epochs)
    warmup = int(config.warmup_frac * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return config.min_lr_factor + (1 - config.min_lr_factor) * 0.5 * (
            1 + math.cos(math.pi * min(progress, 1.0))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_amp(
    config: TrainConfig,
    device: torch.device | None = None,
) -> AmpContext:
    device = torch.device(config.device) if device is None else device
    return AmpContext(
        device=device,
        dtype=torch.bfloat16 if config.amp == "bf16" else torch.float16,
        enabled=device.type == "cuda" and config.amp != "off",
        scaler_enabled=device.type == "cuda" and config.amp == "fp16",
    )


def build_ema(
    config: TrainConfig,
    model,
) -> torch.optim.swa_utils.AveragedModel:
    return torch.optim.swa_utils.AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config.ema_decay),
    )


__all__ = [
    "AmpContext",
    "build_amp",
    "build_datasets",
    "build_ema",
    "build_loaders",
    "build_model",
    "build_optimizer",
    "build_sampler",
    "build_scheduler",
]
