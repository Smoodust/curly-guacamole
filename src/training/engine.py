from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sized
from dataclasses import dataclass, field

import torch
from torch import nn

from src.budget import count_gflops
from src.config import ExperimentConfig
from src.data.data_workspace import DataWorkspace
from src.eval.diagnostics import EvaluationReport
from src.eval.metadata import build_metadata
from src.eval.protocol import EvaluationProtocol
from src.eval.splits import make_stratified_val_folds, train_val_fold_split
from src.forensic.dct.constants import CHANNELS as FMAP_CHANNELS
from src.losses import LossMeter, LossResult, build_loss
from src.progress import ConsoleProgress
from src.training.base import set_random_seed
from src.training.builders import (
    AmpContext,
    build_amp,
    build_datasets,
    build_ema,
    build_loaders,
    build_model,
    build_optimizer,
    build_scheduler,
)
from src.training.metric import AICResult
from src.training.runs import Run
from src.training.validation import validate


@dataclass(frozen=True)
class EpochTrainResult:
    loss: float
    skipped_steps: int
    seen: int
    negative_fraction: float = 0.0
    loss_components: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingState:
    start_epoch: int = 0
    seen_total: int = 0
    best_aic: float = -1.0
    best_result: AICResult | None = None


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, *, arm: str = "baseline") -> None:
        self.config = config
        self.arm = arm
        self.device = torch.device(config.train.device)
        self.amp = build_amp(config.train, self.device)
        self.data_workspace = DataWorkspace(config.paths.data_path)
        self.loss = build_loss(config)

    def run(self) -> Run:
        cfg = self.config
        self._check_resume_protocol()
        ConsoleProgress.info(f"Эксперимент {cfg.paths.run_name}: device={self.device}, amp={cfg.train.amp}, seed={cfg.seed}")
        ConsoleProgress.info(f"Функция потерь: {self.loss}")
        set_random_seed(cfg.seed)

        ConsoleProgress.info("Подготовка метаданных и train/val разбиения")
        train_df, val_df = self._split_data()
        ConsoleProgress.info(f"Разбиение готово: train={len(train_df)}, val={len(val_df)}; создание датасетов и аугментаций")
        train_ds, val_ds = build_datasets(cfg, self.data_workspace, train_df, val_df)
        ConsoleProgress.info(f"Создание модели {cfg.model.encoder_name}, загрузка pretrained-весов и перенос на {self.device}")
        model = build_model(cfg.model).to(self.device, memory_format=torch.channels_last)
        ConsoleProgress.info("Модель готова; подсчёт GFLOPS")
        gflops = count_gflops(model, cfg.dataset.image_size,
                             use_valid_mask=cfg.dataset.resize_mode == "letterbox")
        ConsoleProgress.info(f"Подсчёт завершён: {gflops:.2f} GFLOPS; создание оптимизатора")
        optimizer = build_optimizer(cfg.train, model)
        ConsoleProgress.info(f"Создание DataLoader: batch_size={cfg.train.batch_size}, workers={cfg.train.workers}")
        train_loader, val_loader = build_loaders(cfg.train, train_ds, val_ds)

        ConsoleProgress.info(f"DataLoader готовы: train={len(train_loader)} батчей, val={len(val_loader)}; настройка scheduler, AMP scaler и EMA")
        steps_per_epoch = math.ceil(len(train_loader) / cfg.train.accum_steps)
        scheduler = build_scheduler(cfg.train, optimizer, steps_per_epoch)
        scaler = self.amp.scaler()
        ema = build_ema(cfg.train, model)

        ConsoleProgress.info("Создание папки эксперимента и сохранение конфигурации")
        run = Run.create(cfg.paths.runs_path, cfg.paths.run_name, resume=cfg.train.resume)
        plain_config = cfg.to_flat_dict()
        if cfg.dataset.protocol_path:
            protocol = EvaluationProtocol.load(cfg.dataset.protocol_path)
            plain_config.update(protocol.provenance(train_originals=cfg.dataset.train_originals))
            train_df.to_parquet(run.dir / 'training_rows.parquet', index=False)
            val_df.to_parquet(run.dir / 'development_rows.parquet', index=False)
            run.save_summary({'evaluation_role': 'development', 'protocol_digest': protocol.digest,
                              'holdout_evaluated': False, 'training_complete': False})
        run.save_snapshot(self._snapshot(plain_config, gflops))
        run.info(f"{cfg.paths.run_name}, {gflops} GFLOPS")

        run.info(f"Результаты: {run.dir.resolve()}; проверка возобновления обучения")
        state = self._resume_if_needed(
            run=run,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

        try:
            for epoch in range(state.start_epoch, cfg.train.epochs):
                train_ds.set_epoch(epoch)
                train_ds.augmentations.set_epoch(epoch)
                train_loader.sampler.generator = torch.Generator().manual_seed(cfg.seed + epoch)
                started = time.time()
                run.info(f"Полные кадры: p={train_ds.augmentations.full_frame_probability:.2f}; "
                         f"валидация: {cfg.eval.resolution}")
                run.info(f"Эпоха {epoch + 1}/{cfg.train.epochs}: обучение")
                train_result = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema,
                    amp=self.amp,
                    config=cfg,
                    device=self.device,
                    loss_fn=self.loss,
                )
                state.seen_total += train_result.seen
                run.info(f"Обучение завершено: loss={train_result.loss:.5f}, примеров={train_result.seen}; валидация EMA")
                validation = validate(ema.module, val_loader, self.amp, cfg, self.device)
                tuned = validation.tuned

                self._log_epoch(
                    run=run,
                    epoch=epoch,
                    model=model,
                    train_result=train_result,
                    tuned=tuned,
                    seen_total=state.seen_total,
                    started=started,
                    steps_per_epoch=steps_per_epoch,
                    optimizer=optimizer,
                    validation=validation,
                )

                if tuned.aic > state.best_aic:
                    state.best_aic = tuned.aic
                    state.best_result = tuned
                    run.info("Сохранение лучшего checkpoint: ckpt/best.pt")
                    run.save_state(
                        {
                            "model": model.state_dict(),
                            "ema": ema.module.state_dict(),
                            "epoch": epoch,
                            "samples": state.seen_total,
                            "best_aic": state.best_aic,
                            "operating_point": tuned.as_dict(),
                            "cfg": plain_config,
                        },
                        "best.pt",
                    )
                    run.info("Сохранение OOF-предсказаний, строк валидации и метрик")
                    run.save_eval(validation, val_df)
                    if cfg.dataset.protocol_path:
                        report = EvaluationReport(validation.accumulator, val_df, tuned)
                        report.save(run.dir / 'development')
                        run.save_summary({'development': report.summary()})
                    run.info(f"  новый лучший AIC {state.best_aic:.4f} -> ckpt/best.pt")

                run.info("Сохранение checkpoint для продолжения: ckpt/last.pt")
                run.save_state(
                    {
                        "model": model.state_dict(),
                        "ema": ema.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "samples": state.seen_total,
                        "best_aic": state.best_aic,
                        "cfg": plain_config,
                    },
                    "last.pt",
                )

            run.info("Сохранение итоговой сводки")
            self._save_final_summary(run, state, gflops)
            run.info(f"Эксперимент завершён: лучший AIC={state.best_aic:.4f}; результаты в {run.dir.resolve()}")
            return run
        finally:
            run.close()
            del model, ema, optimizer, val_loader, train_loader, scaler
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _check_resume_protocol(self) -> None:
        cfg = self.config
        run_dir = cfg.paths.runs_path / cfg.paths.run_name
        if not cfg.train.resume or not (run_dir / "ckpt" / "last.pt").exists():
            return
        if (run_dir / 'holdout_claim.json').exists():
            raise ValueError('Cannot resume a run after holdout evaluation was claimed; choose a new run_name')
        snapshot = Run.open(run_dir).snapshot
        saved_eval = snapshot.get("eval", snapshot)
        saved_dataset = snapshot.get("dataset", snapshot)
        if saved_dataset.get('protocol_path') != cfg.dataset.protocol_path:
            raise ValueError('Cannot resume with a different dataset.protocol_path; choose a new run_name')
        if cfg.dataset.protocol_path:
            protocol = EvaluationProtocol.load(cfg.dataset.protocol_path)
            protocol.verify_run(snapshot)
            if saved_dataset.get('train_originals', False) != cfg.dataset.train_originals:
                raise ValueError('Cannot resume with different train_originals')
        if saved_dataset.get("jpeg_qtable_order", "legacy_zigzag") != cfg.dataset.jpeg_qtable_order:
            raise ValueError("Cannot resume with a different dataset.jpeg_qtable_order; choose a new run_name")
        if saved_dataset.get("resize_mode", "stretch") != cfg.dataset.resize_mode:
            raise ValueError("Cannot resume with a different dataset.resize_mode; choose a new run_name")
        if saved_eval.get("resolution", "resized") != cfg.eval.resolution:
            raise ValueError("Cannot resume with a different eval.resolution; choose a new run_name")
        saved_loss = snapshot.get("loss", snapshot)
        if saved_loss.get("name", "bce_dice") != cfg.loss.name:
            raise ValueError("Cannot resume with a different loss.name; choose a new run_name")
        for key, default in (("dice_scope", "all"), ("dice_weight", 1.0)):
            if saved_loss.get(key, default) != getattr(cfg.loss, key):
                raise ValueError(f"Cannot resume with a different loss.{key}; choose a new run_name")
        saved_model = snapshot.get("model", snapshot)
        for key, default in (("aux_weight", .4), ("dct_aux_weight", 0.0)):
            if saved_model.get(key, default) != getattr(cfg.model, key):
                raise ValueError(f"Cannot resume with different loss weight model.{key}; choose a new run_name")
        saved_aug = snapshot.get("augmentation", snapshot)
        for key, default in (("full_frame_probability", 0.0),
                             ("foreground_crop_probability", 0.0),
                             ("final_full_frame_epochs", 0)):
            if saved_aug.get(key, default) != getattr(cfg.augmentation, key):
                raise ValueError(f"Cannot resume with a different augmentation.{key}; choose a new run_name")

    def _split_data(self):
        cfg = self.config
        if cfg.dataset.protocol_path:
            if cfg.eval.resolution != 'original':
                raise ValueError('Independent protocol requires original-resolution evaluation')
            protocol = EvaluationProtocol.load(cfg.dataset.protocol_path)
            development = EvaluationReport.add_jpeg_metadata(protocol.rows('development'),
                                                             self.data_workspace.train_root, cfg.train.workers)
            return protocol.rows('train', include_originals=cfg.dataset.train_originals), development
        metadata = build_metadata(self.data_workspace, workers=cfg.train.workers)
        ConsoleProgress.info(f"Метаданные готовы: {len(metadata)} строк; построение {cfg.dataset.n_folds} фолдов, val fold={cfg.dataset.fold}")
        folds = make_stratified_val_folds(
            metadata,
            n_folds=cfg.dataset.n_folds,
            seed=cfg.seed,
        )
        return train_val_fold_split(folds, fold=cfg.dataset.fold)

    def _snapshot(self, plain_config: dict, gflops: float) -> dict:
        cfg = self.config
        return {
            **plain_config,
            "arm": self.arm,
            "decoder_channels": list(cfg.model.decoder_channels),
            "fmap_ch": list(cfg.model.forensic_channels),
            "crop_scale": list(cfg.augmentation.crop_scale_range),
            "fmap_drop": [],
            "fmap_channels": FMAP_CHANNELS,
            "num_workers": cfg.train.workers,
            "probe_auc": None,
            "gflops": round(gflops, 1),
        }

    def _resume_if_needed(
        self,
        *,
        run: Run,
        model,
        ema,
        optimizer,
        scheduler,
        scaler,
    ) -> TrainingState:
        cfg = self.config
        state = TrainingState()
        if not (cfg.train.resume and (run.dir / "ckpt" / "last.pt").exists()):
            return state

        saved = run.load_state("last.pt", map_location=self.device)
        model.load_state_dict(saved["model"])
        ema.module.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if saved.get("scaler"):
            scaler.load_state_dict(saved["scaler"])

        saved_epoch = int(saved["epoch"])
        state.seen_total = int(
            saved.get(
                "samples",
                (saved_epoch + 1) * cfg.train.epoch_size,
            )
        )
        state.start_epoch = saved_epoch + 1
        state.best_aic = max(
            float(saved.get("best_aic", -1.0)),
            float(run.summary.get("best_aic", -1.0)),
        )
        run.info(
            f"ПРОДОЛЖАЮ: эпоха {state.start_epoch} из {cfg.train.epochs}, "
            f"показов {state.seen_total}, лучший AIC {state.best_aic:.4f}"
        )
        return state

    def _log_epoch(
        self,
        *,
        run: Run,
        epoch: int,
        model,
        train_result: EpochTrainResult,
        tuned: AICResult,
        seen_total: int,
        started: float,
        steps_per_epoch: int,
        optimizer,
        validation=None,
    ) -> None:
        extra = {f"train/loss_{key}": value for key, value in train_result.loss_components.items() if key != "total"}
        if validation is not None:
            extra.update({f"val/loss_{key}": value for key, value in validation.loss_components.items()})
            if validation.loss_components:
                extra["val/loss_main"] = (validation.loss_components.get("bce", 0.0)
                                          + validation.loss_components.get("dice", 0.0))
            if validation.fixed is not None:
                extra.update({"val/aic_fixed": validation.fixed.aic,
                              "val/dice_fixed": validation.fixed.dice_pos,
                              "val/fpr_fixed": validation.fixed.fpr_neg})
        run.log(
            epoch,
            {
                **extra,
                "samples": seen_total,
                "train/loss": train_result.loss,
                "train/negative_fraction": train_result.negative_fraction,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/skipped_steps": train_result.skipped_steps,
                "train/fmap_gamma": model.forensic_gate_stats()["max_abs"],
                "val/aic_tuned": tuned.aic,
                "val/dice_tuned": tuned.dice_pos,
                "val/fpr_tuned": tuned.fpr_neg,
                "val/best_thr": tuned.mask_threshold,
                "val/best_cls_thr": tuned.cls_threshold,
                "epoch_time_s": round(time.time() - started, 1),
                "gpu_gb": (
                    round(torch.cuda.max_memory_allocated() / 1e9, 2)
                    if self.device.type == "cuda"
                    else 0.0
                ),
            },
        )
        run.info(f"эпоха {epoch}: {tuned} | пропущено шагов {train_result.skipped_steps}")
        if train_result.skipped_steps > steps_per_epoch * 0.05:
            run.info("  ВНИМАНИЕ: пропущено >5% шагов - переполнения fp16, ставь amp='bf16'")

    def _save_final_summary(self, run: Run, state: TrainingState, gflops: float) -> None:
        run.save_summary(
            {
                "best_aic": state.best_aic,
                "best": (
                    state.best_result.as_dict()
                    if state.best_result is not None
                    else run.summary.get("best")
                ),
                "samples": state.seen_total,
                "gflops": round(gflops, 1),
                "within_limit": gflops <= 100,
                "validation_resolution": self.config.eval.resolution,
                "training_complete": True,
                "loss": self.config.loss.to_dict(),
            }
        )


def train_one_epoch(
    *,
    model,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    ema,
    amp: AmpContext,
    config: ExperimentConfig,
    device: torch.device,
    loss_fn: Callable[[dict, dict], LossResult] | None = None,
) -> EpochTrainResult:
    model.train()
    criterion = loss_fn or build_loss(config, aux_weight=config.model.aux_weight,
                                      dct_aux_weight=config.model.dct_aux_weight)
    skipped_steps = 0
    meter = LossMeter()
    seen = 0
    negatives = torch.zeros((), device=device)
    total_batches = len(loader) if isinstance(loader, Sized) else None
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(ConsoleProgress.iterate(
        loader, "Обучение, батчи", image_count=lambda batch: len(batch["image"])
    )):
        batch = _move_batch_to_device(batch, device)
        images = batch["image"].to(memory_format=torch.channels_last)

        with amp.autocast():
            model_kwargs = {"valid_mask": batch["valid_mask"]} if "valid_mask" in batch else {}
            loss_result = criterion(
                model(images, batch.get("fmap"), **model_kwargs),
                batch,
            )

        loss = loss_result.total
        scaler.scale(loss / config.train.accum_steps).backward()
        is_accum_boundary = (step + 1) % config.train.accum_steps == 0
        is_last_batch = total_batches is not None and (step + 1) == total_batches
        if is_accum_boundary or is_last_batch:
            if config.train.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped_steps += int(scaler.get_scale() < before)
            optimizer.zero_grad(set_to_none=True)
            ema.update_parameters(model)
            scheduler.step()

        batch_size = images.size(0)
        negatives += (batch["label"] <= 0.5).sum()
        meter.update(loss_result, batch_size)
        seen += batch_size

    components = meter.compute()
    return EpochTrainResult(
        loss=components.get("total", 0.0),
        loss_components=components,
        skipped_steps=skipped_steps,
        seen=seen,
        negative_fraction=float(negatives) / max(1, seen),
    )


def run_experiment(config: ExperimentConfig, *, arm: str = "baseline") -> Run:
    return ExperimentRunner(config, arm=arm).run()


def _move_batch_to_device(
    batch: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }



__all__ = [
    "EpochTrainResult",
    "ExperimentRunner",
    "TrainingState",
    "build_loss",
    "run_experiment",
    "train_one_epoch",
    "validate",
]

