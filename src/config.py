from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

import global_config
from src.data.augmentation.base import AugmentationConfig
from src.training.metric import DEFAULT_AREA_GRID, DEFAULT_CLS_GRID, DEFAULT_MASK_GRID


@dataclass(frozen=True)
class PathsConfig:
    data_path: Path
    runs_path: Path
    run_name: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, base_dir: Path) -> PathsConfig:
        data = _mapping(data, "paths")
        # Accept old snapshots, but resolve storage for the current machine.
        _check_keys(data, {"run_name"}, "paths", optional={"data_path", "runs_path"})
        settings = {**dotenv_values(global_config.PROJECT_ROOT / ".env"), **os.environ}
        return cls(
            data_path=_path(settings.get("AIIJC_DATA_PATH") or global_config.DATA_PATH, global_config.PROJECT_ROOT),
            runs_path=_path(settings.get("AIIJC_RUNS_PATH") or global_config.RUNS_PATH, global_config.PROJECT_ROOT),
            run_name=_non_empty_str(_required(data, "run_name", "paths"), "paths.run_name"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_path": str(self.data_path),
            "runs_path": str(self.runs_path),
            "run_name": self.run_name,
        }


@dataclass(frozen=True)
class ModelConfig:
    encoder_name: str
    decoder_channels: tuple[int, ...]
    forensic_channels: tuple[int, ...]
    aux_weight: float
    norm: str
    use_forensics: bool = True
    dct_aux_weight: float = 0.0
    decoder_name: str = "unet"
    decoder_embed_dim: int = 128
    decoder_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelConfig:
        data = _mapping(data, "model")
        _check_keys(
            data,
            {"encoder_name", "forensic_channels", "aux_weight", "norm"},
            "model",
            optional={"use_forensics", "dct_aux_weight", "decoder_name", "decoder_channels", "decoder_embed_dim", "decoder_kwargs"},
        )
        config = cls(
            encoder_name=_non_empty_str(_required(data, "encoder_name", "model"), "model.encoder_name"),
            decoder_channels=_int_tuple(data.get("decoder_channels", (128, 64, 32, 16, 16)), "model.decoder_channels"),
            forensic_channels=_int_tuple(_required(data, "forensic_channels", "model"), "model.forensic_channels"),
            aux_weight=float(_required(data, "aux_weight", "model")),
            norm=_non_empty_str(_required(data, "norm", "model"), "model.norm"),
            use_forensics=data.get("use_forensics", True),
            dct_aux_weight=float(data.get("dct_aux_weight", 0.0)),
            decoder_name=str(data.get("decoder_name", "unet")),
            decoder_embed_dim=int(data.get("decoder_embed_dim", 128)),
            decoder_kwargs=dict(_mapping(data.get("decoder_kwargs", {}), "model.decoder_kwargs")),
        )
        if not isinstance(config.use_forensics, bool):
            raise ValueError("model.use_forensics must be boolean")
        if not 0 <= config.dct_aux_weight < float("inf"):
            raise ValueError("model.dct_aux_weight must be finite and non-negative")
        if config.dct_aux_weight > 0 and not config.use_forensics:
            raise ValueError("DCT auxiliary head requires use_forensics")
        if not config.decoder_channels:
            raise ValueError("model.decoder_channels must not be empty")
        if not config.decoder_name.strip():
            raise ValueError("model.decoder_name must not be empty")
        reserved = {"encoder_channels", "encoder_strides", "norm", "use_aux"}
        if reserved.intersection(config.decoder_kwargs):
            raise ValueError("model.decoder_kwargs must not override encoder metadata, norm or use_aux")
        if config.decoder_embed_dim <= 0:
            raise ValueError("model.decoder_embed_dim must be positive")
        if len(config.forensic_channels) != 3:
            raise ValueError("model.forensic_channels must contain exactly 3 values")
        if config.aux_weight < 0:
            raise ValueError("model.aux_weight must be non-negative")
        if config.norm not in {"batch", "group"}:
            raise ValueError("model.norm must be 'batch' or 'group'")
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoder_name": self.encoder_name,
            "decoder_channels": self.decoder_channels,
            "forensic_channels": self.forensic_channels,
            "aux_weight": self.aux_weight,
            "norm": self.norm,
            "use_forensics": self.use_forensics,
            "dct_aux_weight": self.dct_aux_weight,
            "decoder_name": self.decoder_name,
            "decoder_embed_dim": self.decoder_embed_dim,
            "decoder_kwargs": dict(self.decoder_kwargs),
        }


@dataclass(frozen=True)
class DatasetConfig:
    fold: int
    n_folds: int
    image_size: int
    resize_mode: str = "stretch"
    jpeg_qtable_order: str = "legacy_zigzag"
    protocol_path: str | None = None
    train_originals: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetConfig:
        data = _mapping(data, "dataset")
        _check_keys(data, {"fold", "n_folds", "image_size"}, "dataset", optional={"resize_mode", "jpeg_qtable_order", "protocol_path", "train_originals"})
        config = cls(
            fold=int(_required(data, "fold", "dataset")),
            n_folds=int(_required(data, "n_folds", "dataset")),
            image_size=int(_required(data, "image_size", "dataset")),
            resize_mode=str(data.get("resize_mode", "stretch")),
            jpeg_qtable_order=str(data.get("jpeg_qtable_order", "legacy_zigzag")),
            protocol_path=data.get("protocol_path"),
            train_originals=data.get("train_originals", False),
        )
        if config.n_folds < 2:
            raise ValueError("dataset.n_folds must be at least 2")
        if config.protocol_path is not None and (not isinstance(config.protocol_path, str) or not config.protocol_path.strip()):
            raise ValueError('dataset.protocol_path must be a nonempty string')
        if not isinstance(config.train_originals, bool) or (config.train_originals and config.protocol_path is None):
            raise ValueError('dataset.train_originals requires a protocol_path and boolean value')
        if config.jpeg_qtable_order not in {"natural", "legacy_zigzag"}:
            raise ValueError("dataset.jpeg_qtable_order must be 'natural' or 'legacy_zigzag'")
        if config.resize_mode not in {"stretch", "letterbox"}:
            raise ValueError("dataset.resize_mode must be 'stretch' or 'letterbox'")
        if not 0 <= config.fold < config.n_folds:
            raise ValueError("dataset.fold must be in [0, n_folds)")
        if config.image_size < 8 or config.image_size % 8 != 0:
            raise ValueError("dataset.image_size must be divisible by 8 and at least 8")
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "n_folds": self.n_folds,
            "image_size": self.image_size,
            "resize_mode": self.resize_mode,
            "jpeg_qtable_order": self.jpeg_qtable_order,
            "protocol_path": self.protocol_path,
            "train_originals": self.train_originals,
        }


@dataclass(frozen=True)
class TrainConfig:
    device: str
    workers: int
    encoder_lr: float
    fmap_lr: float
    lr: float
    weight_decay: float
    epochs: int
    epoch_size: int
    val_frac: float
    val_keep_negatives: bool
    negative_fraction: float
    batch_size: int
    accum_steps: int
    warmup_frac: float
    min_lr_factor: float
    amp: str
    ema_decay: float
    grad_clip: float
    resume: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainConfig:
        data = _mapping(data, "train")
        _check_keys(
            data,
            {
                "device",
                "workers",
                "encoder_lr",
                "fmap_lr",
                "lr",
                "weight_decay",
                "epochs",
                "epoch_size",
                "val_frac",
                "val_keep_negatives",
                "negative_fraction",
                "batch_size",
                "accum_steps",
                "warmup_frac",
                "min_lr_factor",
                "amp",
                "ema_decay",
                "grad_clip",
                "resume",
            },
            "train",
        )
        config = cls(
            device=_non_empty_str(_required(data, "device", "train"), "train.device"),
            workers=int(_required(data, "workers", "train")),
            encoder_lr=float(_required(data, "encoder_lr", "train")),
            fmap_lr=float(_required(data, "fmap_lr", "train")),
            lr=float(_required(data, "lr", "train")),
            weight_decay=float(_required(data, "weight_decay", "train")),
            epochs=int(_required(data, "epochs", "train")),
            epoch_size=int(_required(data, "epoch_size", "train")),
            val_frac=float(_required(data, "val_frac", "train")),
            val_keep_negatives=bool(_required(data, "val_keep_negatives", "train")),
            negative_fraction=float(_required(data, "negative_fraction", "train")),
            batch_size=int(_required(data, "batch_size", "train")),
            accum_steps=int(_required(data, "accum_steps", "train")),
            warmup_frac=float(_required(data, "warmup_frac", "train")),
            min_lr_factor=float(_required(data, "min_lr_factor", "train")),
            amp=_non_empty_str(_required(data, "amp", "train"), "train.amp"),
            ema_decay=float(_required(data, "ema_decay", "train")),
            grad_clip=float(_required(data, "grad_clip", "train")),
            resume=bool(_required(data, "resume", "train")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not (self.device == "cpu" or self.device == "mps" or self.device.startswith("cuda")):
            raise ValueError("train.device must be 'cpu', 'mps', 'cuda' or 'cuda:<index>'")
        if self.workers < 0:
            raise ValueError("train.workers must be non-negative")
        if min(self.encoder_lr, self.fmap_lr, self.lr) <= 0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0:
            raise ValueError("train.weight_decay must be non-negative")
        if self.epochs <= 0:
            raise ValueError("train.epochs must be positive")
        if self.epoch_size <= 0:
            raise ValueError("train.epoch_size must be positive")
        if not 0.0 <= self.val_frac <= 1.0:
            raise ValueError("train.val_frac must be in [0, 1]")
        if not 0.0 < self.negative_fraction < 1.0:
            raise ValueError("train.negative_fraction must be in (0, 1)")
        if self.batch_size <= 0:
            raise ValueError("train.batch_size must be positive")
        if self.accum_steps <= 0:
            raise ValueError("train.accum_steps must be positive")
        if not 0.0 <= self.warmup_frac <= 1.0:
            raise ValueError("train.warmup_frac must be in [0, 1]")
        if not 0.0 <= self.min_lr_factor <= 1.0:
            raise ValueError("train.min_lr_factor must be in [0, 1]")
        if self.amp not in {"off", "fp16", "bf16"}:
            raise ValueError("train.amp must be 'off', 'fp16' or 'bf16'")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("train.ema_decay must be in [0, 1)")
        if self.grad_clip < 0:
            raise ValueError("train.grad_clip must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "workers": self.workers,
            "encoder_lr": self.encoder_lr,
            "fmap_lr": self.fmap_lr,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "epoch_size": self.epoch_size,
            "val_frac": self.val_frac,
            "val_keep_negatives": self.val_keep_negatives,
            "negative_fraction": self.negative_fraction,
            "batch_size": self.batch_size,
            "accum_steps": self.accum_steps,
            "warmup_frac": self.warmup_frac,
            "min_lr_factor": self.min_lr_factor,
            "amp": self.amp,
            "ema_decay": self.ema_decay,
            "grad_clip": self.grad_clip,
            "resume": self.resume,
        }


@dataclass(frozen=True)
class EvalConfig:
    n_bins: int = 256
    mask_thresholds: tuple[float, ...] = DEFAULT_MASK_GRID
    cls_thresholds: tuple[float, ...] = DEFAULT_CLS_GRID
    min_areas: tuple[float, ...] = DEFAULT_AREA_GRID
    resolution: str = "resized"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> EvalConfig:
        defaults = cls()
        if data is None:
            return defaults
        data = _mapping(data, "eval")
        _check_keys(data, {"n_bins", "mask_thresholds", "cls_thresholds", "min_areas"}, "eval",
                    optional={"resolution"})
        config = cls(
            n_bins=int(data.get("n_bins", defaults.n_bins)),
            mask_thresholds=_float_tuple(data.get("mask_thresholds", defaults.mask_thresholds), "eval.mask_thresholds"),
            cls_thresholds=_float_tuple(data.get("cls_thresholds", defaults.cls_thresholds), "eval.cls_thresholds"),
            min_areas=_float_tuple(data.get("min_areas", defaults.min_areas), "eval.min_areas"),
            resolution=str(data.get("resolution", defaults.resolution)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.n_bins <= 0:
            raise ValueError("eval.n_bins must be positive")
        if self.resolution not in {"resized", "original"}:
            raise ValueError("eval.resolution must be 'resized' or 'original'")
        _check_probability_grid(self.mask_thresholds, "eval.mask_thresholds")
        _check_probability_grid(self.cls_thresholds, "eval.cls_thresholds")
        _check_probability_grid(self.min_areas, "eval.min_areas")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_bins": self.n_bins,
            "mask_thresholds": self.mask_thresholds,
            "cls_thresholds": self.cls_thresholds,
            "min_areas": self.min_areas,
            "resolution": self.resolution,
        }


@dataclass(frozen=True)
class LossConfig:
    """Выбор цели обучения из реестра `src.losses` и её параметры.

    `dice_scope`/`dice_weight` относятся только к базовой цели `bce_dice` —
    missing settings preserve historical runs, new ablations opt in explicitly.
    Остальные цели реестра получают собственные параметры через `kwargs`.
    `aux_weight`/`dct_aux_weight` сюда не входят: они же включают
    соответствующие головы в декодере, поэтому остаются в `model` — иначе
    конфиг мог бы попросить глубокую супервизию у модели, собранной без неё.
    """

    name: str = "bce_dice"
    dice_scope: str = "all"
    dice_weight: float = 1.0
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> LossConfig:
        defaults = cls()
        if data is None:
            data = {}
        data = _mapping(data, "loss")
        _check_keys(data, set(), "loss", optional={"name", "dice_scope", "dice_weight", "kwargs"})
        config = cls(
            name=_non_empty_str(data.get("name", defaults.name), "loss.name"),
            dice_scope=str(data.get("dice_scope", defaults.dice_scope)),
            dice_weight=float(data.get("dice_weight", defaults.dice_weight)),
            kwargs=dict(_mapping(data.get("kwargs", {}), "loss.kwargs")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.dice_scope not in {"all", "positive"}:
            raise ValueError("loss.dice_scope must be 'all' or 'positive'")
        if not 0 <= self.dice_weight < float("inf"):
            raise ValueError("loss.dice_weight must be finite and non-negative")
        reserved = {"cls_weight", "aux_weight", "dct_aux_weight"}.intersection(self.kwargs)
        if reserved:
            raise ValueError("loss.kwargs must not override cls_weight or aux_weight or dct_aux_weight")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dice_scope": self.dice_scope,
            "dice_weight": self.dice_weight,
            "kwargs": dict(self.kwargs),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathsConfig
    seed: int
    model: ModelConfig
    augmentation: AugmentationConfig
    dataset: DatasetConfig
    train: TrainConfig
    eval: EvalConfig
    loss: LossConfig = field(default_factory=LossConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, base_dir: str | Path = ".") -> ExperimentConfig:
        data = _mapping(data, "experiment")
        _check_keys(data, {"paths", "seed", "model", "augmentation", "dataset", "train", "eval"}, "experiment", optional={"loss"})
        base_dir = Path(base_dir)
        config = cls(
            paths=PathsConfig.from_dict(_required(data, "paths", "experiment"), base_dir=base_dir),
            seed=int(_required(data, "seed", "experiment")),
            model=ModelConfig.from_dict(_required(data, "model", "experiment")),
            augmentation=_augmentation_from_dict(_required(data, "augmentation", "experiment")),
            dataset=DatasetConfig.from_dict(_required(data, "dataset", "experiment")),
            train=TrainConfig.from_dict(_required(data, "train", "experiment")),
            eval=EvalConfig.from_dict(data.get("eval")),
            loss=LossConfig.from_dict(data.get("loss")),
        )
        if config.seed < 0:
            raise ValueError("seed must be non-negative")
        if config.augmentation.final_full_frame_epochs > config.train.epochs:
            raise ValueError("augmentation.final_full_frame_epochs must not exceed train.epochs")
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "paths": self.paths.to_dict(),
            "seed": self.seed,
            "model": self.model.to_dict(),
            "augmentation": _augmentation_to_dict(self.augmentation),
            "dataset": self.dataset.to_dict(),
            "train": self.train.to_dict(),
            "eval": self.eval.to_dict(),
            "loss": self.loss.to_dict(),
        }

    def to_flat_dict(self) -> dict[str, Any]:
        plain = {
            "data_path": str(self.paths.data_path),
            "runs_path": str(self.paths.runs_path),
            "run_name": self.paths.run_name,
            "seed": self.seed,
        }
        for section in (self.model, self.augmentation, self.dataset, self.train, self.eval, self.loss):
            if isinstance(section, AugmentationConfig):
                plain.update(_augmentation_to_dict(section))
            else:
                plain.update(section.to_dict())
        # Ключи цели обучения ("name") слишком общие для плоского снимка.
        plain.update({f"loss_{key}": value for key, value in self.loss.to_dict().items()})
        return plain


class _ConfigLoader:
    """Resolve single-parent YAML inheritance before validating an experiment."""

    def load(self, path: Path, chain: tuple[Path, ...] = ()) -> dict[str, Any]:
        path = path.resolve()
        if path in chain:
            names = " -> ".join(str(item) for item in (*chain, path))
            raise ValueError(f"Cyclic config inheritance: {names}")
        with path.open("r", encoding="utf-8") as fh:
            data = dict(_mapping(yaml.safe_load(fh), str(path)))
        if "extends" not in data:
            return data
        parent = data.pop("extends")
        if not isinstance(parent, str) or not parent.strip():
            raise ValueError(f"extends in {path} must be a non-empty string")
        inherited = self.load(path.parent / parent, (*chain, path))
        return self._merge(inherited, data)

    def _merge(self, base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in overrides.items():
            if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
                merged[key] = self._merge(merged[key], value)
            else:
                merged[key] = value
        return merged


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load YAML, recursively merging an optional relative or absolute `extends`."""
    path = Path(path)
    return ExperimentConfig.from_dict(_ConfigLoader().load(path), base_dir=path.parent)


def _augmentation_from_dict(data: Mapping[str, Any]) -> AugmentationConfig:
    data = _mapping(data, "augmentation")
    _check_keys(
        data,
        {
            "crop_scale_range",
            "jpeg_recompression_probability",
            "jpeg_recompression_quality_range",
            "full_frame",
        },
        "augmentation",
        optional={"full_frame_probability", "foreground_crop_probability", "final_full_frame_epochs"},
    )
    config = AugmentationConfig(
        crop_scale_range=_float_tuple(
            _required(data, "crop_scale_range", "augmentation"),
            "augmentation.crop_scale_range",
        ),
        jpeg_recompression_probability=float(
            _required(data, "jpeg_recompression_probability", "augmentation")
        ),
        jpeg_recompression_quality_range=_int_tuple(
            _required(data, "jpeg_recompression_quality_range", "augmentation"),
            "augmentation.jpeg_recompression_quality_range",
        ),
        full_frame=bool(_required(data, "full_frame", "augmentation")),
        full_frame_probability=float(data.get("full_frame_probability", 0.0)),
        foreground_crop_probability=float(data.get("foreground_crop_probability", 0.0)),
        final_full_frame_epochs=int(data.get("final_full_frame_epochs", 0)),
    )
    if len(config.crop_scale_range) != 2:
        raise ValueError("augmentation.crop_scale_range must contain exactly 2 values")
    crop_min, crop_max = config.crop_scale_range
    if not 0.0 < crop_min <= crop_max <= 1.0:
        raise ValueError("augmentation.crop_scale_range must satisfy 0 < min <= max <= 1")
    if not 0.0 <= config.jpeg_recompression_probability <= 1.0:
        raise ValueError("augmentation.jpeg_recompression_probability must be in [0, 1]")
    if len(config.jpeg_recompression_quality_range) != 2:
        raise ValueError("augmentation.jpeg_recompression_quality_range must contain exactly 2 values")
    quality_min, quality_max = config.jpeg_recompression_quality_range
    if not 1 <= quality_min <= quality_max <= 100:
        raise ValueError("augmentation.jpeg_recompression_quality_range must satisfy 1 <= min <= max <= 100")
    return config


def _augmentation_to_dict(config: AugmentationConfig) -> dict[str, Any]:
    return {
        "crop_scale_range": config.crop_scale_range,
        "jpeg_recompression_probability": config.jpeg_recompression_probability,
        "jpeg_recompression_quality_range": config.jpeg_recompression_quality_range,
        "full_frame": config.full_frame,
        "full_frame_probability": config.full_frame_probability,
        "foreground_crop_probability": config.foreground_crop_probability,
        "final_full_frame_epochs": config.final_full_frame_epochs,
    }


def _mapping(data: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{section} must be a mapping")
    return data


def _check_keys(data: Mapping[str, Any], allowed: set[str], section: str,
                *, optional: set[str] = frozenset()) -> None:
    unknown = sorted(set(data) - allowed - optional)
    if unknown:
        raise ValueError(f"unknown keys in {section}: {', '.join(unknown)}")
    missing = sorted(allowed - set(data))
    if missing:
        raise ValueError(f"missing keys in {section}: {', '.join(missing)}")


def _required(data: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in data:
        raise ValueError(f"missing key in {section}: {key}")
    return data[key]


def _non_empty_str(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _path(value: Any, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _sequence(value, name))


def _float_tuple(value: Any, name: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _sequence(value, name))


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(value)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _check_probability_grid(values: tuple[float, ...], name: str) -> None:
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"{name} values must be in [0, 1]")


__all__ = [
    "AugmentationConfig",
    "DatasetConfig",
    "EvalConfig",
    "ExperimentConfig",
    "LossConfig",
    "ModelConfig",
    "PathsConfig",
    "TrainConfig",
    "load_experiment_config",
]
