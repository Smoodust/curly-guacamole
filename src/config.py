from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

import global_config
from src.data.augmentation.base import AugmentationConfig
from src.training.metric import DEFAULT_MASK_GRID

PIPELINE_VERSION = 'emcad_v1'
DEFAULT_PROTOCOL_PATH = 'runs/validation_protocol_20260908/protocol'


class ConfigSection:
    """Strict keys with dataclass defaults; snapshots contain resolved values."""

    @classmethod
    def from_dict(cls, data):
        data = _mapping(data, cls.__name__)
        section = cls.__name__.removesuffix('Config').lower()
        _check_keys(data, set(), section, optional={f.name for f in fields(cls)})
        return cls(**_float_fields(cls, data))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PathsConfig(ConfigSection):
    data_path: Path
    runs_path: Path
    run_name: str

    @staticmethod
    def current_data_path() -> Path:
        """Resolve data on this machine, independent of saved run paths."""
        settings = {**dotenv_values(global_config.PROJECT_ROOT / '.env'), **os.environ}
        return _path(settings.get('AIIJC_DATA_PATH') or global_config.DATA_PATH,
                     global_config.PROJECT_ROOT)

    @classmethod
    def from_dict(cls, data, *, base_dir=Path('.')):
        data = _mapping(data, 'paths')
        _check_keys(data, {'run_name'}, 'paths', optional={'data_path', 'runs_path'})
        settings = {**dotenv_values(global_config.PROJECT_ROOT / '.env'), **os.environ}
        return cls(
            cls.current_data_path(),
            _path(settings.get('AIIJC_RUNS_PATH') or global_config.RUNS_PATH, global_config.PROJECT_ROOT),
            _non_empty_str(data['run_name'], 'paths.run_name'),
        )

    def to_dict(self):
        return dict(data_path=str(self.data_path), runs_path=str(self.runs_path), run_name=self.run_name)


@dataclass(frozen=True)
class ModelConfig(ConfigSection):
    encoder_name: str = 'pvt_v2_b2'
    forensic_channels: tuple[int, ...] = (64, 96, 128)
    aux_weight: float = 0.4
    norm: str = 'batch'
    use_forensics: bool = True
    forensic_mode: str = 'maps'
    jpeg_pretrained: str | None = None
    jpeg_variant: str = 'baseline'
    fusion_variant: str = 'baseline'
    dct_aux_weight: float = 0.0
    decoder_kwargs: dict[str, Any] = field(default_factory=dict)
    local_image_size: int = 0
    luma_image_size: int = 0
    strided_resize: bool = False
    resize_variant: str = 'linear'

    def __post_init__(self):
        _non_empty_str(self.encoder_name, 'model.encoder_name')
        if type(self.strided_resize) is not bool:
            raise ValueError('model.strided_resize must be boolean')
        if self.resize_variant not in {'linear', 'nonlinear', 'residual_paper', 'residual_compact'}:
            raise ValueError('unknown model.resize_variant')
        if self.resize_variant != 'linear' and not self.strided_resize:
            raise ValueError('resize_variant requires strided_resize')
        if self.strided_resize and (self.use_forensics or self.local_image_size or self.luma_image_size):
            raise ValueError('strided_resize requires RGB-only without local/luma branches')
        object.__setattr__(self, 'forensic_channels', tuple(self.forensic_channels))
        if len(self.forensic_channels) != 3 or any(type(c) is not int or c <= 0 for c in self.forensic_channels):
            raise ValueError('model.forensic_channels must contain three positive integers')
        if self.norm not in {'batch', 'group'}:
            raise ValueError("model.norm must be 'batch' or 'group'")
        if type(self.use_forensics) is not bool:
            raise ValueError('model.use_forensics must be boolean')
        for name in ('aux_weight', 'dct_aux_weight'):
            _nonnegative(getattr(self, name), f'model.{name}')
        if self.forensic_mode not in {'maps', 'jpeg'}:
            raise ValueError('forensic_mode must be maps or jpeg')
        if self.forensic_mode == 'jpeg' and not self.use_forensics:
            raise ValueError('forensic_mode=jpeg requires use_forensics')
        if self.jpeg_variant not in {'baseline', 'signed', 'attention', 'subblock4'}:
            raise ValueError('unknown model.jpeg_variant')
        if self.fusion_variant not in {'baseline', 'local', 'spatial', 'channel_spatial', 'film', 'cross_attention'}:
            raise ValueError('unknown model.fusion_variant')
        if self.fusion_variant != 'baseline' and self.forensic_mode != 'jpeg':
            raise ValueError('fusion_variant requires forensic_mode=jpeg')
        if self.jpeg_variant != 'baseline' and self.forensic_mode != 'jpeg':
            raise ValueError('jpeg_variant requires forensic_mode=jpeg')
        if self.jpeg_pretrained is not None:
            _non_empty_str(self.jpeg_pretrained, 'model.jpeg_pretrained')
            if self.forensic_mode != 'jpeg':
                raise ValueError('jpeg_pretrained requires forensic_mode=jpeg')
        if self.dct_aux_weight and not self.use_forensics:
            raise ValueError('DCT auxiliary head requires use_forensics')
        if type(self.local_image_size) is not int or self.local_image_size < 0 or self.local_image_size % 32:
            raise ValueError('model.local_image_size must be 0 or a positive multiple of 32')
        if type(self.luma_image_size) is not int or self.luma_image_size < 0 or self.luma_image_size % 32:
            raise ValueError('model.luma_image_size must be 0 or a positive multiple of 32')
        if self.luma_image_size and self.local_image_size:
            raise ValueError('luma_image_size and local_image_size are mutually exclusive')
        _mapping(self.decoder_kwargs, 'model.decoder_kwargs')
        reserved = {'encoder_channels', 'encoder_strides', 'norm', 'use_aux'}
        if reserved.intersection(self.decoder_kwargs):
            raise ValueError('model.decoder_kwargs must not override encoder metadata, norm or use_aux')
        if self.local_image_size and any(self.decoder_kwargs.get(k, 0) for k in
                                        ('rgb_refinement_channels', 'output_refinement_channels')):
            raise ValueError('local_image_size must not be combined with decoder refinement experiments')
        if self.luma_image_size and any(self.decoder_kwargs.get(k, 0) for k in
                                       ('rgb_refinement_channels', 'output_refinement_channels')):
            raise ValueError('luma_image_size must not be combined with decoder refinement experiments')


@dataclass(frozen=True)
class DatasetConfig(ConfigSection):
    image_size: int = 640
    resize_mode: str = 'stretch'
    # A storage location, never an enable/disable switch. The manifest is mandatory.
    protocol_path: str = DEFAULT_PROTOCOL_PATH

    def __post_init__(self):
        if type(self.image_size) is not int or self.image_size < 8 or self.image_size % 8:
            raise ValueError('dataset.image_size must be divisible by 8 and at least 8')
        if self.resize_mode not in {'stretch', 'letterbox'}:
            raise ValueError("dataset.resize_mode must be 'stretch' or 'letterbox'")
        _non_empty_str(self.protocol_path, 'dataset.protocol_path')


@dataclass(frozen=True)
class TrainConfig(ConfigSection):
    device: str = 'cuda'
    workers: int = 10
    encoder_lr: float = 1e-4
    fmap_lr: float = 3e-4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 6
    epoch_size: int = 24000
    full_train_epochs: int = 0
    negative_fraction: float = .25
    sampling_strategy: str = 'negative_fraction'
    batch_size: int = 4
    accum_steps: int = 4
    warmup_frac: float = .05
    min_lr_factor: float = .02
    amp: str = 'bf16'
    ema_decay: float = .999
    grad_clip: float = 1.0
    resume: bool = False
    # Model weights only; relative paths are resolved against paths.runs_path.
    finetune_from: str | None = None

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not isinstance(self.device, str) or not (self.device in {'cpu', 'mps'} or self.device.startswith('cuda')):
            raise ValueError("train.device must be 'cpu', 'mps', 'cuda' or 'cuda:<index>'")
        for name in ('epochs', 'epoch_size', 'batch_size', 'accum_steps'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'train.{name} must be a positive integer')
        if type(self.workers) is not int or self.workers < 0:
            raise ValueError('train.workers must be a nonnegative integer')
        if type(self.full_train_epochs) is not int or not 0 <= self.full_train_epochs <= self.epochs:
            raise ValueError('train.full_train_epochs must be an integer in [0, epochs]')
        for name in ('encoder_lr', 'fmap_lr', 'lr'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'train.{name} must be finite and positive')
        for name in ('weight_decay', 'grad_clip'):
            _nonnegative(getattr(self, name), f'train.{name}')
        for name in ('warmup_frac', 'min_lr_factor'):
            _check_probability_grid((getattr(self, name),), f'train.{name}')
        if not 0 < self.negative_fraction < 1:
            raise ValueError('train.negative_fraction must be in (0, 1)')
        if self.sampling_strategy not in {'negative_fraction', 'uniform_mask_area'}:
            raise ValueError('train.sampling_strategy must be negative_fraction or uniform_mask_area')
        if not 0 <= self.ema_decay < 1:
            raise ValueError('train.ema_decay must be in [0, 1)')
        if self.amp not in {'off', 'fp16', 'bf16'}:
            raise ValueError("train.amp must be 'off', 'fp16' or 'bf16'")
        if type(self.resume) is not bool:
            raise ValueError('train.resume must be boolean')
        if self.finetune_from is not None:
            _non_empty_str(self.finetune_from, 'train.finetune_from')


@dataclass(frozen=True)
class EvalConfig(ConfigSection):
    n_bins: int = 256
    small_mask_weight: float = 1.0
    mask_thresholds: tuple[float, ...] = DEFAULT_MASK_GRID
    cls_thresholds: tuple[float, ...] = (0., .2, .4, .5, .6, .7, .8, .9, .95)
    min_areas: tuple[float, ...] = (0.,)

    def __post_init__(self):
        if isinstance(self.small_mask_weight, bool) or not math.isfinite(self.small_mask_weight) or self.small_mask_weight <= 0:
            raise ValueError('eval.small_mask_weight must be finite and positive')
        if type(self.n_bins) is not int or self.n_bins <= 0:
            raise ValueError('eval.n_bins must be a positive integer')
        for name in ('mask_thresholds', 'cls_thresholds', 'min_areas'):
            values = tuple(float(value) for value in getattr(self, name))
            object.__setattr__(self, name, values)
            _check_probability_grid(values, f'eval.{name}')


@dataclass(frozen=True)
class LossConfig(ConfigSection):
    dice_scope: str = 'all'
    dice_weight: float = 1.0
    pixel_loss: str = 'bce'
    focal_gamma: float = 1.0
    dice_area_reference: float = 0.0
    dice_area_max_weight: float = 3.0
    boundary_weight: float = 0.0
    aux_loss_weight: float | None = None

    def __post_init__(self):
        if self.dice_scope not in {'all', 'positive'}:
            raise ValueError("loss.dice_scope must be 'all' or 'positive'")
        _nonnegative(self.dice_weight, 'loss.dice_weight')
        if self.pixel_loss not in {'bce', 'focal'}:
            raise ValueError("loss.pixel_loss must be 'bce' or 'focal'")
        for name in ('focal_gamma', 'boundary_weight', 'dice_area_reference', 'dice_area_max_weight'):
            _nonnegative(getattr(self, name), f'loss.{name}')
        if self.dice_area_reference > 1 or self.dice_area_max_weight < 1:
            raise ValueError('loss area reference must be <= 1 and maximum weight >= 1')
        if self.aux_loss_weight is not None:
            _nonnegative(self.aux_loss_weight, 'loss.aux_loss_weight')


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathsConfig
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    @classmethod
    def from_dict(cls, data, *, base_dir='.'):
        data = _mapping(data, 'experiment')
        _check_keys(data, {'paths'}, 'experiment', optional={
            'seed', 'model', 'augmentation', 'dataset', 'train', 'eval', 'loss', 'pipeline_version'})
        if data.get('pipeline_version', PIPELINE_VERSION) != PIPELINE_VERSION:
            raise ValueError('Unsupported pipeline_version')
        return cls(
            paths=PathsConfig.from_dict(data['paths'], base_dir=Path(base_dir)),
            seed=data.get('seed', 42),
            model=ModelConfig.from_dict(data.get('model', {})),
            augmentation=_augmentation_from_dict(data.get('augmentation', {})),
            dataset=DatasetConfig.from_dict(data.get('dataset', {})),
            train=TrainConfig.from_dict(data.get('train', {})),
            eval=EvalConfig.from_dict(data.get('eval', {})),
            loss=LossConfig.from_dict(data.get('loss', {})),
        )

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if self.augmentation.final_full_frame_epochs > self.train.epochs:
            raise ValueError('augmentation.final_full_frame_epochs must not exceed train.epochs')
        if self.train.full_train_epochs > self.augmentation.final_full_frame_epochs:
            raise ValueError('train.full_train_epochs requires matching final_full_frame_epochs')
        if self.model.local_image_size and self.dataset.resize_mode != 'stretch':
            raise ValueError('local_image_size currently requires stretch geometry')
        if self.model.forensic_mode == 'jpeg' and self.dataset.resize_mode != 'stretch':
            raise ValueError('forensic_mode=jpeg requires stretch geometry')
        if self.model.luma_image_size and self.dataset.resize_mode != 'stretch':
            raise ValueError('luma_image_size currently requires stretch geometry')
        if self.model.strided_resize and self.dataset.resize_mode != 'stretch':
            raise ValueError('strided_resize requires stretch geometry')

    def to_dict(self):
        result = {'pipeline_version': PIPELINE_VERSION, 'paths': self.paths.to_dict(), 'seed': self.seed}
        for name in ('model', 'dataset', 'train', 'eval', 'loss'):
            result[name] = getattr(self, name).to_dict()
        result['augmentation'] = asdict(self.augmentation)
        return result

    def to_flat_dict(self):
        plain = {'pipeline_version': PIPELINE_VERSION, **self.paths.to_dict(), 'seed': self.seed}
        for name in ('model', 'dataset', 'train', 'eval', 'loss'):
            plain.update(getattr(self, name).to_dict())
        plain.update(asdict(self.augmentation))
        return plain


class _ConfigLoader:
    """Single-parent YAML inheritance; resolved snapshots have no parent dependency."""

    def load(self, path: Path, chain=()):
        path = path.resolve()
        if path in chain:
            raise ValueError('Cyclic config inheritance: ' + ' -> '.join(map(str, (*chain, path))))
        with path.open(encoding='utf-8') as stream:
            data = dict(_mapping(yaml.safe_load(stream), str(path)))
        if 'extends' not in data:
            return data
        parent = data.pop('extends')
        if not isinstance(parent, str) or not parent.strip():
            raise ValueError(f'extends in {path} must be a non-empty string')
        return self._merge(self.load(path.parent / parent, (*chain, path)), data)

    def _merge(self, base, overrides):
        merged = dict(base)
        for key, value in overrides.items():
            merged[key] = self._merge(merged[key], value) if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping) else value
        return merged


class RuntimeEnvironment:
    """Machine overrides for recipes; snapshot deserialization stays unchanged."""

    FIELDS = {'batch_size': int, 'accum_steps': int, 'amp': str,
              'device': str, 'workers': int}

    def apply(self, data):
        settings = {**dotenv_values(global_config.PROJECT_ROOT / '.env'), **os.environ}
        train = dict(_mapping(data.get('train', {}), 'train'))
        for name, cast in self.FIELDS.items():
            key = 'AIIJC_' + name.upper()
            value = settings.get(key)
            if value is None or not value.strip():
                continue
            try:
                train[name] = cast(value.strip())
            except ValueError as exc:
                raise ValueError(f'{key} must be a valid {cast.__name__}') from exc
        return {**data, 'train': train}


def load_experiment_config(path):
    path = Path(path)
    data = RuntimeEnvironment().apply(_ConfigLoader().load(path))
    return ExperimentConfig.from_dict(data, base_dir=path.parent)


def _augmentation_from_dict(data):
    data = dict(_mapping(data, 'augmentation'))
    _check_keys(data, set(), 'augmentation', optional={f.name for f in fields(AugmentationConfig)})
    data = _float_fields(AugmentationConfig, data)
    for name in ('crop_scale_range', 'jpeg_recompression_quality_range'):
        if name in data:
            cast = float if name == 'crop_scale_range' else int
            data[name] = tuple(cast(value) for value in data[name])
    config = AugmentationConfig(**data)
    if len(config.crop_scale_range) != 2 or not 0 < config.crop_scale_range[0] <= config.crop_scale_range[1] <= 1:
        raise ValueError('augmentation.crop_scale_range must satisfy 0 < min <= max <= 1')
    if len(config.jpeg_recompression_quality_range) != 2 or not 1 <= config.jpeg_recompression_quality_range[0] <= config.jpeg_recompression_quality_range[1] <= 100:
        raise ValueError('augmentation.jpeg_recompression_quality_range must satisfy 1 <= min <= max <= 100')
    _check_probability_grid((config.jpeg_recompression_probability,), 'augmentation.jpeg_recompression_probability')
    return config


def _float_fields(cls, data):
    """PyYAML can read scientific notation such as 1e-4 as a string."""
    result = dict(data)
    for item in fields(cls):
        if item.name in result and item.type in (float, 'float'):
            result[item.name] = float(result[item.name])
    return result


def _mapping(data, section):
    if not isinstance(data, Mapping):
        raise TypeError(f'{section} must be a mapping')
    return data


def _check_keys(data, allowed, section, *, optional=frozenset()):
    unknown = sorted(set(data) - allowed - optional)
    if unknown:
        raise ValueError(f"unknown keys in {section}: {', '.join(unknown)}")
    missing = sorted(allowed - set(data))
    if missing:
        raise ValueError(f"missing keys in {section}: {', '.join(missing)}")


def _non_empty_str(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value.strip()


def _nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and non-negative')


def _path(value, base_dir):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def _check_probability_grid(values, name):
    if not values or any(not 0 <= v <= 1 for v in values):
        raise ValueError(f'{name} must contain probabilities in [0, 1]')


__all__ = ['AugmentationConfig', 'DatasetConfig', 'EvalConfig', 'ExperimentConfig', 'ModelConfig',
           'LossConfig', 'PathsConfig', 'TrainConfig', 'load_experiment_config', 'PIPELINE_VERSION']
