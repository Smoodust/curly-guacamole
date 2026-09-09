from collections.abc import Callable
from fnmatch import fnmatchcase

_LOSSES: dict[str, Callable[..., object]] = {}


def register_loss(name: str):
    """Register a loss class or factory under an explicit experiment name."""
    if not isinstance(name, str) or not name or name.strip() != name:
        raise ValueError('Loss name must be a nonempty string without surrounding whitespace')

    def register(factory):
        if name in _LOSSES:
            raise ValueError(f'Loss {name!r} is already registered')
        _LOSSES[name] = factory
        return factory

    return register


def list_losses(filter: str = '*') -> list[str]:
    """List registered names in alphabetical order, optionally using a glob."""
    return sorted(name for name in _LOSSES if fnmatchcase(name, filter))


def is_loss(name: str) -> bool:
    return name in _LOSSES


def create_loss(name: str, *, aux_weight: float = 0.0, dct_aux_weight: float = 0.0, **kwargs):
    """Build a registered loss; unsupported options raise TypeError."""
    if not is_loss(name):
        raise ValueError(f'Unknown loss {name!r}. Available: {", ".join(list_losses())}')
    return _LOSSES[name](aux_weight=aux_weight, dct_aux_weight=dct_aux_weight, **kwargs)


def build_loss(config, *, aux_weight: float | None = None, dct_aux_weight: float | None = None):
    """Собрать цель из `ExperimentConfig`; вес aux/DCT-головы по умолчанию берётся из модели.

    `dice_scope`/`dice_weight` уходят в `kwargs` только для `bce_dice` — у
    остальных целей реестра их нет, и передавать всегда лишний ключ незачем.
    """
    loss_cfg = config.loss
    kwargs = dict(loss_cfg.kwargs)
    if loss_cfg.name == "bce_dice":
        kwargs.setdefault("dice_scope", loss_cfg.dice_scope)
        kwargs.setdefault("dice_weight", loss_cfg.dice_weight)
    resolved_aux = config.model.aux_weight if aux_weight is None else aux_weight
    resolved_dct = config.model.dct_aux_weight if dct_aux_weight is None else dct_aux_weight
    return create_loss(loss_cfg.name, aux_weight=resolved_aux, dct_aux_weight=resolved_dct, **kwargs)


__all__ = ['build_loss', 'create_loss', 'is_loss', 'list_losses', 'register_loss']
