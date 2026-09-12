from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from src.losses.mask_batch import EPS

LOG2 = math.log(2.0)


SOFT_FALSE_POSITIVE_MODES = ("sigmoid", "rational", "hinge")


def soft_false_positive(
    area: Tensor,
    *,
    threshold: float = 0.01,
    mode: str = "sigmoid",
    softness: float = 0.5,
) -> Tensor:
    """Дифференцируемая замена индикатора `1[area >= threshold]` из метрики.

    `sigmoid` — сигмоида по логарифму площади: почти ноль заметно ниже порога,
    0.5 на пороге, не больше 1 сверху. Далеко над порогом градиент затухает,
    поэтому рядом должно быть слагаемое, которое работает и там (BCE).
    `rational` — `area / (area + threshold)`: те же границы, 0.5 на пороге, но
    градиент не исчезает ни на какой площади, зато и ниже порога уже не ноль.
    `hinge` — softplus: точнее всех повторяет форму метрики (ровно 1 на пороге,
    линейный рост дальше), но не ограничен сверху: на случайной инициализации
    площадь около 0.5 даёт значение под 140, и слагаемое подминает под себя
    весь градиент. Пригоден только с маленьким весом.
    """
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    if softness <= 0.0:
        raise ValueError("softness must be positive")
    if mode == "rational":
        return area / (area + threshold)
    if mode == "sigmoid":
        return torch.sigmoid(torch.log(area.clamp_min(EPS) / threshold) / softness)
    if mode == "hinge":
        return F.softplus((area - threshold) / (threshold * softness)) / LOG2
    raise ValueError(
        f"unknown soft false positive mode {mode!r}; use one of {SOFT_FALSE_POSITIVE_MODES}"
    )


def harmonic_aic(dice: Tensor, false_positive_rate: Tensor) -> Tensor:
    """Свёртка метрики: гармоническое среднее Dice и (1 - FPR)."""
    specificity = 1.0 - false_positive_rate
    return 2.0 * dice * specificity / (dice + specificity + EPS)


__all__ = [
    "SOFT_FALSE_POSITIVE_MODES",
    "harmonic_aic",
    "soft_false_positive",
]
