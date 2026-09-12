from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-6


class MaskBatch:
    """Одна пачка масок, разложенная по картинкам: (B, N) в float32.

    Метрика AIC усредняется по кадрам, а не по пикселям, и по-разному считает
    позитивы и негативы. Поэтому все величины здесь возвращаются вектором длины
    B: функция потерь сама решает, какие кадры и с каким весом складывать.
    """

    def __init__(self, logits: Tensor, targets: Tensor, valid_mask: Tensor | None = None) -> None:
        self.logits = logits.float().flatten(1)
        self.probs = torch.sigmoid(self.logits)
        self.targets = targets.float().flatten(1)
        self.valid = (
            torch.ones_like(self.probs) if valid_mask is None else valid_mask.float().flatten(1)
        )

    @property
    def valid_pixels(self) -> Tensor:
        return self.valid.sum(1).clamp_min(1.0)

    @property
    def target_area(self) -> Tensor:
        """Доля кадра, занятая правкой в разметке."""
        return (self.targets * self.valid).sum(1) / self.valid_pixels

    @property
    def predicted_area(self) -> Tensor:
        """Мягкая площадь предсказания: ожидаемая доля кадра под сигмоидой."""
        return (self.probs * self.valid).sum(1) / self.valid_pixels

    @property
    def is_positive(self) -> Tensor:
        """1.0 для кадров с правкой в разметке — как делит кадры сама метрика."""
        return ((self.targets * self.valid).sum(1) > 0).float()

    def dice(self, smooth: float = 1.0) -> Tensor:
        """Мягкий Dice по каждому кадру; чем больше, тем лучше."""
        probs, targets = self.probs * self.valid, self.targets * self.valid
        intersection = (probs * targets).sum(1)
        return (2.0 * intersection + smooth) / (probs.sum(1) + targets.sum(1) + smooth)

    def tversky(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0) -> Tensor:
        """Dice с раздельными весами ложных срабатываний (alpha) и пропусков (beta)."""
        probs, targets = self.probs * self.valid, self.targets * self.valid
        true_positive = (probs * targets).sum(1)
        false_positive = (probs * (self.valid - targets)).sum(1)
        false_negative = ((self.valid - probs) * targets).sum(1)
        return (true_positive + smooth) / (
            true_positive + alpha * false_positive + beta * false_negative + smooth
        )

    def bce(
        self,
        *,
        pos_weight: Tensor | None = None,
        focal_gamma: float = 0.0,
        focal_alpha: float | None = None,
        normalize: bool = False,
    ) -> Tensor:
        """Пиксельная BCE, усреднённая по валидным пикселям каждого кадра.

        `normalize=True` делит на сумму весов, а не на число пикселей: перевес
        отдельных пикселей тогда меняет только их долю в сумме, а не масштаб
        всего слагаемого — иначе focal и pos_weight пришлось бы вручную
        подкручивать весом в конфиге.
        """
        loss = F.binary_cross_entropy_with_logits(self.logits, self.targets, reduction="none")
        weight = self.valid
        if pos_weight is not None:
            weight = weight * (1.0 + (pos_weight.unsqueeze(1) - 1.0) * self.targets)
        if focal_gamma > 0.0:
            confidence = self.probs * self.targets + (1.0 - self.probs) * (1.0 - self.targets)
            weight = weight * (1.0 - confidence).pow(focal_gamma)
        if focal_alpha is not None:
            weight = weight * (focal_alpha * self.targets + (1.0 - focal_alpha) * (1.0 - self.targets))
        denominator = weight.sum(1) if normalize else self.valid.sum(1)
        return (loss * weight).sum(1) / denominator.clamp_min(EPS)

    def balancing_pos_weight(self, max_weight: float) -> Tensor:
        """Вес позитивных пикселей, уравнивающий их вклад с фоном, но не выше `max_weight`.

        Правка на 0.5% кадра даёт 0.5% слагаемых BCE и тонет в фоне; здесь её
        доля восстанавливается с ограничением сверху, чтобы совсем маленькая
        маска не взорвала градиент.
        """
        area = self.target_area
        return ((1.0 - area) / area.clamp_min(EPS)).clamp(1.0, max_weight)

    def mean_over(self, values: Tensor, keep: Tensor) -> Tensor:
        """Среднее `values` по кадрам с `keep=1`; для пустого подмножества — ноль."""
        return (values * keep).sum() / keep.sum().clamp_min(1.0)


__all__ = ["EPS", "MaskBatch"]
