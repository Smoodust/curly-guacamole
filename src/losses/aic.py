from __future__ import annotations

from torch import Tensor

from src.losses.base import LossArm
from src.losses.functional import harmonic_aic, soft_false_positive
from src.losses.mask_batch import MaskBatch
from src.losses.registry import register_loss


class AicLoss(LossArm):
    """Общая часть целей, скроенных по метрике: разделение кадров и мягкий FPR."""

    def __init__(self, *, cls_weight: float = 0.3, aux_weight: float = 0.0, dct_aux_weight: float = 0.0,
                 bce_weight: float = 1.0, smooth: float = 1.0,
                 area_threshold: float = 0.01, area_mode: str = "sigmoid",
                 area_softness: float = 0.5) -> None:
        super().__init__(cls_weight=cls_weight, aux_weight=aux_weight, dct_aux_weight=dct_aux_weight)
        if bce_weight < 0.0:
            raise ValueError("bce_weight must be non-negative")
        if smooth <= 0.0:
            raise ValueError("smooth must be positive")
        if not 0.0 < area_threshold < 1.0:
            raise ValueError("area_threshold must be in (0, 1)")
        self.bce_weight = float(bce_weight)
        self.smooth = float(smooth)
        self.area_threshold = float(area_threshold)
        self.area_mode = str(area_mode)
        self.area_softness = float(area_softness)

    def false_positive(self, batch: MaskBatch) -> Tensor:
        return soft_false_positive(
            batch.predicted_area,
            threshold=self.area_threshold,
            mode=self.area_mode,
            softness=self.area_softness,
        )


@register_loss("aic_surrogate")
class AicSurrogateLoss(AicLoss):
    """Dice считается только по позитивам, негативы штрафуются за площадь.

    Отличие от базовой цели ровно одно. Dice на негативном кадре равен
    `1 - smooth / (площадь + smooth)`, то есть уже за крошечное пятно берёт
    почти максимальный штраф, тогда как метрика не замечает ничего вплоть до
    1% кадра. Здесь это слагаемое заменено на мягкий FPR с тем же порогом, а
    пиксельная BCE остаётся на всех кадрах: она и удерживает негативы в начале
    обучения, пока площадь ещё далеко от порога.
    """

    def __init__(self, *, cls_weight: float = 0.3, aux_weight: float = 0.0, dct_aux_weight: float = 0.0,
                 bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1.0,
                 fpr_weight: float = 1.0, area_threshold: float = 0.01,
                 area_mode: str = "sigmoid", area_softness: float = 0.5) -> None:
        super().__init__(cls_weight=cls_weight, aux_weight=aux_weight, dct_aux_weight=dct_aux_weight,
                         bce_weight=bce_weight, smooth=smooth, area_threshold=area_threshold,
                         area_mode=area_mode, area_softness=area_softness)
        if min(dice_weight, fpr_weight) < 0.0:
            raise ValueError("dice_weight and fpr_weight must be non-negative")
        self.dice_weight = float(dice_weight)
        self.fpr_weight = float(fpr_weight)

    def mask_term(self, logits: Tensor, targets: Tensor, valid_mask: Tensor | None) -> dict[str, Tensor]:
        batch = MaskBatch(logits, targets, valid_mask)
        positive = batch.is_positive

        dice = batch.mean_over(1.0 - batch.dice(self.smooth), positive)
        false_positive = batch.mean_over(self.false_positive(batch), 1.0 - positive)

        return {
            "bce": self.bce_weight * batch.bce().mean(),
            "dice": self.dice_weight * dice,
            "fpr": self.fpr_weight * false_positive,
        }


@register_loss("aic_harmonic")
class AicHarmonicLoss(AicLoss):
    """Единица минус AIC, посчитанный по батчу из мягких Dice и FPR.

    Гармоническое среднее само распределяет градиент: пока хуже Dice, тянет
    Dice, как только начинает расти FPR — переключается на него. Пиксельная BCE
    остаётся рядом, потому что в начале обучения площадь предсказания далека от
    порога, и один только мягкий FPR почти не даёт сигнала.
    """

    def __init__(self, *, cls_weight: float = 0.3, aux_weight: float = 0.0, dct_aux_weight: float = 0.0,
                 bce_weight: float = 1.0, aic_weight: float = 1.0, smooth: float = 1.0,
                 area_threshold: float = 0.01, area_mode: str = "rational",
                 area_softness: float = 0.5) -> None:
        super().__init__(cls_weight=cls_weight, aux_weight=aux_weight, dct_aux_weight=dct_aux_weight,
                         bce_weight=bce_weight, smooth=smooth, area_threshold=area_threshold,
                         area_mode=area_mode, area_softness=area_softness)
        if aic_weight < 0.0:
            raise ValueError("aic_weight must be non-negative")
        self.aic_weight = float(aic_weight)

    def mask_term(self, logits: Tensor, targets: Tensor, valid_mask: Tensor | None) -> dict[str, Tensor]:
        batch = MaskBatch(logits, targets, valid_mask)
        positive = batch.is_positive

        dice = batch.mean_over(batch.dice(self.smooth), positive)
        false_positive = batch.mean_over(self.false_positive(batch), 1.0 - positive).clamp(0.0, 1.0)

        # В батче без позитивов Dice равен нулю и обнуляет всю свёртку вместе с
        # её градиентом, хотя штрафовать негативы всё ещё нужно.
        has_positive = (positive.sum() > 0).float()
        region = has_positive * (1.0 - harmonic_aic(dice, false_positive)) + (
            1.0 - has_positive
        ) * false_positive

        return {"bce": self.bce_weight * batch.bce().mean(), "harmonic": self.aic_weight * region}


__all__ = ["AicHarmonicLoss", "AicLoss", "AicSurrogateLoss"]
