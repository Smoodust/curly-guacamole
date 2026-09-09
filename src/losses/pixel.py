from __future__ import annotations

from torch import Tensor

from src.losses.base import LossArm
from src.losses.mask_batch import MaskBatch
from src.losses.registry import register_loss


class PixelRegionLoss(LossArm):
    """Общий каркас «пиксельное слагаемое + Dice»: различается только BCE."""

    def __init__(self, *, cls_weight: float = 0.3, aux_weight: float = 0.0, dct_aux_weight: float = 0.0,
                 bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1.0) -> None:
        super().__init__(cls_weight=cls_weight, aux_weight=aux_weight, dct_aux_weight=dct_aux_weight)
        if min(bce_weight, dice_weight) < 0.0:
            raise ValueError("bce_weight and dice_weight must be non-negative")
        if smooth <= 0.0:
            raise ValueError("smooth must be positive")
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

    def mask_term(self, logits: Tensor, targets: Tensor, valid_mask: Tensor | None) -> dict[str, Tensor]:
        batch = MaskBatch(logits, targets, valid_mask)
        dice = (1.0 - batch.dice(self.smooth)).mean()
        return {"bce": self.bce_weight * self.pixel_term(batch), "dice": self.dice_weight * dice}

    def pixel_term(self, batch: MaskBatch) -> Tensor:
        return batch.bce().mean()


@register_loss("balanced_bce_dice")
class BalancedBceDiceLoss(PixelRegionLoss):
    """BCE с поднятым весом позитивных пикселей внутри каждого кадра.

    Маска на 0.5% кадра даёт 0.5% слагаемых BCE, и градиент по ней теряется на
    фоне. Вес выравнивает вклад правки и фона в пределах кадра, ограничение
    сверху не даёт совсем крошечной маске захватить весь градиент.
    """

    def __init__(self, *, cls_weight: float = 0.3, aux_weight: float = 0.0, dct_aux_weight: float = 0.0,
                 bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1.0,
                 max_pos_weight: float = 20.0) -> None:
        super().__init__(cls_weight=cls_weight, aux_weight=aux_weight, dct_aux_weight=dct_aux_weight,
                         bce_weight=bce_weight, dice_weight=dice_weight, smooth=smooth)
        if max_pos_weight < 1.0:
            raise ValueError("max_pos_weight must be at least 1")
        self.max_pos_weight = float(max_pos_weight)

    def pixel_term(self, batch: MaskBatch) -> Tensor:
        weight = batch.balancing_pos_weight(self.max_pos_weight)
        return batch.bce(pos_weight=weight, normalize=True).mean()


@register_loss("focal_dice")
class FocalDiceLoss(PixelRegionLoss):
    """Focal-версия пиксельного слагаемого: вес уходит с уверенно угаданного фона.

    Нормировка на сумму весов сохраняет масштаб слагаемого, поэтому эксперимент
    проверяет именно перераспределение градиента, а не заодно и уменьшение
    вклада BCE относительно Dice.
    """

    def __init__(self, *, cls_weight: float = 0.3, aux_weight: float = 0.0, dct_aux_weight: float = 0.0,
                 bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1.0,
                 focal_gamma: float = 2.0, focal_alpha: float | None = None) -> None:
        super().__init__(cls_weight=cls_weight, aux_weight=aux_weight, dct_aux_weight=dct_aux_weight,
                         bce_weight=bce_weight, dice_weight=dice_weight, smooth=smooth)
        if focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative")
        if focal_alpha is not None and not 0.0 < focal_alpha < 1.0:
            raise ValueError("focal_alpha must be in (0, 1)")
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = None if focal_alpha is None else float(focal_alpha)

    def pixel_term(self, batch: MaskBatch) -> Tensor:
        return batch.bce(focal_gamma=self.focal_gamma, focal_alpha=self.focal_alpha,
                         normalize=True).mean()


__all__ = ["BalancedBceDiceLoss", "FocalDiceLoss", "PixelRegionLoss"]
