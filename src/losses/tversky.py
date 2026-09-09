from __future__ import annotations

from torch import Tensor

from src.losses.base import LossArm
from src.losses.mask_batch import MaskBatch
from src.losses.registry import register_loss


@register_loss("focal_tversky")
class FocalTverskyLoss(LossArm):
    """BCE плюс focal Tversky: Dice с перекосом в сторону пропусков.

    Dice штрафует ложное срабатывание и пропуск одинаково, а на маленькой маске
    осторожная модель почти ничего не теряет, просто её не найдя. `beta > alpha`
    делает пропуск дороже, показатель `gamma < 1` дополнительно поднимает
    градиент на кадрах, где Tversky уже низкий.
    """

    def __init__(self, *, cls_weight: float = 0.3, aux_weight: float = 0.0, dct_aux_weight: float = 0.0,
                 bce_weight: float = 1.0, tversky_weight: float = 1.0,
                 alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75,
                 smooth: float = 1.0) -> None:
        super().__init__(cls_weight=cls_weight, aux_weight=aux_weight, dct_aux_weight=dct_aux_weight)
        if min(bce_weight, tversky_weight, alpha, beta) < 0.0:
            raise ValueError("weights, alpha and beta must be non-negative")
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if smooth <= 0.0:
            raise ValueError("smooth must be positive")
        self.bce_weight = float(bce_weight)
        self.tversky_weight = float(tversky_weight)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    def mask_term(self, logits: Tensor, targets: Tensor, valid_mask: Tensor | None) -> dict[str, Tensor]:
        batch = MaskBatch(logits, targets, valid_mask)
        index = batch.tversky(self.alpha, self.beta, self.smooth)
        region = (1.0 - index).clamp_min(0.0).pow(self.gamma).mean()
        return {"bce": self.bce_weight * batch.bce().mean(), "tversky": self.tversky_weight * region}


__all__ = ["FocalTverskyLoss"]
