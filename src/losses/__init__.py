"""BCE+Dice/dice_scope legacy target, plus a timm-style registry for new objectives."""

from src.losses.aic import AicHarmonicLoss, AicLoss, AicSurrogateLoss
from src.losses.base import LossArm
from src.losses.functional import harmonic_aic, soft_false_positive
from src.losses.legacy import LossMeter, LossResult, SegmentationLoss, bce_loss, compute_loss, soft_dice_loss
from src.losses.mask_batch import MaskBatch
from src.losses.pixel import BalancedBceDiceLoss, FocalDiceLoss, PixelRegionLoss
from src.losses.registry import build_loss, create_loss, is_loss, list_losses, register_loss
from src.losses.tversky import FocalTverskyLoss

__all__ = [
    "AicHarmonicLoss",
    "AicLoss",
    "AicSurrogateLoss",
    "BalancedBceDiceLoss",
    "FocalDiceLoss",
    "FocalTverskyLoss",
    "LossArm",
    "LossMeter",
    "LossResult",
    "MaskBatch",
    "PixelRegionLoss",
    "SegmentationLoss",
    "bce_loss",
    "build_loss",
    "compute_loss",
    "create_loss",
    "harmonic_aic",
    "is_loss",
    "list_losses",
    "register_loss",
    "soft_dice_loss",
    "soft_false_positive",
]
