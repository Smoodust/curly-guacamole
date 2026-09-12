"""BCE+Dice/boundary objective, plus the AIC-shaped arms selected by loss.objective."""

from src.losses.aic import AicHarmonicLoss, AicLoss, AicSurrogateLoss
from src.losses.base import LossArm
from src.losses.factory import OBJECTIVE_FIELDS, OBJECTIVES, SHARED_FIELDS, build_criterion
from src.losses.functional import harmonic_aic, soft_false_positive
from src.losses.legacy import (
    BinaryFocalLoss,
    BoundaryLoss,
    LossMeter,
    LossResult,
    SegmentationLoss,
    bce_loss,
    compute_loss,
    soft_dice_loss,
)
from src.losses.mask_batch import MaskBatch

__all__ = [
    "AicHarmonicLoss",
    "AicLoss",
    "AicSurrogateLoss",
    "BinaryFocalLoss",
    "BoundaryLoss",
    "LossArm",
    "LossMeter",
    "LossResult",
    "MaskBatch",
    "OBJECTIVES",
    "OBJECTIVE_FIELDS",
    "SHARED_FIELDS",
    "SegmentationLoss",
    "bce_loss",
    "build_criterion",
    "compute_loss",
    "harmonic_aic",
    "soft_dice_loss",
    "soft_false_positive",
]
