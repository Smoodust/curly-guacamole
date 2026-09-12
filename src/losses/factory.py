from __future__ import annotations

from src.losses.aic import AicHarmonicLoss, AicSurrogateLoss
from src.losses.legacy import SegmentationLoss

# Which LossConfig fields each objective actually consumes. Anything outside the
# selected set is rejected by LossConfig rather than silently ignored: a boundary
# weight left on an AIC arm would otherwise cost a full run before anyone noticed.
OBJECTIVE_FIELDS = {
    "bce_dice": frozenset({
        "dice_scope", "dice_weight", "pixel_loss", "focal_gamma",
        "dice_area_reference", "dice_area_max_weight", "boundary_weight",
    }),
    "aic_surrogate": frozenset({
        "dice_weight", "bce_weight", "fpr_weight",
        "area_threshold", "area_mode", "area_softness",
    }),
    "aic_harmonic": frozenset({
        "bce_weight", "aic_weight",
        "area_threshold", "area_mode", "area_softness",
    }),
}

# aux_loss_weight overrides deep supervision for every objective.
SHARED_FIELDS = frozenset({"aux_loss_weight"})

_CLASSES = {
    "bce_dice": SegmentationLoss,
    "aic_surrogate": AicSurrogateLoss,
    "aic_harmonic": AicHarmonicLoss,
}

OBJECTIVES = tuple(sorted(_CLASSES))


def build_criterion(config, *, aux_weight: float = 0.0, dct_aux_weight: float = 0.0):
    """Build the configured objective, passing only the fields it consumes.

    Validation builds the same objective without deep supervision, matching the
    reported train and validation loss of a bce_dice run.
    """
    loss = config.loss
    objective = loss.objective
    if objective not in _CLASSES:
        raise ValueError(f"unknown loss.objective {objective!r}; use one of {', '.join(OBJECTIVES)}")
    settings = loss.to_dict()
    kwargs = {
        name: value for name, value in settings.items()
        if name in OBJECTIVE_FIELDS[objective] or name in SHARED_FIELDS
    }
    return _CLASSES[objective](aux_weight=aux_weight, dct_aux_weight=dct_aux_weight, **kwargs)


__all__ = ["OBJECTIVE_FIELDS", "OBJECTIVES", "SHARED_FIELDS", "build_criterion"]
