from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn.functional as F
from torch import Tensor

from src.losses.legacy import LossResult, bce_loss
from src.losses.mask_batch import MaskBatch


class LossArm(ABC):
    """Objective contract: own mask term, everything else shared with SegmentationLoss.

    The gate head, deep supervision and the reported LossResult match
    SegmentationLoss exactly, so LossMeter and epoch logging stay objective
    agnostic. Diagnostics are always the plain main-head Dice loss, never the
    arm's own mask term: otherwise dice_pos/dice_neg could not be compared
    against a bce_dice run. The aux head learns the same term as the main head,
    or an experiment would test two things at once.
    """

    def __init__(self, *, cls_weight=0.3, aux_weight=0.0, dct_aux_weight=0.0,
                 aux_loss_weight=None):
        for name, value in (("cls_weight", cls_weight), ("aux_weight", aux_weight),
                            ("dct_aux_weight", dct_aux_weight)):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if aux_loss_weight is not None and aux_loss_weight < 0.0:
            raise ValueError("aux_loss_weight must be non-negative")
        self.cls_weight = float(cls_weight)
        # Keeps aux modules loadable under strict checkpoint loading while the
        # auxiliary term itself can be switched off, as SegmentationLoss does.
        self.aux_weight = float(aux_weight if aux_loss_weight is None else aux_loss_weight)
        self.dct_aux_weight = float(dct_aux_weight)

    def __call__(self, out: dict, batch: dict) -> LossResult:
        target = batch["mask"].float()
        valid = batch.get("valid_mask")
        components = dict(self.mask_term(out["logits"].float(), target, valid))

        main = MaskBatch(out["logits"], target, valid)
        positive = main.is_positive
        labels = batch.get("label")
        if labels is None:
            labels = positive.reshape_as(out["cls_logits"])
        if self.cls_weight > 0.0:
            components["cls"] = self.cls_weight * bce_loss(out["cls_logits"].float(), labels.float())

        if self.aux_weight > 0.0 and "aux_logits" in out:
            for key, value in self.mask_term(out["aux_logits"].float(), target, valid).items():
                components[f"aux_{key}"] = self.aux_weight * value

        available = out.get("dct_aux_available")
        if self.dct_aux_weight > 0.0 and "dct_aux_logits" in out and (
                available is None or available.any()):
            logits = out["dct_aux_logits"].float()
            dct_target, dct_valid = target, valid
            if available is not None:
                logits = logits[available]
                dct_target = dct_target[available]
                dct_valid = dct_valid[available] if dct_valid is not None else None
            size = logits.shape[-2:]
            if dct_valid is not None:
                aux_valid = F.interpolate(dct_valid.float(), size=size, mode="area")
                dct_target = F.interpolate(dct_target * dct_valid, size=size, mode="area") / aux_valid.clamp_min(1e-6)
            else:
                aux_valid = None
                dct_target = F.interpolate(dct_target, size=size, mode="area")
            for key, value in self.mask_term(logits, dct_target, aux_valid).items():
                components[f"dct_aux_{key}"] = self.dct_aux_weight * value

        # Dice loss per frame, matching the SegmentationLoss diagnostic.
        per_image = 1.0 - main.dice()
        diagnostics = {
            "dice_pos": ((per_image * positive).sum(), positive.sum()),
            "dice_neg": ((per_image * (1.0 - positive)).sum(), (1.0 - positive).sum()),
        }
        return LossResult(sum(components.values()), components, diagnostics)

    @abstractmethod
    def mask_term(self, logits: Tensor, targets: Tensor, valid_mask: Tensor | None) -> dict[str, Tensor]:
        """Named mask contributions for one head, for example bce and dice."""
        raise NotImplementedError

    def __repr__(self) -> str:
        options = ", ".join(f"{name}={value!r}" for name, value in sorted(vars(self).items()))
        return f"{type(self).__name__}({options})"


__all__ = ["LossArm"]
