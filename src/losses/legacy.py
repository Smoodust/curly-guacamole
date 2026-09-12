from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


def soft_dice_loss(logits, targets, smooth=1.0, valid_mask=None):
    probs = torch.sigmoid(logits.float()).flatten(1)
    targets = targets.float().flatten(1)
    inter = probs * targets
    if valid_mask is not None:
        valid = valid_mask.float().flatten(1)
        inter = inter * valid
        probs = probs * valid
        targets = targets * valid
    inter = inter.sum(1)
    return (1.0 - (2.0 * inter + smooth) / (probs.sum(1) + targets.sum(1) + smooth)).mean()


def bce_loss(logits, targets, valid_mask=None):
    if valid_mask is not None:
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weighted = (loss * valid_mask).flatten(1).sum(1)
        return (weighted / valid_mask.flatten(1).sum(1).clamp_min(1)).mean()
    return F.binary_cross_entropy_with_logits(logits, targets)

@dataclass(frozen=True)
class LossResult:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    diagnostics: dict[str, tuple[torch.Tensor, torch.Tensor]]


class BinaryFocalLoss(torch.nn.Module):
    """Stable binary focal loss without alpha class balancing."""

    def __init__(self, gamma=1.0):
        super().__init__()
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError('focal gamma must be finite and non-negative')
        self.gamma = gamma

    def forward(self, logits, target, valid_mask=None):
        ce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction='none')
        # Avoid the infinite derivative of x**gamma at x=0 for 0<gamma<1.
        modulation = (-torch.expm1(-ce)).clamp_min(torch.finfo(ce.dtype).tiny)
        loss = modulation.pow(self.gamma) * ce
        if valid_mask is None:
            return loss.mean()
        return ((loss * valid_mask).flatten(1).sum(1) /
                valid_mask.flatten(1).sum(1).clamp_min(1)).mean()


class BoundaryLoss(torch.nn.Module):
    """Mean foreground probability times normalized signed GT distance.

    Maps are prepared in loader workers. Empty/full valid targets have zero
    maps and are excluded from the conditional mean. The loss can be negative.
    """

    def forward(self, logits, distance, valid=None):
        if logits.shape != distance.shape:
            raise ValueError('Boundary distance must match segmentation logits')
        weighted = logits.float().sigmoid() * distance.float()
        if valid is not None:
            distance = distance * valid
            weighted = weighted * valid
            per_image = weighted.flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1)
        else:
            per_image = weighted.flatten(1).mean(1)
        selected = distance.flatten(1).ne(0).any(1)
        return (per_image * selected).sum() / selected.sum().clamp_min(1)


class SegmentationLoss(torch.nn.Module):
    """Configurable pixel, area-weighted Dice, boundary and classification loss.

    Components are weighted contributions whose sum equals total. Diagnostics
    contain unweighted main-head Dice sums/counts, separated by target group.
    """

    def __init__(self, *, dice_scope="all", dice_weight=1.0, aux_weight=0.0, dct_aux_weight=0.0,
                 pixel_loss='bce', focal_gamma=1.0, dice_area_reference=0.0,
                 dice_area_max_weight=3.0, boundary_weight=0.0, aux_loss_weight=None):
        super().__init__()
        if dice_scope not in {"all", "positive"}:
            raise ValueError("loss.dice_scope must be 'all' or 'positive'")
        for name, value in (("dice_weight", dice_weight), ("aux_weight", aux_weight),
                            ("dct_aux_weight", dct_aux_weight)):
            if not 0 <= value < float("inf"):
                raise ValueError(f"loss.{name} must be finite and non-negative")
        self.dice_scope = dice_scope
        self.dice_weight = dice_weight
        from src.config import LossConfig
        LossConfig(dice_scope=dice_scope, dice_weight=dice_weight, pixel_loss=pixel_loss,
                   focal_gamma=focal_gamma, dice_area_reference=dice_area_reference,
                   dice_area_max_weight=dice_area_max_weight, boundary_weight=boundary_weight,
                   aux_loss_weight=aux_loss_weight)
        self.aux_weight = aux_weight if aux_loss_weight is None else aux_loss_weight
        self.dct_aux_weight = dct_aux_weight
        self.pixel_loss = pixel_loss
        self.focal = BinaryFocalLoss(focal_gamma)
        self.dice_area_reference = dice_area_reference
        self.dice_area_max_weight = dice_area_max_weight
        self.boundary_weight = boundary_weight
        self.boundary = BoundaryLoss()

    def _dice(self, logits, target, valid):
        # Accumulate in float32, including under mixed precision training.
        probs = logits.float().sigmoid().flatten(1)
        target = target.float().flatten(1)
        intersection = probs * target
        if valid is not None:
            valid = valid.float().flatten(1)
            intersection = intersection * valid
            probs = probs * valid
            target = target * valid
        positive = target.sum(1) > 0
        losses = 1 - (2 * intersection.sum(1) + 1) / (probs.sum(1) + target.sum(1) + 1)
        selected = positive if self.dice_scope == "positive" else torch.ones_like(positive)
        # Zero with a gradient path when the batch has no positive targets.
        weights = selected.float()
        if self.dice_area_reference > 0:
            pixels = target.shape[1] if valid is None else valid.flatten(1).sum(1).clamp_min(1)
            area = target.sum(1) / pixels
            size_weight = (self.dice_area_reference / area.clamp_min(1e-8)).sqrt()
            size_weight = size_weight.clamp(1, self.dice_area_max_weight)
            weights = weights * torch.where(positive, size_weight, 1.)
        dice = (losses * weights).sum() / weights.sum().clamp_min(1)
        return dice, losses, positive

    def forward(self, out, batch):
        target = batch["mask"].float()
        valid = batch.get("valid_mask")
        dice, per_image, positive = self._dice(out["logits"], target, valid)
        labels = batch.get("label")
        if labels is None:
            labels = positive.float().reshape_as(out["cls_logits"])
        components = {
            self.pixel_loss: (self.focal if self.pixel_loss == 'focal' else bce_loss)(
                out["logits"].float(), target, valid_mask=valid),
            "dice": self.dice_weight * dice,
            "cls": .3 * bce_loss(out["cls_logits"].float(), labels.float()),
        }
        if self.boundary_weight > 0:
            if 'boundary_distance' not in batch:
                raise ValueError('Boundary loss requires boundary_distance from the training dataset')
            components['boundary'] = self.boundary_weight * self.boundary(
                out['logits'], batch['boundary_distance'], valid)
        if self.aux_weight > 0 and "aux_logits" in out:
            logits = out["aux_logits"].float()
            components["aux_bce"] = self.aux_weight * bce_loss(logits, target, valid_mask=valid)
            components["aux_dice"] = self.aux_weight * self.dice_weight * self._dice(logits, target, valid)[0]
        available = out.get('dct_aux_available')
        if self.dct_aux_weight > 0 and (available is None or available.any()):
            logits = out["dct_aux_logits"].float()
            if available is not None:
                logits = logits[available]
                target = target[available]
                valid = valid[available] if valid is not None else None
            size = logits.shape[-2:]
            aux_valid = None
            if valid is not None:
                aux_valid = F.interpolate(valid.float(), size=size, mode="area")
                target = F.interpolate(target * valid, size=size, mode="area") / aux_valid.clamp_min(1e-6)
            else:
                target = F.interpolate(target, size=size, mode="area")
            components["dct_aux_bce"] = self.dct_aux_weight * bce_loss(logits, target, valid_mask=aux_valid)
            components["dct_aux_dice"] = self.dct_aux_weight * self.dice_weight * self._dice(logits, target, aux_valid)[0]
        diagnostics = {
            "dice_pos": ((per_image * positive).sum(), positive.sum()),
            "dice_neg": ((per_image * ~positive).sum(), (~positive).sum()),
        }
        return LossResult(sum(components.values()), components, diagnostics)


class LossMeter:
    """Epoch means; conditional Dice diagnostics use actual group counts.

    The objective and its contributions are averaged by batch image count,
    matching train/loss. Missing diagnostic groups are omitted, not logged as 0.
    Detached sums remain on device until compute(), avoiding per-term GPU sync.
    """

    def __init__(self):
        self.sums = {}
        self.counts = {}

    def update(self, result: LossResult, batch_size: int):
        for key, value in {"total": result.total, **result.components}.items():
            self._add(key, value.detach() * batch_size, batch_size)
        for key, (value, count) in result.diagnostics.items():
            self._add(key, value.detach(), count.detach())

    def _add(self, key, value, count):
        self.sums[key] = self.sums.get(key, 0) + value
        self.counts[key] = self.counts.get(key, 0) + count

    def compute(self):
        return {key: float(value / self.counts[key]) for key, value in self.sums.items()
                if float(self.counts[key]) > 0}


def compute_loss(out, batch, aux_weight: float = 0.0, dct_aux_weight: float = 0.0):
    """Legacy scalar API: BCE + all-image Dice + classifier and auxiliary losses."""
    return SegmentationLoss(aux_weight=aux_weight, dct_aux_weight=dct_aux_weight)(out, batch).total
