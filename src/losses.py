import math
from dataclasses import dataclass

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


class SegmentationLoss(torch.nn.Module):
    """BCE + all-image Dice, classifier BCE and decoder auxiliary supervision."""

    def __init__(self, *, dice_weight=1.0, aux_weight=.4):
        super().__init__()
        for value in (dice_weight, aux_weight):
            if not math.isfinite(value) or value < 0:
                raise ValueError('Loss weights must be finite and nonnegative')
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight

    @staticmethod
    def _dice(logits, target):
        probs = logits.float().sigmoid().flatten(1)
        target = target.float().flatten(1)
        positive = target.sum(1) > 0
        losses = 1 - (2 * (probs * target).sum(1) + 1) / (probs.sum(1) + target.sum(1) + 1)
        return losses.mean(), losses, positive

    def forward(self, out, batch):
        target = batch['mask'].float()
        dice, per_image, positive = self._dice(out['logits'], target)
        labels = batch.get('label')
        if labels is None:
            labels = positive.float().reshape_as(out['cls_logits'])
        components = {
            'bce': bce_loss(out['logits'].float(), target),
            'dice': self.dice_weight * dice,
            'cls': .3 * bce_loss(out['cls_logits'].float(), labels.float()),
        }
        if self.aux_weight > 0 and 'aux_logits' in out:
            logits = out['aux_logits'].float()
            components['aux_bce'] = self.aux_weight * bce_loss(logits, target)
            components['aux_dice'] = self.aux_weight * self.dice_weight * self._dice(logits, target)[0]
        diagnostics = {'dice_pos': ((per_image * positive).sum(), positive.sum()),
                       'dice_neg': ((per_image * ~positive).sum(), (~positive).sum())}
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
        result = {key: float(value / self.counts[key]) for key, value in self.sums.items()
                  if float(self.counts[key]) > 0}
        return result


def compute_loss(out, batch, aux_weight: float = 0.0):
    """Scalar convenience API for the baseline objective."""
    return SegmentationLoss(aux_weight=aux_weight)(out, batch).total
