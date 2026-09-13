"""Training-only adaptation of NC-Net equation (4) to forensic features."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class ForensicProjectionHead(nn.Sequential):
    """Pointwise projection without batch statistics or shared RNG side effects."""

    def __init__(self, channels, dim):
        # Construction must not shift the initialization of subsequent shared layers.
        with torch.random.fork_rng(devices=[]):
            super().__init__(nn.Conv2d(channels, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1))


@dataclass(frozen=True)
class ContrastiveResult:
    loss: torch.Tensor
    diagnostics: dict[str, tuple[torch.Tensor, torch.Tensor]]


class ForensicIntraContrastiveLoss(nn.Module):
    """Equal-image mean over available images with >=2 positives and >=1 negative.

    Sampling uses the global torch RNG, saved by the training checkpoint. Only
    sampled vectors enter similarity matrices; mixed and padded cells are ignored.
    """

    def __init__(self, temperature=.1, max_samples=256, positive_fraction=.9):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('forensic_intra_temperature must be finite and positive')
        if type(max_samples) is not int or max_samples < 2:
            raise ValueError('forensic_intra_max_samples must be an integer >= 2')
        if not math.isfinite(positive_fraction) or not 0 < positive_fraction <= 1:
            raise ValueError('forensic_intra_positive_fraction must be in (0, 1]')
        self.temperature = temperature
        self.max_samples = max_samples
        self.positive_fraction = positive_fraction

    @torch.no_grad()
    def mask_cells(self, mask, valid_mask, size):
        valid = torch.ones_like(mask) if valid_mask is None else valid_mask
        valid_fraction = F.interpolate(valid.float(), size=size, mode='area')
        foreground = F.interpolate(mask.float() * valid.float(), size=size, mode='area')
        foreground = foreground / valid_fraction.clamp_min(1e-6)
        fully_valid = valid_fraction >= 1 - 1e-6
        return (fully_valid & (foreground >= self.positive_fraction),
                fully_valid & (foreground <= 1e-6))

    def sample(self, indices):
        if indices.numel() <= self.max_samples:
            return indices
        return indices[torch.randperm(indices.numel(), device=indices.device)[:self.max_samples]]

    def _selected(self, pos, neg):
        # .float() alone is insufficient inside an outer autocast context.
        with torch.autocast(device_type=pos.device.type, enabled=False):
            pos, neg = pos.float(), neg.float()
            pp = pos @ pos.T
            positive_logit = (pp.sum(1) - pp.diagonal()) / (pos.shape[0] - 1)
            negative_logits = pos @ neg.T
            logits = torch.cat([positive_logit[:, None], negative_logits], dim=1)
            labels = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
            loss = F.cross_entropy(logits / self.temperature, labels)
            return loss, positive_logit.mean().detach(), negative_logits.mean().detach()

    def selected_loss(self, pos, neg):
        """Equation (4) for already normalized [N,C], [M,C] vectors."""
        if pos.ndim != 2 or neg.ndim != 2 or pos.shape[1] != neg.shape[1]:
            raise ValueError('selected vectors must have shapes [N,C] and [M,C]')
        if len(pos) < 2 or len(neg) < 1:
            raise ValueError('selected loss requires >=2 positive and >=1 negative vectors')
        return self._selected(pos, neg)[0]

    def forward(self, embeddings, mask, available, valid_mask=None):
        if embeddings.ndim != 4 or mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError('forensic embeddings and mask must be [B,C,H,W] and [B,1,H,W]')
        if mask.shape[0] != embeddings.shape[0]:
            raise ValueError('forensic embeddings and mask batch sizes must match')
        if available.dtype != torch.bool or available.shape != (embeddings.shape[0],):
            raise ValueError('forensic_available must be BoolTensor[B]')
        if valid_mask is not None and valid_mask.shape != mask.shape:
            raise ValueError('valid_mask must match mask shape')
        if any(t.device != embeddings.device for t in (mask, available)):
            raise ValueError('forensic inputs must be on the same device')
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            positive, negative = self.mask_cells(mask, valid_mask, embeddings.shape[-2:])
            positive, negative = positive.flatten(1), negative.flatten(1)
            eligible = available & (positive.sum(1) >= 2) & (negative.sum(1) >= 1)
            vectors = embeddings.float().flatten(2).transpose(1, 2)
            # Empty sum is finite and differentiable without reading unused values.
            loss_sum = embeddings.float().reshape(-1)[:0].sum()
            pos_count = embeddings.new_zeros((), dtype=torch.float32)
            neg_count = pos_count.clone()
            pos_similarity, neg_similarity = pos_count.clone(), pos_count.clone()
            for index in eligible.nonzero(as_tuple=True)[0].tolist():
                pos_indices = self.sample(positive[index].nonzero(as_tuple=True)[0])
                neg_indices = self.sample(negative[index].nonzero(as_tuple=True)[0])
                pos = F.normalize(vectors[index, pos_indices], dim=1, eps=1e-6)
                neg = F.normalize(vectors[index, neg_indices], dim=1, eps=1e-6)
                loss, s_pos, s_neg = self._selected(pos, neg)
                loss_sum = loss_sum + loss
                pos_count += len(pos_indices)
                neg_count += len(neg_indices)
                pos_similarity += s_pos
                neg_similarity += s_neg
            count = eligible.sum()
            total = count.new_tensor(len(embeddings))
            diagnostics = {
                'forensic_intra_raw': (loss_sum.detach(), count),
                'forensic_intra_coverage': (count, total),
                'forensic_intra_positive_samples': (pos_count, count),
                'forensic_intra_negative_samples': (neg_count, count),
                'forensic_intra_s_pos': (pos_similarity, count),
                'forensic_intra_s_neg': (neg_similarity, count),
            }
            return ContrastiveResult(loss_sum / count.clamp_min(1), diagnostics)
