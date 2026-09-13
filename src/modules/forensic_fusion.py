from dataclasses import dataclass

import torch
from torch import nn

from src.forensic.dct.constants import CHANNEL_COUNT
from src.modules.forensic_branch import ForensicBranch
from src.modules.forensic_contrastive import ForensicProjectionHead
from src.modules.gated_fuse import GatedFuse
from src.modules.jpeg_branch import JPEGBranch


@dataclass(frozen=True)
class ForensicTrainingResult:
    features: list[torch.Tensor]
    aux_logits: torch.Tensor | None = None
    embeddings: torch.Tensor | None = None
    available: torch.Tensor | None = None


class ForensicFusion(nn.Module):
    """Строит forensic-пирамиду и сливает её с encoder-фичами."""

    FUSION_STRIDES = (8, 16, 32)

    def __init__(self, encoder_strides, encoder_channels, forensic_channels, use_aux=False, forensic_mode='maps', jpeg_variant='baseline', fusion_variant='baseline', forensic_contrastive_dim=0):
        super().__init__()
        if type(forensic_contrastive_dim) is not int or forensic_contrastive_dim < 0:
            raise ValueError('forensic_contrastive_dim must be an integer >= 0')
        self.contrastive_dim = forensic_contrastive_dim

        if len(encoder_strides) != len(encoder_channels):
            raise ValueError("encoder strides and channels must have the same length")
        if len(set(encoder_strides)) != len(encoder_strides):
            raise ValueError("encoder strides must be unique")
        if len(forensic_channels) != len(self.FUSION_STRIDES):
            raise ValueError(
                f"expected {len(self.FUSION_STRIDES)} forensic channel groups, got {len(forensic_channels)}"
            )

        self.fusion_strides = (4, 8, 16, 32) if jpeg_variant == 'subblock4' else self.FUSION_STRIDES
        missing_strides = [
            stride for stride in self.fusion_strides
            if stride not in encoder_strides
        ]
        if missing_strides:
            raise ValueError(f"encoder is missing fusion strides: {missing_strides}")

        self.encoder_strides = encoder_strides
        self.forensic_mode = forensic_mode

        self.branch = JPEGBranch(forensic_channels, variant=jpeg_variant) if forensic_mode == 'jpeg' else ForensicBranch(
            in_ch=CHANNEL_COUNT,
            channels=forensic_channels,
        )

        self.aux_head = nn.Conv2d(forensic_channels[0], 1, 1) if use_aux else None

        self.fusion_blocks = nn.ModuleDict({
            str(stride): GatedFuse(
                encoder_channels[encoder_strides.index(stride)],
                self.branch.channels_by_stride[stride],
                variant=fusion_variant,
            )
            for stride in self.fusion_strides
        })
        if forensic_contrastive_dim:
            self.contrastive_head = ForensicProjectionHead(forensic_channels[0], forensic_contrastive_dim)

    def gate_stats(self) -> dict[str, float]:
        return {
            "max_abs": max(
                float(block.channel_gate.detach().abs().max())
                for block in self.fusion_blocks.values()
            )
        }

    def forward(self, encoder_features, forensic_map, *, return_aux=False, jpeg=None):
        """Preserve the legacy list / (list, auxiliary logits) return contract."""
        result = self._forward(encoder_features, forensic_map, jpeg=jpeg, return_aux=return_aux,
                               contrastive=False)
        return (result.features, result.aux_logits) if return_aux else result.features

    def forward_training(self, encoder_features, forensic_map, *, jpeg=None):
        """Explicit structured training output; projection is never run in eval."""
        return self._forward(encoder_features, forensic_map, jpeg=jpeg, return_aux=True,
                             contrastive=self.training and self.contrastive_dim > 0)

    @staticmethod
    def _scatter(value, indices, batch_size):
        if value is None:
            return None
        return value.new_zeros((batch_size, *value.shape[1:])).index_copy(
            0, torch.as_tensor(indices, device=value.device), value)

    def _forward(self, encoder_features, forensic_map, *, jpeg, return_aux, contrastive):
        embeddings, availability = None, None
        reference = encoder_features[self.encoder_strides.index(8)]
        if contrastive:
            availability = torch.ones(reference.shape[0], dtype=torch.bool, device=reference.device)
        if self.forensic_mode == 'jpeg':
            available = [i for i, sample in enumerate(jpeg) if sample.get('available', True)]
            if not available:
                if contrastive:
                    availability.zero_()
                    # The single-device trainer tolerates unused branch parameters.
                    # Connect a zero to head parameters without running its projection.
                    zero = sum(p.reshape(-1)[:0].sum() for p in self.contrastive_head.parameters())
                    embeddings = reference.new_zeros((len(jpeg), self.contrastive_dim, *reference.shape[-2:])) + zero
                return ForensicTrainingResult(encoder_features, embeddings=embeddings, available=availability)
            if len(available) != len(jpeg):
                # Exclude PNG samples from the branch and fusion BatchNorm too.
                subset = [feature[available] for feature in encoder_features]
                subset_result = self._forward(subset, None, return_aux=return_aux,
                                              jpeg=[jpeg[i] for i in available], contrastive=contrastive)
                result = [feature.clone() for feature in encoder_features]
                for original, updated in zip(result, subset_result.features, strict=True):
                    original[available] = updated
                if contrastive:
                    availability.zero_()
                    availability[available] = True
                return ForensicTrainingResult(
                    result, self._scatter(subset_result.aux_logits, available, len(jpeg)),
                    self._scatter(subset_result.embeddings, available, len(jpeg)), availability)
            sizes = {s: encoder_features[self.encoder_strides.index(s)].shape[-2:] for s in self.fusion_strides}
            forensic_features = self.branch(jpeg, sizes)
        else:
            forensic_features = self.branch(forensic_map)

        if contrastive:
            aligned = GatedFuse._resize_like(forensic_features[8], reference)
            embeddings = self.contrastive_head(aligned)

        for stride in self.fusion_strides:
            index = self.encoder_strides.index(stride)

            encoder_features[index] = self.fusion_blocks[str(stride)](
                encoder_features[index],
                forensic_features[stride],
            )

        aux = self.aux_head(forensic_features[8]) if return_aux and self.aux_head is not None else None
        return ForensicTrainingResult(encoder_features, aux, embeddings, availability)
