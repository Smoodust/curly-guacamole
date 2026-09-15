import torch
import torch.nn.functional as F
from torch import nn

from src.modules.fusion_residuals import LocalFusionResidual


class GatedFuse(nn.Module):
    """Безопасно добавляет auxiliary-фичи к encoder-фичам через обучаемый поканальный gate."""

    def __init__(self, encoder_channels: int, aux_channels: int) -> None:
        super().__init__()
        self.channel_gate = nn.Parameter(torch.zeros(1, encoder_channels, 1, 1))
        self.residual = LocalFusionResidual(encoder_channels, aux_channels)

    @staticmethod
    def _resize_like(x, reference):
        if x.shape[-2:] == reference.shape[-2:]:
            return x

        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, encoder_features, aux_features):
        aux_features = self._resize_like(
            aux_features,
            encoder_features,
        )

        return encoder_features + self.channel_gate * self.residual(encoder_features, aux_features)
