"""Compact RGB/JPEG fusion residuals, evaluated on the aligned encoder grid."""

import torch
from torch import nn


class LocalFusionResidual(nn.Module):
    """Local RGB/JPEG residual on the aligned encoder grid."""

    def __init__(self, encoder_channels, aux_channels, width=32):
        super().__init__()
        self.rgb = nn.Conv2d(encoder_channels, width, 1)
        self.jpeg = nn.Conv2d(aux_channels, width, 1)
        self.context = nn.Sequential(
            nn.Conv2d(2 * width, 2 * width, 3, padding=1, groups=2 * width),
            nn.GELU())
        self.output = nn.Conv2d(2 * width, encoder_channels, 1)

    def forward(self, rgb, jpeg):
        context = self.context(torch.cat((self.rgb(rgb), self.jpeg(jpeg)), dim=1))
        residual = self.output(context)
        return residual
