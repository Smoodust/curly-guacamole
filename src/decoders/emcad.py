"""EMCAD architecture adapted to the project's single-mask decoder contract.

Reference: Rahman et al., CVPR 2024, https://arxiv.org/abs/2405.06880.
Implements parallel additive multi-scale convolutions and a shared spatial
attention map predictor. Auxiliary supervision defaults to the final merged
skip before refinement.
The paper's multi-head training scheme is not reproduced.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .layers import make_norm


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(1, channels // 16)
        self.project = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, features):
        average = features.mean(dim=(-2, -1), keepdim=True)
        maximum = features.amax(dim=(-2, -1), keepdim=True)
        return features * torch.sigmoid(self.project(average) + self.project(maximum))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, features):
        summary = torch.cat((features.mean(1, keepdim=True),
                             features.amax(1, keepdim=True)), dim=1)
        return features * torch.sigmoid(self.project(summary))


class MultiScaleConvolution(nn.Module):
    """Pointwise expansion, parallel depthwise kernels, shuffle and residual."""

    def __init__(self, channels, kernel_sizes, expansion_factor, norm, activation):
        super().__init__()
        hidden = int(channels * expansion_factor)
        activation_class = nn.ReLU if activation == 'relu' else nn.ReLU6
        self.expand = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            make_norm(norm, hidden), activation_class(inplace=True),
        )
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel, padding=kernel // 2,
                          groups=hidden, bias=False),
                make_norm(norm, hidden), activation_class(inplace=True),
            ) for kernel in kernel_sizes
        ])
        self.shuffle = nn.ChannelShuffle(math.gcd(hidden, channels))
        self.project = nn.Sequential(
            nn.Conv2d(hidden, channels, 1, bias=False), make_norm(norm, channels),
        )

    def forward(self, features):
        expanded = self.expand(features)
        combined = self.scales[0](expanded)
        for branch in self.scales[1:]:
            combined = combined + branch(expanded)
        return features + self.project(self.shuffle(combined))


class EfficientUpsample(nn.Module):
    def __init__(self, input_channels, output_channels, norm):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1,
                      groups=input_channels, bias=False),
            make_norm(norm, input_channels), nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, 1),
        )

    def forward(self, features, size):
        # Explicit size also supports odd and rectangular encoder feature maps.
        return self.refine(F.interpolate(features, size=size, mode='nearest'))


class GroupedAttentionGate(nn.Module):
    def __init__(self, channels, kernel_size, norm):
        super().__init__()
        hidden = channels // 2
        groups = hidden if kernel_size > 1 else 1

        def projection():
            return nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size, padding=kernel_size // 2,
                          groups=groups), make_norm(norm, hidden),
            )

        self.query = projection()
        self.skip = projection()
        self.mask = nn.Sequential(
            nn.ReLU(inplace=True), nn.Conv2d(hidden, 1, 1),
            make_norm(norm, 1), nn.Sigmoid(),
        )

    def forward(self, features, skip):
        return skip * self.mask(self.query(features) + self.skip(skip))


class EMCADDecoder(nn.Module):
    """Four-scale EMCAD with encoder-width stages and output at stride 4."""

    head_kernel_size = 1

    def __init__(self, encoder_channels, encoder_strides, norm='batch', use_aux=False,
                 kernel_sizes=(1, 3, 5), expansion_factor=2, lgag_kernel_size=3,
                 activation='relu'):
        super().__init__()
        if len(encoder_channels) != 4 or tuple(encoder_strides) != (4, 8, 16, 32):
            raise ValueError('EMCAD requires four encoder scales at strides 4, 8, 16, 32')
        if any(c < 2 or c % 2 for c in encoder_channels):
            raise ValueError('EMCAD requires positive even encoder channels')
        if not kernel_sizes or any(not isinstance(k, int) or k < 1 or k % 2 == 0
                                   for k in (*kernel_sizes, lgag_kernel_size)):
            raise ValueError('EMCAD kernel sizes must be positive odd integers')
        if not isinstance(expansion_factor, int) or expansion_factor < 1:
            raise ValueError('expansion_factor must be a positive integer')
        if activation not in {'relu', 'relu6'}:
            raise ValueError('activation must be relu or relu6')
        if norm == 'group' and any((c // 2) % min(32, c // 2) for c in encoder_channels[:-1]):
            raise ValueError('EMCAD group norm requires gate widths divisible by their group count')

        channels = list(reversed(encoder_channels))
        self.channel_attention = nn.ModuleList([ChannelAttention(c) for c in channels])
        self.spatial_attention = SpatialAttention()
        self.refinement = nn.ModuleList([
            MultiScaleConvolution(c, kernel_sizes, expansion_factor, norm, activation)
            for c in channels
        ])
        self.upsample = nn.ModuleList([
            EfficientUpsample(a, b, norm) for a, b in zip(channels, channels[1:], strict=False)
        ])
        self.gates = nn.ModuleList([
            GroupedAttentionGate(c, lgag_kernel_size, norm) for c in channels[1:]
        ])
        self.out_channels = channels[-1]
        self.output_stride = 4
        self.aux_head = nn.Conv2d(channels[3], 1, 1) if use_aux else None

    def forward(self, encoder_features):
        if len(encoder_features) != 4:
            raise ValueError('EMCAD requires four encoder features')
        skips = list(reversed(encoder_features))
        features = skips[0]
        aux_logits = None
        for stage, (attention, refine) in enumerate(zip(self.channel_attention, self.refinement, strict=True)):
            if stage:
                skip = skips[stage]
                features = self.upsample[stage - 1](features, skip.shape[-2:])
                features = features + self.gates[stage - 1](features, skip)
            supervise = stage == 3 and self.training and self.aux_head is not None
            if supervise:
                aux_logits = self.aux_head(features)
            features = refine(self.spatial_attention(attention(features)))
        return features, aux_logits
