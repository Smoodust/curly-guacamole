"""Native Haar detail extraction on the model device and late EMCAD fusion."""

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


class WaveletPreprocessor(nn.Module):
    """uint8 RGB -> signed Haar details and their absolute responses.

    Extract before reduction; average absolute responses separately so opposing
    signs cannot erase texture energy. size is the equivalent image size: the
    output grid is size/2. No wavelet maps are built or transferred by workers.
    """

    def __init__(self, size):
        super().__init__()
        if type(size) is not int or size <= 0 or size % 32:
            raise ValueError('wavelet_image_size must be a positive multiple of 32')
        self.size = size
        haar = torch.tensor([[[1., -1.], [1., -1.]],
                             [[1., 1.], [-1., -1.]],
                             [[1., -1.], [-1., 1.]]]) / 2
        luma = torch.tensor([.299, .587, .114]) / 255
        # Fuse RGB->Y and the three fixed Haar filters into one convolution.
        self.register_buffer('kernels', haar[:, None] * luma[None, :, None, None], persistent=False)

    @torch.no_grad()
    def forward(self, images):
        views = []
        for image in images:
            if image.ndim != 3 or image.shape[0] != 3 or image.dtype != torch.uint8:
                raise ValueError('native_rgb must contain uint8 CHW RGB images')
            if image.device != self.kernels.device:
                raise ValueError('native_rgb must be on the model device')
            if min(image.shape[-2:]) < 1:
                raise ValueError('native_rgb images must be nonempty')
            # FP32 avoids AMP rounding weak forensic signals before pooling.
            autocast_off = (nullcontext() if image.device.type == 'meta' else
                            torch.autocast(image.device.type, enabled=False))
            with autocast_off:
                x = image[None].float()
                h, w = image.shape[-2:]
                if h % 2 or w % 2:
                    x = F.pad(x, (0, w % 2, 0, h % 2), mode='replicate')
                details = F.conv2d(x, self.kernels.float(), stride=2)
                features = torch.cat((details, details.abs()), dim=1)
                views.append(F.adaptive_avg_pool2d(features, (self.size // 2, self.size // 2)))
        return torch.cat(views).contiguous(memory_format=torch.channels_last)


class WaveletBranch(nn.Module):
    """Small detail CNN and context-conditioned residual before the mask head."""

    def __init__(self, global_channels, size, use_aux):
        super().__init__()
        self.preprocess = WaveletPreprocessor(size)
        # Learned convolutions run on bounded grids, never on native RGB sizes.
        self.stem = nn.Sequential(nn.Conv2d(6, 16, 3, stride=2, padding=1), nn.GELU(),
                                  nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.GELU(),
                                  nn.Conv2d(16, 24, 1), nn.GELU())
        self.context = nn.Conv2d(global_channels, 16, 1)
        self.fusion = nn.Sequential(nn.Conv2d(40, 40, 3, padding=1, groups=40), nn.GELU(),
                                    nn.Conv2d(40, global_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(()))
        self.aux_head = nn.Conv2d(24, 1, 1) if use_aux else None

    def forward(self, native_rgb, global_features):
        detail = self.stem(self.preprocess(native_rgb))
        aux = self.aux_head(detail) if self.training and self.aux_head is not None else None
        detail = F.interpolate(detail, global_features.shape[-2:], mode='bilinear', align_corners=False)
        residual = self.fusion(torch.cat((self.context(global_features), detail), dim=1))
        return global_features + self.gamma * residual, aux
