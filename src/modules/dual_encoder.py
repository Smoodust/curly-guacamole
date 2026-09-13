"""Two independent RGB/noise backbones with paper-inspired scale-wise FAM."""

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from src.modules.utils import build_timm_encoder


class SobelMagnitude(nn.Module):
    """Channel-wise |dx| + |dy|, with replicated edges and kernels / 8."""

    def __init__(self):
        super().__init__()
        dx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        self.register_buffer('kernels', torch.stack((dx, dx.T))[:, None], persistent=False)

    def forward(self, x):
        channels = x.shape[1]
        gradients = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'),
                             self.kernels.to(dtype=x.dtype).repeat(channels, 1, 1, 1),
                             groups=channels)
        return gradients.reshape(x.shape[0], channels, 2, *x.shape[-2:]).abs().sum(2)


class GuidedNoise(nn.Module):
    """Native self-guided residual + Sobel; fixed scaling, no image-wise stats."""

    def __init__(self, radius=2, epsilon=.01, scale=.25):
        super().__init__()
        if type(radius) is not int or radius < 1:
            raise ValueError('guided radius must be a positive integer')
        if not 0 < epsilon < float('inf') or not 0 < scale < float('inf'):
            raise ValueError('guided epsilon and scale must be finite and positive')
        self.radius, self.epsilon, self.scale = radius, epsilon, scale
        self.sobel = SobelMagnitude()

    def box(self, x):
        r = self.radius
        return F.avg_pool2d(F.pad(x, (r, r, r, r), mode='replicate'), 2*r+1, stride=1)

    @torch.no_grad()
    def forward(self, images, size):
        views = []
        for image in images:
            if image.ndim != 3 or image.shape[0] != 3 or image.dtype != torch.uint8:
                raise ValueError('native_rgb must contain uint8 CHW RGB images')
            if min(image.shape[-2:]) < 1 or image.device != self.sobel.kernels.device:
                raise ValueError('native_rgb must be nonempty and on the model device')
            context = nullcontext() if image.device.type == 'meta' else torch.autocast(image.device.type, enabled=False)
            with context:
                x = image[None].float() / 255
                mean = self.box(x)
                variance = (self.box(x.square()) - mean.square()).clamp_min(0)
                a = variance / (variance + self.epsilon)
                b = mean * (1 - a)
                guided = self.box(a) * x + self.box(b)
                noise = ((x - guided).abs() + self.sobel(x)) / self.scale
                # Reduce first in mixed geometries, then enlarge only if needed.
                reduced = tuple(min(a, b) for a, b in zip(x.shape[-2:], size))
                noise = F.adaptive_avg_pool2d(noise, reduced)
                if reduced != tuple(size):
                    noise = F.interpolate(noise, size=size, mode='bilinear', align_corners=False)
                views.append(noise)
        return torch.cat(views).contiguous(memory_format=torch.channels_last)


class DynamicConvolution(nn.Module):
    """Per-image softmax mixture of kernels, evaluated as weighted expert outputs."""

    def __init__(self, channels, experts=4):
        super().__init__()
        self.routing = nn.Linear(channels, experts)
        self.experts = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1, bias=False) for _ in range(experts)])

    def forward(self, x):
        weights = self.routing(x.mean((-2, -1))).float().softmax(-1).to(x.dtype)
        outputs = [expert(x) * weights[:, i, None, None, None]
                   for i, expert in enumerate(self.experts)]
        return sum(outputs)


class FeatureAggregation(nn.Module):
    """FAM equations 5-7 with bounded hidden width and GroupNorm instead of BN."""

    def __init__(self, rgb_channels, noise_channels, output_channels, hidden=16):
        super().__init__()
        self.rgb_residual = nn.Sequential(
            SobelMagnitude(), nn.Conv2d(rgb_channels, hidden, 1),
            DynamicConvolution(hidden), nn.Conv2d(hidden, rgb_channels, 5, padding=2))
        self.noise_residual = nn.Sequential(
            nn.Conv2d(noise_channels, hidden, 1), nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(hidden, noise_channels, 7, padding=3))
        self.project = nn.Sequential(
            nn.Conv2d(rgb_channels + noise_channels, output_channels, 1, bias=False),
            nn.GroupNorm(32, output_channels), nn.ReLU())

    def forward(self, rgb, noise):
        if rgb.shape[-2:] != noise.shape[-2:]:
            raise ValueError('dual encoder feature grids must match')
        return self.project(torch.cat((rgb + self.rgb_residual(rgb),
                                       noise + self.noise_residual(noise)), dim=1))


class DualEncoder(nn.Module):
    """Drop-in four-scale feature producer; native RGB comes from wavelet pipeline."""

    strides = (4, 8, 16, 32)
    channels = (64, 128, 320, 512)

    def __init__(self, rgb_name, noise_name, *, pretrained=True,
                 radius=2, epsilon=.01, scale=.25, hidden=16):
        super().__init__()
        self.rgb, rgb_strides, rgb_channels = build_timm_encoder(rgb_name, pretrained=pretrained)
        self.noise, noise_strides, noise_channels = build_timm_encoder(noise_name, pretrained=pretrained)
        if tuple(rgb_strides) != self.strides or tuple(noise_strides) != self.strides:
            raise ValueError('dual encoders must expose exactly strides 4/8/16/32')
        self.preprocess = GuidedNoise(radius, epsilon, scale)
        self.fusions = nn.ModuleList([FeatureAggregation(r, n, c, hidden)
                                     for r, n, c in zip(rgb_channels, noise_channels, self.channels)])

    def forward(self, image, native_rgb):
        if not isinstance(native_rgb, (list, tuple)) or len(native_rgb) != image.shape[0]:
            raise ValueError('dual encoder requires one native_rgb image per sample')
        noise = self.preprocess(native_rgb, image.shape[-2:])
        return [fuse(rgb, noise) for fuse, rgb, noise in
                zip(self.fusions, self.rgb(image), self.noise(noise))]
