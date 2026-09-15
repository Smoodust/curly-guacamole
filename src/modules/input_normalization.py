"""Normalize transported uint8 RGB on the model's device."""

import torch
from torch import nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class ImageNetInputNormalization(nn.Module):
    """Float inputs are already normalized; uint8 inputs are raw RGB.

    Constants are created in FP32 on the input device, including when the model
    was explicitly converted to half precision. No checkpoint keys are added.
    """

    def forward(self, image):
        if image.dtype != torch.uint8:
            return image
        mean = image.new_tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float32)[None, :, None, None] * 255
        std = image.new_tensor(IMAGENET_DEFAULT_STD, dtype=torch.float32)[None, :, None, None] * 255
        return (image.float() - mean) * std.reciprocal()
