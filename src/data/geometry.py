"""Restore fixed-grid probabilities to the original image resolution."""

import torch.nn.functional as F


def restore_probability(probability, size):
    return F.interpolate(probability, size=size, mode="bilinear", align_corners=False)
