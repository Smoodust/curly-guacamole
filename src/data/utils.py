"""Compatibility imports for data helpers; new code uses sample_io/preprocess."""

from src.data.preprocess import image_resize, imagenet_normalize
from src.data.sample_io import read_image

__all__ = ["align8", "read_image", "image_resize", "imagenet_normalize"]


def align8(value: int) -> int:
    """Round down to the nearest complete DCT block."""
    return (int(value) // 8) * 8
