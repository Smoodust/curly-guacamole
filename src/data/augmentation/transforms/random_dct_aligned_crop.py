from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.data.augmentation.base import AIIJCAugmentation, AugmentationStage, require_rng
from src.data.data_sample import DataSample
from src.data.utils import align8


class RandomDCTAlignedCrop(AIIJCAugmentation):
    def __init__(self, crop_scale_range: tuple[float, float], full_frame: bool,
                 foreground_probability: float = 0.0):
        super().__init__()
        self.crop_scale_range = crop_scale_range
        self.full_frame = full_frame
        self.foreground_probability = foreground_probability

    def sample_crop(
            self,
            height: int,
            width: int,
            rng: np.random.Generator,
    ) -> tuple[int, int, int]:
        """Sample square crop with size and coordinates divisible by 8."""
        max_side = align8(min(height, width))
        if max_side < 8:
            raise ValueError(f"image {height}x{width} is smaller than 8x8")

        side = int(round(np.sqrt(rng.uniform(*self.crop_scale_range)) * min(height, width)))
        side = min(align8(max(64, side)), max_side)

        top = align8(int(rng.integers(0, height - side + 1)))
        left = align8(int(rng.integers(0, width - side + 1)))
        return top, left, side

    def apply(self, sample: DataSample, rng: np.random.Generator | None = None) -> DataSample:
        if self.full_frame:
            return sample

        rng = require_rng(rng, AugmentationStage.AFTER_FORENSICS)
        height, width = sample.image.shape[:2]
        top, left, side = self.sample_crop(height, width, rng)
        if (sample.mask is not None and self.foreground_probability > 0
                and rng.random() < self.foreground_probability):
            # Only choose pixels reachable by an 8-aligned crop.
            points = np.argwhere(sample.mask[:align8(height), :align8(width)] > 0.5)
            if len(points):
                y, x = points[int(rng.integers(len(points)))]
                top = self._origin_containing(int(y), height, side, rng)
                left = self._origin_containing(int(x), width, side, rng)

        image = sample.image[top:top + side, left:left + side]
        mask = (
            sample.mask[top:top + side, left:left + side]
            if sample.mask is not None
            else None
        )

        jpeg = sample.jpeg.crop(top, left, side, side) if sample.jpeg is not None else None
        return replace(sample, image=image, mask=mask, jpeg=jpeg)

    @staticmethod
    def _origin_containing(pixel: int, length: int, side: int, rng) -> int:
        first_block = (max(0, pixel - side + 1) + 7) // 8
        last_block = min(pixel, length - side) // 8
        return int(rng.integers(first_block, last_block + 1)) * 8
