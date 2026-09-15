from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.data.augmentation.base import AIIJCAugmentation, AugmentationStage, require_rng
from src.data.data_sample import DataSample


class RandomRotateFlip(AIIJCAugmentation):
    def __init__(self, full_frame):
        super().__init__()
        self.full_frame = full_frame

    @staticmethod
    def rotate_flip(
            array: np.ndarray,
            rotations: int,
            flip_horizontal: bool,
            flip_vertical: bool,
            spatial: tuple[int, int],
    ) -> np.ndarray:
        """Apply k*90 degree rotation and flips along the given spatial axes."""
        if rotations:
            array = np.rot90(array, rotations, axes=spatial)
        if flip_horizontal:
            array = np.flip(array, axis=spatial[1])
        if flip_vertical:
            array = np.flip(array, axis=spatial[0])

        return np.ascontiguousarray(array)

    def apply(self, sample: DataSample, rng: np.random.Generator | None = None) -> DataSample:
        rng = require_rng(rng, AugmentationStage.AFTER_FORENSICS)

        rotations = 0 if self.full_frame else int(rng.integers(4))
        flip_horizontal = rng.random() < 0.5
        flip_vertical = rng.random() < 0.2

        image = self.rotate_flip(
            sample.image,
            rotations,
            flip_horizontal,
            flip_vertical,
            spatial=(0, 1),
        )
        mask = (
            self.rotate_flip(
                sample.mask,
                rotations,
                flip_horizontal,
                flip_vertical,
                spatial=(0, 1),
            )
            if sample.mask is not None
            else None
        )

        jpeg = (sample.jpeg.transform(rotations, flip_horizontal, flip_vertical)
                if sample.jpeg is not None else None)
        return replace(sample, image=image, mask=mask, jpeg=jpeg)
