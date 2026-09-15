from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from src.data.augmentation.base import AIIJCAugmentation, AugmentationStage, require_rng
from src.data.data_sample import DataSample
from src.forensic.jpeg import luma_qtable
from src.forensic.jpeg_input import JPEGInput


class RandomJPEGRecompression(AIIJCAugmentation):
    def __init__(self, quality_range: tuple[int, int], probability: float) -> None:
        super().__init__()
        self.quality_range = quality_range
        self.probability = probability

    def jpeg_recompression(
            self,
            image: np.ndarray,
            rng: np.random.Generator | None,
            *, native=False, include_coefficients=False,
    ):
        """JPEG re-encode image and extract the resulting luminance qtable."""
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        quality = int(rng.integers(*self.quality_range))
        ok, buffer = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not ok:
            if native:
                raise ValueError('JPEG recompression failed')
            return image, None

        jpeg_bytes = buffer.tobytes()
        decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        if native:
            jpeg = JPEGInput.read(jpeg_bytes, include_coefficients=include_coefficients)
            return decoded, jpeg.qtable, jpeg
        return decoded, luma_qtable(jpeg_bytes)

    def apply(self, sample: DataSample, rng: np.random.Generator | None = None) -> DataSample:
        if self.probability <= 0:
            return sample

        rng = require_rng(rng, AugmentationStage.BEFORE_FORENSICS)
        if rng.random() >= self.probability:
            return sample

        if sample.jpeg is not None:
            image, qtable, jpeg = self.jpeg_recompression(
                sample.image, rng, native=True, include_coefficients=sample.jpeg.coefficients is not None)
            return replace(sample, image=image, qtable=qtable, jpeg=jpeg)
        image, qtable = self.jpeg_recompression(sample.image, rng)
        return replace(sample, image=image, qtable=qtable)
