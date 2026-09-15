"""Full-frame original/recompressed pairs with identical target geometry."""

import cv2
import numpy as np

from src.data.dataset import AIIJCDataset
from src.data.data_sample import DataSample
from src.forensic.jpeg_input import JPEGInput


class JPEGPairDataset(AIIJCDataset):
    def __init__(self, *args, pair_quality=(80, 95), **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_quality = pair_quality
        # This bounded recipe uses unchanged full frames in both arms.
        self.use_augmentations = False

    def __getitem__(self, index):
        output = super().__getitem__(index)
        image = self.load_image(self._image_path(self.df.iloc[index]))
        quality = int(self._make_rng(index).integers(self.pair_quality[0], self.pair_quality[1] + 1))
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("Failed to encode paired JPEG")
        compressed = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        jpeg = JPEGInput.read(encoded.tobytes(), include_coefficients=self.jpeg_coefficients)
        h, w = compressed.shape[:2]
        sample = DataSample(image=compressed, fmap=np.zeros((1, h // 8, w // 8), np.float32))
        prepared = self.preprocessor.to_output(self.preprocessor.resize(sample))
        output["paired_image"] = prepared["image"]
        output["paired_jpeg"] = jpeg.tensors()
        output["paired_quality"] = quality
        return output
