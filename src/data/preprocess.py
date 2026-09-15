"""Resize and convert samples while preserving the baseline interpolation contract."""

from dataclasses import replace

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.data.data_sample import DataSample
from src.data.targets import mask_to_tensor


def image_resize(sample: DataSample, size: int) -> DataSample:
    image = cv2.resize(sample.image, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = (cv2.resize(sample.mask, (size, size), interpolation=cv2.INTER_LINEAR)
            if sample.mask is not None else None)
    return replace(sample, image=image, mask=mask)


def imagenet_normalize(sample: DataSample) -> DataSample:
    transform = A.Compose([
        A.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        ToTensorV2(),
    ])

    image = transform(image=sample.image)["image"]
    return replace(sample, image=image)


class SamplePreprocessor:
    """Resize image/mask together and produce normalized tensors."""

    def __init__(self, image_size: int):
        self.image_size = image_size

    def resize(self, sample: DataSample) -> DataSample:
        return image_resize(sample, self.image_size)

    def to_output(self, sample: DataSample) -> dict[str, torch.Tensor]:
        sample = imagenet_normalize(sample)
        output = {"image": sample.image}
        if sample.mask is not None:
            mask = mask_to_tensor(sample.mask)
            output["mask"] = mask
            output["label"] = torch.tensor([float(mask.max() > 0)], dtype=torch.float32)
        return output
