"""Checkpoint inference and binary masks at the original image resolution."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import torch

from src.data.geometry import restore_probability
from src.training.builders import AmpContext
from src.training.transfer import BatchTransfer


@dataclass(frozen=True)
class ThresholdConfig:
    mask_threshold: float = 0.5
    cls_threshold: float = 0.0
    min_area: float = 0.0

    def __post_init__(self) -> None:
        for name in ("mask_threshold", "cls_threshold", "min_area"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class Prediction:
    image_path: str
    mask: np.ndarray


class Predictor:
    def __init__(self, model, thresholds: ThresholdConfig, amp: AmpContext) -> None:
        self.model = model.to(amp.device)
        self.thresholds = thresholds
        self.amp = amp

    def predict(self, loader: Iterable) -> Iterator[Prediction]:
        """Resize probabilities before thresholding; gate on the restored mask area."""
        self.model.eval()
        for batch in loader:
            # Leave inference/autocast contexts before yielding to caller code.
            with torch.inference_mode(), self.amp.autocast():
                kwargs = {'jpeg': BatchTransfer.move_jpeg(batch['jpeg'], self.amp.device)} if 'jpeg' in batch else {}
                output = self.model(batch['image'].to(self.amp.device), **kwargs)
                probabilities = output["logits"].float().sigmoid()
                cls_probs = output["cls_logits"].float().sigmoid().flatten()
            for index, image_path in enumerate(batch["image_path"]):
                size = tuple(int(value) for value in batch["original_size"][index])
                mask = self.binary_mask(probabilities[index:index + 1], float(cls_probs[index]), size)
                yield Prediction(image_path, mask)

    def binary_mask(
        self, probability: torch.Tensor, cls_probability: float, size: tuple[int, int],
    ) -> np.ndarray:
        if len(size) != 2 or min(size) <= 0:
            raise ValueError("original size must contain positive height and width")
        restored = restore_probability(probability, size)
        mask = restored[0, 0].cpu().numpy() >= self.thresholds.mask_threshold
        if cls_probability < self.thresholds.cls_threshold or mask.mean() < self.thresholds.min_area:
            mask[:] = False
        return mask.astype(np.uint8) * 255
