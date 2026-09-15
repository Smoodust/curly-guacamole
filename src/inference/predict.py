"""Checkpoint inference and binary masks at the original image resolution."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import torch

from src.data.geometry import restore_probability
from src.training.builders import AmpContext
from src.training.metric import operating_bins
from src.training.transfer import BatchTransfer


@dataclass(frozen=True)
class ThresholdConfig:
    mask_threshold: float = 0.5
    cls_threshold: float = 0.0
    min_area: float = 0.0
    # 0 keeps the historical gate, which zeroes a doubted frame outright. Above 0
    # the gate instead raises that frame's threshold until its area drops below
    # the cap, so a wrongly gated positive keeps a partial mask.
    area_cap: float = 0.0
    # Thresholds are resolved on the validation histogram grid; inference must
    # use the same one or a tuned operating point does not reproduce here.
    n_bins: int = 256

    def __post_init__(self) -> None:
        for name in ("mask_threshold", "cls_threshold", "min_area", "area_cap"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if type(self.n_bins) is not int or self.n_bins <= 0:
            raise ValueError("n_bins must be a positive integer")
        if self.mask_threshold * self.n_bins != int(self.mask_threshold * self.n_bins):
            raise ValueError(
                f"mask_threshold {self.mask_threshold} is not a boundary of the {self.n_bins}-bin "
                "histogram grid; validation could not have selected it")

    @property
    def bin_index(self) -> int:
        return min(int(self.mask_threshold * self.n_bins), self.n_bins - 1)


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
        probabilities = restored[0, 0].cpu().numpy()
        bins, blank = self.operating_point(probabilities, cls_probability)
        if blank:
            mask = np.zeros(probabilities.shape, dtype=bool)
        else:
            mask = probabilities >= bins / self.thresholds.n_bins
        return mask.astype(np.uint8) * 255

    def operating_point(self, probabilities: np.ndarray, cls_probability: float) -> tuple[int, bool]:
        """Per-frame threshold bin and blanking flag, from the validation rule itself.

        The sweep that chose these thresholds calls the same `operating_bins`, on
        histograms of the same restored probabilities. Reimplementing the rule here
        would let inference and selection drift apart silently, and the reported
        score would stop describing the submission.
        """
        n_bins, start = self.thresholds.n_bins, self.thresholds.bin_index
        size = probabilities.size
        # Only bins at or above bin_index can change the answer: operating_bins
        # takes max(bin_index, cap_index), so a cap satisfied lower down still
        # resolves to bin_index. Histogramming the suprathreshold pixels alone
        # turns a full-frame pass into one over the predicted region, which on a
        # negative frame is a rounding error. Bins below start are filled with
        # the frame size, an area of 1.0, so cap_bins never selects them.
        pred_counts = np.full((1, n_bins), float(size))
        above = probabilities.reshape(-1)
        above = above[above >= start / n_bins]
        if above.size:
            index = np.clip((above * n_bins).astype(np.int32), start, n_bins - 1)
            histogram = np.bincount(index - start, minlength=n_bins - start)
            pred_counts[0, start:] = np.cumsum(histogram[::-1])[::-1]
        else:
            pred_counts[0, start:] = 0.0
        bins, blank = operating_bins(
            pred_counts,
            np.array([probabilities.size], dtype=np.float64),
            np.array([cls_probability], dtype=np.float64),
            self.thresholds.bin_index,
            self.thresholds.cls_threshold,
            self.thresholds.min_area,
            self.thresholds.area_cap,
        )
        return int(bins[0]), bool(blank[0])
