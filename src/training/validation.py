from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.data.letterbox import Letterbox
from src.losses import LossMeter, build_loss
from src.progress import ConsoleProgress
from src.training.metric import AICAccumulator, AICResult

if TYPE_CHECKING:
    from src.config import ExperimentConfig
    from src.training.builders import AmpContext


@dataclass(frozen=True)
class ValidationResult:
    """Histograms and the selected metrics from one validation pass."""

    accumulator: AICAccumulator
    tuned: AICResult
    resolution: str = "resized"
    loss_components: dict[str, float] = field(default_factory=dict)
    fixed: AICResult | None = None

    @property
    def operating_point(self) -> tuple[float, float, float]:
        return (self.tuned.mask_threshold, self.tuned.cls_threshold, self.tuned.min_area)


@torch.no_grad()
def validate(
    model,
    loader: Iterable[dict[str, torch.Tensor]],
    amp: AmpContext,
    config: ExperimentConfig,
    device: torch.device,
    *,
    thresholds=None,
) -> ValidationResult:
    model.eval()
    n_bins = config.eval.n_bins
    acc = AICAccumulator(n_bins=n_bins)
    meter = LossMeter()
    criterion = build_loss(config, aux_weight=0.0, dct_aux_weight=0.0)

    for batch in ConsoleProgress.iterate(loader, "Валидация, батчи"):
        images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
        fmap = batch["fmap"].to(device, non_blocking=True) if "fmap" in batch else None

        with amp.autocast():
            kwargs = {"valid_mask": batch["valid_mask"].to(device)} if "valid_mask" in batch else {}
            out = model(images, fmap, **kwargs)

        # Comparable main-head loss on the network grid; AIC can use original GT.
        loss_batch = {key: batch[key].to(device) for key in ("mask", "label", "valid_mask") if key in batch}
        meter.update(criterion(out, loss_batch), len(images))
        probs = torch.sigmoid(out["logits"].float())
        cls = torch.sigmoid(out["cls_logits"].float()).reshape(-1)

        if config.eval.resolution == "original":
            if "original_mask" not in batch:
                raise ValueError("original validation requires original_mask from the dataset")
            for index, mask in enumerate(batch["original_mask"]):
                content = batch["content_size"][index] if "content_size" in batch else None
                restored = Letterbox.restore(probs[index:index + 1], mask.shape[-2:], content)
                _update_histograms(acc, restored, mask.to(device).reshape(1, 1, *mask.shape[-2:]),
                                   cls[index:index + 1])
        else:
            masks = batch["mask"].to(device, non_blocking=True)
            if "content_size" in batch:
                for index, content in enumerate(batch["content_size"]):
                    _update_histograms(acc, Letterbox.crop(probs[index:index + 1], content),
                                       Letterbox.crop(masks[index:index + 1], content), cls[index:index + 1])
            else:
                _update_histograms(acc, probs, masks, cls)

    if thresholds is None:
        ConsoleProgress.info("Подбор порогов маски, классификации и минимальной площади по AIC")
        tuned = acc.best(config.eval.mask_thresholds, config.eval.cls_thresholds, config.eval.min_areas)
    else:
        if thresholds.mask_threshold >= 1 or thresholds.mask_threshold * n_bins != int(thresholds.mask_threshold * n_bins):
            raise ValueError('Frozen mask threshold must match an exact histogram boundary')
        tuned = acc.evaluate(thresholds.mask_threshold, thresholds.cls_threshold, thresholds.min_area)
    ConsoleProgress.info(f"Оценка завершена: {tuned}")
    return ValidationResult(accumulator=acc, tuned=tuned, resolution=config.eval.resolution,
                            loss_components=meter.compute(), fixed=acc.evaluate(.5, .0, .0))


def _update_histograms(acc: AICAccumulator, probs, masks, cls) -> None:
    batch_size = probs.shape[0]
    n_bins = acc.n_bins
    flat = probs.clamp(0, 1).reshape(batch_size, -1)
    idx = (flat * n_bins).long().clamp_(max=n_bins - 1)
    offset = torch.arange(batch_size, device=idx.device).unsqueeze(1) * n_bins
    gt = masks.reshape(batch_size, -1) > 0.5
    hist_all = torch.bincount((idx + offset).reshape(-1), minlength=batch_size * n_bins)
    hist_gt = torch.bincount((idx + offset)[gt], minlength=batch_size * n_bins)
    acc.update_hist(
        hist_all.reshape(batch_size, n_bins).cpu().numpy(),
        hist_gt.reshape(batch_size, n_bins).cpu().numpy(),
        gt.sum(1).cpu().numpy(),
        np.full(batch_size, flat.shape[1], dtype=np.int64),
        cls.cpu().numpy(),
    )



