from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn.functional as F
from torch import Tensor

from src.losses.legacy import LossResult, bce_loss
from src.losses.mask_batch import MaskBatch


class LossArm(ABC):
    """Контракт цели обучения из реестра: своё слагаемое по маске, остальное общее.

    Гейт-голова и глубокая супервизия (aux, DCT-aux) заведены так же, как в
    `SegmentationLoss`, и выдают ту же `LossResult` — иначе `LossMeter` и
    логирование эпохи пришлось бы разводить по целям. Диагностика (`dice_pos`,
    `dice_neg`) всегда считается по обычному Dice основной головы, а не по
    `mask_term`: иначе арм с другим слагаемым сравнивать с остальными было бы
    нечем. Aux-голова учится тем же слагаемым, что и основная, — иначе
    эксперимент проверял бы сразу две разные вещи.
    """

    def __init__(self, *, cls_weight: float = 0.3, aux_weight: float = 0.0,
                 dct_aux_weight: float = 0.0) -> None:
        for name, value in (("cls_weight", cls_weight), ("aux_weight", aux_weight),
                            ("dct_aux_weight", dct_aux_weight)):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self.cls_weight = float(cls_weight)
        self.aux_weight = float(aux_weight)
        self.dct_aux_weight = float(dct_aux_weight)

    def __call__(self, out: dict, batch: dict) -> LossResult:
        target = batch["mask"].float()
        valid = batch.get("valid_mask")
        components = dict(self.mask_term(out["logits"].float(), target, valid))

        positive = MaskBatch(out["logits"], target, valid).is_positive
        labels = batch.get("label")
        if labels is None:
            labels = positive.reshape_as(out["cls_logits"])
        if self.cls_weight > 0.0:
            components["cls"] = self.cls_weight * bce_loss(out["cls_logits"].float(), labels.float())

        if self.aux_weight > 0.0 and "aux_logits" in out:
            for key, value in self.mask_term(out["aux_logits"].float(), target, valid).items():
                components[f"aux_{key}"] = self.aux_weight * value

        if self.dct_aux_weight > 0.0 and "dct_aux_logits" in out:
            logits = out["dct_aux_logits"].float()
            size = logits.shape[-2:]
            if valid is not None:
                dct_valid = F.interpolate(valid.float(), size=size, mode="area")
                dct_target = F.interpolate(target * valid, size=size, mode="area") / dct_valid.clamp_min(1e-6)
            else:
                dct_valid = None
                dct_target = F.interpolate(target, size=size, mode="area")
            for key, value in self.mask_term(logits, dct_target, dct_valid).items():
                components[f"dct_aux_{key}"] = self.dct_aux_weight * value

        diagnostics = {
            "dice_pos": _masked_dice_sum(out["logits"], target, valid, positive),
            "dice_neg": _masked_dice_sum(out["logits"], target, valid, 1.0 - positive),
        }
        return LossResult(sum(components.values()), components, diagnostics)

    @abstractmethod
    def mask_term(self, logits: Tensor, targets: Tensor, valid_mask: Tensor | None) -> dict[str, Tensor]:
        """Именованные слагаемые по маске для одной головы (например, `bce`/`dice`)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        options = ", ".join(f"{name}={value!r}" for name, value in sorted(vars(self).items()))
        return f"{type(self).__name__}({options})"


def _masked_dice_sum(logits: Tensor, targets: Tensor, valid_mask: Tensor | None,
                      keep: Tensor) -> tuple[Tensor, Tensor]:
    per_image = MaskBatch(logits, targets, valid_mask).dice()
    return (per_image * keep).sum(), keep.sum()


__all__ = ["LossArm"]
