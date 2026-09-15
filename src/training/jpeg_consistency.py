"""Paired JPEG fine-tuning with a fixed teacher and unchanged inference model."""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from src.losses import LossResult, SegmentationLoss


def freeze_except_jpeg(model):
    """Freeze both parameters and batch statistics outside the JPEG branch."""
    model.eval()
    model.requires_grad_(False)
    model.forensic_fusion.branch.train()
    model.forensic_fusion.branch.requires_grad_(True)


class BalancedFeatureConsistency(nn.Module):
    """Equal foreground/local-background cosine error, then equal eligible images."""

    def forward(self, student, teacher, mask, reliable):
        with torch.autocast(device_type=student.device.type, enabled=False):
            student = F.normalize(student.float(), dim=1, eps=1e-6)
            teacher = F.normalize(teacher.detach().float(), dim=1, eps=1e-6)
            area = F.interpolate(mask.float(), size=student.shape[-2:], mode="area")[:, 0]
            foreground = area >= 0.9
            nearby = F.max_pool2d((area > 0).float()[:, None], 11, 1, 5)[:, 0] > 0
            background = nearby & (area <= 0.01)
            nf = foreground.sum((1, 2))
            nb = background.sum((1, 2))
            eligible = reliable & (nf > 0) & (nb > 0)
            distance = 1 - (student * teacher).sum(1)
            positive = (distance * foreground).sum((1, 2)) / nf.clamp_min(1)
            negative = (distance * background).sum((1, 2)) / nb.clamp_min(1)
            value = (0.5 * (positive + negative) * eligible).sum() / eligible.sum().clamp_min(1)
            return value, eligible.sum()


@contextmanager
def capture_stride8(model):
    captured = []
    handle = model.forensic_fusion.fusion_blocks["8"].register_forward_pre_hook(
        lambda module, args: captured.append(args[1])
    )
    try:
        yield captured
    finally:
        handle.remove()


class JPEGPairObjective:
    """Average segmentation over both views; align the compressed view to teacher."""

    def __init__(self, teacher, criterion, weight, min_dice=0.8, threshold=0.47265625):
        self.teacher = teacher.eval().requires_grad_(False)
        self.criterion = criterion
        self.weight = weight
        self.min_dice = min_dice
        self.threshold = threshold
        self.consistency = BalancedFeatureConsistency()

    @classmethod
    def from_config(cls, cfg, device):
        from src.training.builders import build_model, configure_memory_format
        from src.eval.protocol import EvaluationProtocol

        checkpoint = cfg.paths.runs_path / cfg.train.finetune_from
        if (checkpoint.parent.parent / "holdout_claim.json").exists():
            raise ValueError("Cannot use a teacher after holdout evaluation was claimed")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        EvaluationProtocol.load(cfg.dataset.protocol_path).verify_run(saved["cfg"])
        teacher = build_model(cfg.model, pretrained=False)
        teacher.load_state_dict(saved[cfg.train.finetune_weights])
        teacher = configure_memory_format(teacher.to(device))
        criterion = SegmentationLoss(
            **cfg.loss.to_dict(), aux_weight=cfg.model.aux_weight, dct_aux_weight=cfg.model.dct_aux_weight
        )
        return cls(
            teacher,
            criterion,
            cfg.train.jpeg_consistency_weight,
            cfg.train.jpeg_teacher_min_dice,
            cfg.train.jpeg_teacher_threshold,
        )

    def __call__(self, model, batch):
        # Frozen decoder/fusion still propagate gradients to the JPEG branch.
        original_image = batch["image"].to(memory_format=torch.channels_last)
        compressed_image = batch["paired_image"].to(memory_format=torch.channels_last)
        original = model(original_image, jpeg=batch["jpeg"])
        with capture_stride8(model) as student_features:
            compressed = model(compressed_image, jpeg=batch["paired_jpeg"])
        with torch.no_grad(), capture_stride8(self.teacher) as teacher_features:
            reference = self.teacher(original_image, jpeg=batch["jpeg"])
        first = self.criterion(original, batch)
        second = self.criterion(compressed, batch)
        components = {key: 0.5 * (first.components[key] + second.components[key]) for key in first.components}
        diagnostics = {
            key: (
                first.diagnostics[key][0] + second.diagnostics[key][0],
                first.diagnostics[key][1] + second.diagnostics[key][1],
            )
            for key in first.diagnostics
        }
        target = batch["mask"] > 0.5
        prediction = reference["logits"].float().sigmoid() >= self.threshold
        axes = (1, 2, 3)
        dice = 2 * (prediction & target).sum(axes) / (prediction.sum(axes) + target.sum(axes) + 1e-6)
        available = torch.tensor(
            [sample.get("available", True) for sample in batch["jpeg"]], dtype=torch.bool, device=target.device
        )
        reliable = (dice >= self.min_dice) & target.flatten(1).any(1)
        alignment = compressed["logits"].sum() * 0
        eligible = torch.zeros((), device=target.device)
        if teacher_features and student_features:
            # The original view can contain PNG negatives; teacher fusion excludes them.
            alignment, eligible = self.consistency(
                student_features[0][available], teacher_features[0], batch["mask"][available], reliable[available]
            )
        components["jpeg_consistency"] = self.weight * alignment
        count = torch.tensor(len(target), device=target.device)
        diagnostics["jpeg_consistency_raw"] = (alignment.detach() * eligible, eligible)
        diagnostics["jpeg_consistency_coverage"] = (eligible, count)
        return LossResult(sum(components.values()), components, diagnostics)
