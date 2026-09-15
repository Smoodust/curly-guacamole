import numpy as np
import pytest
import torch
from torch import nn

from src.training.jpeg_consistency import BalancedFeatureConsistency, freeze_except_jpeg


def test_pair_dataset_preserves_mask_and_reads_the_encoded_jpeg(tmp_path):
    import cv2
    import pandas as pd
    from src.data.jpeg_pair import JPEGPairDataset
    from src.data.data_workspace import DataWorkspace
    from src.data.collation import ValidationCollator
    from src.training.transfer import BatchTransfer

    workspace = DataWorkspace(tmp_path)
    workspace.train_root.mkdir(parents=True)
    image = np.random.default_rng(3).integers(0, 256, (40, 56, 3), dtype=np.uint8)
    mask = np.zeros((40, 56), np.uint8)
    mask[8:25, 12:32] = 255
    cv2.imwrite(str(workspace.train_root / "image.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imwrite(str(workspace.train_root / "mask.png"), mask)
    rows = pd.DataFrame([dict(chng_img_path="image.jpg", gt_path="mask.png", is_negative=False)])
    ds = JPEGPairDataset(workspace, rows, True, 32, 42, forensic_mode="jpeg", pair_quality=(85, 85))
    sample = ds[0]
    assert sample["paired_quality"] == 85
    assert sample["image"].shape == sample["paired_image"].shape == (3, 32, 32)
    assert not torch.equal(sample["image"], sample["paired_image"])
    assert sample["jpeg"]["geometry"] == sample["paired_jpeg"]["geometry"]
    assert sample["jpeg"]["qtable"].max() == 1
    assert sample["paired_jpeg"]["qtable"].max() > 1
    batch = BatchTransfer("cpu")(ValidationCollator()([sample, sample]))
    assert len(batch["paired_jpeg"]) == 2 and batch["mask"].shape == (2, 1, 32, 32)
    assert torch.equal(batch["mask"][0], sample["mask"])


def test_paired_config_requires_teacher_and_valid_quality():
    from src.config import TrainConfig

    with pytest.raises(ValueError, match="teacher"):
        TrainConfig(jpeg_pair_training=True)
    with pytest.raises(ValueError, match="quality"):
        TrainConfig(jpeg_pair_quality_min=96, jpeg_pair_quality_max=80)


def test_foreground_background_have_equal_weight_and_teacher_is_detached():
    target = torch.zeros(1, 1, 4, 4)
    target[:, :, 0, 0] = 1
    teacher = torch.zeros(1, 2, 4, 4, requires_grad=True)
    teacher.data[:, 0] = 1
    student = teacher.detach().clone()
    student[:, :, 0, 0] = torch.tensor([0.0, 1.0])
    student.requires_grad_()
    loss, eligible = BalancedFeatureConsistency()(student, teacher, target, torch.tensor([True]))
    assert eligible.item() == 1
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert teacher.grad is None and student.grad.abs().sum() > 0


def test_mixed_cells_bad_teacher_and_empty_masks_do_not_align():
    teacher = torch.randn(3, 4, 2, 2)
    student = torch.randn_like(teacher, requires_grad=True)
    target = torch.zeros(3, 1, 2, 2)
    target[0, 0, 0, 0] = 0.5
    target[1, 0, 0, 0] = 1
    loss, n = BalancedFeatureConsistency()(student, teacher, target, torch.tensor([True, False, True]))
    assert n.item() == 0 and loss.item() == 0
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() == 0


def test_only_jpeg_parameters_and_normalization_train():
    model = nn.Module()
    model.encoder = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))
    model.forensic_fusion = nn.Module()
    model.forensic_fusion.branch = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))
    model.decoder = nn.BatchNorm2d(2)
    model.train()
    freeze_except_jpeg(model)
    assert not model.encoder.training and not model.decoder.training
    assert model.forensic_fusion.branch.training
    for name, p in model.named_parameters():
        assert p.requires_grad == name.startswith("forensic_fusion.branch.")
    x = torch.randn(2, 2, 4, 4)
    before = model.decoder.running_mean.clone()
    model.decoder(model.encoder(x) + model.forensic_fusion.branch(x)).sum().backward()
    assert torch.equal(before, model.decoder.running_mean)
    assert model.encoder[0].weight.grad is None
    assert model.forensic_fusion.branch[0].weight.grad is not None


class TinyFuse(nn.Module):
    def forward(self, rgb, jpeg):
        return rgb + jpeg


class TinySegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(1, 2, 1)
        self.forensic_fusion = nn.Module()
        self.forensic_fusion.branch = nn.Conv2d(1, 2, 1)
        self.forensic_fusion.fusion_blocks = nn.ModuleDict({"8": TinyFuse()})
        self.decoder = nn.Conv2d(2, 1, 1)

    def forward(self, image, jpeg):
        feature = self.forensic_fusion.branch(torch.cat([j["image"] for j in jpeg]))
        fused = self.forensic_fusion.fusion_blocks["8"](self.encoder(image), feature)
        logits = self.decoder(fused)
        return {"logits": logits, "cls_logits": logits.mean((2, 3))}


def test_complete_objective_has_both_views_and_no_teacher_or_frozen_gradients():
    import copy
    from src.training.jpeg_consistency import JPEGPairObjective
    from src.losses import SegmentationLoss

    torch.manual_seed(7)
    model = TinySegmenter()
    teacher = copy.deepcopy(model)
    freeze_except_jpeg(model)
    image = torch.randn(1, 1, 8, 8)
    compressed = image + 0.5 * torch.randn_like(image)
    mask = torch.zeros_like(image)
    mask[:, :, :4, :4] = 1
    batch = dict(
        image=image,
        paired_image=compressed,
        mask=mask,
        label=torch.ones(1, 1),
        jpeg=[{"image": image}],
        paired_jpeg=[{"image": compressed}],
    )
    criterion = SegmentationLoss(aux_weight=0)
    objective = JPEGPairObjective(teacher, criterion, 0.05, min_dice=0)
    first = criterion(model(image, jpeg=batch["jpeg"]), batch).total
    second = criterion(model(compressed, jpeg=batch["paired_jpeg"]), batch).total
    result = objective(model, batch)
    assert torch.allclose(result.total, 0.5 * (first + second) + result.components["jpeg_consistency"])
    assert result.diagnostics["jpeg_consistency_coverage"][0].item() == 1
    assert result.components["jpeg_consistency"] > 0
    result.total.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert model.encoder.weight.grad is None and model.decoder.weight.grad is None
    assert model.forensic_fusion.branch.weight.grad.abs().sum() > 0
    assert not teacher.forensic_fusion.fusion_blocks["8"]._forward_pre_hooks
    assert not model.forensic_fusion.fusion_blocks["8"]._forward_pre_hooks


def test_experiment_configs_differ_only_by_name_and_consistency_weight():
    from src.config import load_experiment_config

    control = load_experiment_config("configs/jpeg640_pair_branch_control_3ep.yaml")
    trial = load_experiment_config("configs/jpeg640_pair_s8_consistency_3ep.yaml")
    a, b = control.to_flat_dict(), trial.to_flat_dict()
    assert {key for key in a if a[key] != b[key]} == {"run_name", "jpeg_consistency_weight"}
    assert trial.train.finetune_from == "jpeg640_plain_nonunit_focus_3ep/ckpt/best.pt"
    assert trial.train.finetune_weights == "ema"
