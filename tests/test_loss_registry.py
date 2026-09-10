"""Tests for the pluggable loss registry (src/losses), on top of the legacy
SegmentationLoss/LossMeter/dct_aux machinery covered by tests/test_losses.py.
"""

from dataclasses import replace

import pytest
import torch

from src.config import ExperimentConfig, LossConfig, load_experiment_config
from src.losses import build_loss, compute_loss, create_loss, is_loss, list_losses, soft_false_positive
from src.losses.pixel import PixelRegionLoss

AIC_AREA_THRESHOLD = 0.01


def _batch(kind, *, size=16, valid_mask=None):
    """kind: 'mixed', 'positive' or 'negative' frames in a batch of four."""
    torch.manual_seed(0)
    logits = torch.randn(4, 1, size, size, requires_grad=True)
    mask = torch.zeros(4, 1, size, size)
    positives = {"mixed": 2, "positive": 4, "negative": 0}[kind]
    mask[:positives, :, 1:4, 1:4] = 1.0
    batch = {"mask": mask, "label": (mask.flatten(1).sum(1, keepdim=True) > 0).float()}
    if valid_mask is not None:
        batch["valid_mask"] = valid_mask
    out = {"logits": logits, "aux_logits": logits, "cls_logits": torch.randn(4, 1)}
    return logits, out, batch


def test_registry_lists_every_experiment_arm():
    assert list_losses() == ["aic_harmonic", "aic_surrogate", "balanced_bce_dice", "bce_dice",
                             "focal_dice", "focal_tversky"]
    assert is_loss("aic_surrogate")
    assert not is_loss("nonexistent")


def test_unknown_loss_and_unknown_option_fail_early():
    with pytest.raises(ValueError, match="Unknown loss"):
        create_loss("nonexistent")
    with pytest.raises(TypeError):
        create_loss("bce_dice", typo=1.0)


def test_default_objective_reproduces_the_baseline_formula():
    """bce_dice is the control arm, so it must equal the pre-registry compute_loss."""
    _, out, batch = _batch("mixed")
    assert create_loss("bce_dice", aux_weight=0.4)(out, batch).total.item() == pytest.approx(
        compute_loss(out, batch, 0.4).item(), abs=1e-6)


@pytest.mark.parametrize("name", list_losses())
@pytest.mark.parametrize("kind", ["mixed", "positive", "negative"])
def test_every_objective_is_finite_and_trains_both_heads(name, kind):
    """A batch without positives (or without negatives) must still produce gradient."""
    logits, out, batch = _batch(kind)
    loss = create_loss(name, aux_weight=0.4)(out, batch).total
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


@pytest.mark.parametrize("name", list_losses())
def test_padding_changes_neither_value_nor_gradient(name):
    """Letterbox invariant: everything outside valid_mask must be ignored."""
    valid = torch.zeros(4, 1, 16, 16)
    valid[:, :, :8, :12] = 1.0
    logits, out, batch = _batch("mixed", valid_mask=valid)

    loss = create_loss(name, aux_weight=0.4)(out, batch).total
    padded = logits.detach().clone()
    padded[valid == 0] = 100.0
    moved = create_loss(name, aux_weight=0.4)(
        {**out, "logits": padded, "aux_logits": padded}, batch).total

    assert loss.item() == pytest.approx(moved.item(), abs=1e-5)
    loss.backward()
    assert not logits.grad[valid == 0].any()


@pytest.mark.parametrize("name", list_losses())
def test_no_objective_is_dominated_by_a_single_term_at_initialisation(name):
    """Random logits mean ~50% predicted area; an unbounded term would swamp the rest."""
    logits, out, batch = _batch("mixed")
    create_loss(name, aux_weight=0.4)(out, batch).total.backward()
    baseline_logits, baseline_out, baseline_batch = _batch("mixed")
    create_loss("bce_dice", aux_weight=0.4)(baseline_out, baseline_batch).total.backward()

    ratio = logits.grad.abs().sum() / baseline_logits.grad.abs().sum()
    assert 0.2 < ratio < 5.0, f"{name} gradient scale is {ratio:.1f}x the baseline"


@pytest.mark.parametrize("name", list_losses())
def test_every_objective_wires_the_dct_aux_head(name):
    """The DCT auxiliary head shares the same mask_term as the main head, for any arm."""
    logits, out, batch = _batch("mixed")
    out = {**out, "dct_aux_logits": logits[:, :, ::4, ::4]}

    loss = create_loss(name, aux_weight=0.4, dct_aux_weight=0.2)(out, batch)

    assert torch.isfinite(loss.total)
    assert any(key.startswith("dct_aux_") for key in loss.components)


@pytest.mark.parametrize("mode", ["sigmoid", "rational"])
def test_bounded_false_positive_modes_follow_the_metric_threshold(mode):
    area = torch.tensor([0.0, 0.002, AIC_AREA_THRESHOLD, 0.05, 0.5])
    value = soft_false_positive(area, threshold=AIC_AREA_THRESHOLD, mode=mode)

    assert value[0].item() == pytest.approx(0.0, abs=1e-3)
    assert value[2].item() == pytest.approx(0.5, abs=1e-6)
    assert torch.all(value <= 1.0)
    assert torch.all(value[1:] > value[:-1])


def test_hinge_false_positive_is_one_at_the_metric_threshold():
    value = soft_false_positive(torch.tensor([AIC_AREA_THRESHOLD]), threshold=AIC_AREA_THRESHOLD,
                                mode="hinge")
    assert value.item() == pytest.approx(1.0, abs=1e-6)


def test_unknown_false_positive_mode_fails_early():
    with pytest.raises(ValueError, match="unknown soft false positive mode"):
        soft_false_positive(torch.zeros(1), mode="linear")


def test_aic_surrogate_frees_negatives_below_the_metric_threshold():
    """A blob under 1% costs the baseline a full Dice penalty but the metric nothing."""
    logits = torch.full((1, 1, 100, 100), -12.0)
    logits[:, :, :5, :10] = 12.0  # 0.5% of the frame, half of the metric threshold
    out = {"logits": logits, "cls_logits": torch.zeros(1, 1)}
    batch = {"mask": torch.zeros(1, 1, 100, 100), "label": torch.zeros(1, 1)}

    baseline = PixelRegionLoss(cls_weight=0.0)(out, batch).total
    surrogate = create_loss("aic_surrogate", cls_weight=0.0)(out, batch).total

    assert baseline.item() > 0.9
    assert surrogate.item() < 0.3


def test_balanced_bce_lifts_the_gradient_of_a_small_mask():
    """The point of the arm: a mask worth 0.25% of the pixels must still be felt."""
    logits, out, batch = _batch("positive", size=64)
    PixelRegionLoss(cls_weight=0.0, dice_weight=0.0)(out, batch).total.backward()
    baseline = logits.grad[batch["mask"] > 0].abs().sum().item()

    logits, out, batch = _batch("positive", size=64)
    create_loss("balanced_bce_dice", cls_weight=0.0, dice_weight=0.0)(out, batch).total.backward()
    balanced = logits.grad[batch["mask"] > 0].abs().sum().item()

    assert balanced > 5.0 * baseline


def test_loss_config_defaults_keep_legacy_yaml_on_the_baseline():
    config = load_experiment_config("configs/baseline_mixed_original.yaml")

    assert config.loss == LossConfig()
    assert config.loss.name == "bce_dice"
    assert type(build_loss(config)).__name__ == "SegmentationLoss"
    assert build_loss(config).aux_weight == config.model.aux_weight


def test_loss_config_reaches_the_objective_and_the_snapshot():
    config = load_experiment_config("configs/loss_l4_aic_surrogate.yaml")
    loss = build_loss(config)

    assert config.loss.name == "aic_surrogate"
    assert loss.fpr_weight == config.loss.kwargs["fpr_weight"]
    assert config.to_flat_dict()["loss_name"] == "aic_surrogate"
    assert config.to_dict()["loss"]["kwargs"] == config.loss.kwargs


def test_loss_kwargs_may_not_shadow_the_shared_weights():
    raw = load_experiment_config("configs/baseline_mixed_original.yaml").to_dict()
    raw["loss"] = {"name": "bce_dice", "kwargs": {"aux_weight": 1.0}}

    with pytest.raises(ValueError, match="must not override cls_weight or aux_weight"):
        ExperimentConfig.from_dict(raw, base_dir="configs")


def test_unknown_loss_key_fails_early():
    raw = load_experiment_config("configs/baseline_mixed_original.yaml").to_dict()
    raw["loss"] = {"name": "bce_dice", "weight": 1.0}

    with pytest.raises(ValueError, match="unknown keys in loss: weight"):
        ExperimentConfig.from_dict(raw, base_dir="configs")


def test_resume_rejects_a_different_loss(tmp_path):
    from src.eval.protocol import EvaluationProtocol
    from src.training.engine import ExperimentRunner
    from src.training.runs import Run

    config = load_experiment_config("configs/loss_l4_aic_surrogate.yaml")
    config = replace(config, paths=replace(config.paths, runs_path=tmp_path),
                     train=replace(config.train, device="cpu", resume=True))
    # The arms run under the independent protocol, so the snapshot a resume is
    # checked against carries its provenance, exactly as ExperimentRunner.run writes it.
    snapshot = config.to_flat_dict()
    protocol = EvaluationProtocol.load(config.dataset.protocol_path)
    snapshot.update(protocol.provenance(train_originals=config.dataset.train_originals))
    run = Run.create(tmp_path, config.paths.run_name, resume=False)
    run.save_snapshot(snapshot)
    run.save_state({"model": {}}, "last.pt")
    run.close()

    ExperimentRunner(config)._check_resume_protocol()
    with pytest.raises(ValueError, match="different loss.name"):
        ExperimentRunner(replace(config, loss=LossConfig()))._check_resume_protocol()
