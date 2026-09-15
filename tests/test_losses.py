from dataclasses import replace

import pytest
import torch

from src.losses import LossMeter, SegmentationLoss, compute_loss, soft_dice_loss


def sample():
    target = torch.zeros(3, 1, 4, 4)
    target[0, :, :2] = 1
    target[1] = 1
    logits = torch.zeros_like(target, requires_grad=True)
    return {"logits": logits, "cls_logits": torch.zeros(3, 1, requires_grad=True)}, {
        "mask": target, "label": torch.tensor([[1.], [1.], [0.]])}


def test_all_image_dice_and_bce_train_negatives():
    out, batch = sample()
    result = SegmentationLoss(dice_weight=.75)(out, batch)
    expected = .75 * soft_dice_loss(out["logits"], batch["mask"])
    torch.testing.assert_close(result.components["dice"], expected)
    result.total.backward()
    assert out["logits"].grad[2].gt(0).all()
    assert result.total.item() == pytest.approx(sum(v.item() for v in result.components.values()))




def test_all_negative_batch_has_finite_loss_and_bce_gradient():
    out, batch = sample()
    batch["mask"].zero_()
    batch["label"].zero_()
    result = SegmentationLoss(aux_weight=.4)(
        {**out, "aux_logits": out["logits"]}, batch)
    for key in ("dice", "aux_dice"):
        assert result.components[key].item() > 0
    result.total.backward()
    assert out["logits"].grad.gt(0).all()


def test_legacy_wrapper_and_positive_diagnostics():
    out, batch = sample()
    result = SegmentationLoss()(out, batch)
    torch.testing.assert_close(result.total, compute_loss(out, batch))
    torch.testing.assert_close(result.components["dice"], soft_dice_loss(out["logits"], batch["mask"]))
    meter = LossMeter()
    meter.update(result, 3)
    out2 = {k: v[2:] for k, v in out.items()}
    batch2 = {k: v[2:] for k, v in batch.items()}
    meter.update(SegmentationLoss()(out2, batch2), 1)
    values = meter.compute()
    assert values["dice_pos"] == pytest.approx(soft_dice_loss(out["logits"][:2], batch["mask"][:2]).item())
    assert values["total"] == pytest.approx((3 * result.total.item() + SegmentationLoss()(out2, batch2).total.item()) / 4)


def test_loss_config_roundtrip_and_legacy_resume_guard(tmp_path):
    from src.config import ExperimentConfig, load_experiment_config
    from src.training.engine import ExperimentRunner
    from src.training.runs import Run

    old = load_experiment_config("configs/baseline.yaml")
    assert old.loss.aux_weight == .4
    raw = old.to_dict()
    raw["loss"] = {"aux_weight": .2, "dice_weight": .75}
    new = ExperimentConfig.from_dict(raw)
    assert ExperimentConfig.from_dict(new.to_dict()) == new
    assert new.to_flat_dict()["aux_weight"] == .2
    new = replace(new, paths=replace(new.paths, runs_path=tmp_path),
                  train=replace(new.train, device="cpu", amp="off", resume=True))
    run = Run.create(tmp_path, new.run_name, tensorboard=False)
    run.save_snapshot(old.to_flat_dict())
    (run.dir / "ckpt" / "last.pt").touch()
    with pytest.raises(ValueError, match="loss"):
        ExperimentRunner(new)._check_resume_protocol()


@pytest.mark.parametrize("options", [{"dice_scope": "typo"}, {"dice_weight": -1}, {"dice_weight": float("nan")}])
def test_invalid_loss_settings(options):
    from src.config import ExperimentConfig, load_experiment_config
    raw = load_experiment_config("configs/baseline.yaml").to_dict()
    raw["loss"] = options
    with pytest.raises(ValueError, match="loss"):
        ExperimentConfig.from_dict(raw)








def test_legacy_scalar_matches_independent_formula():
    out, batch = sample()
    out["aux_logits"] = out["logits"] + .7
    def reference(logits):
        probs = logits.sigmoid().flatten(1)
        target = batch["mask"].flatten(1)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["mask"]) + (
            1 - (2 * (probs * target).sum(1) + 1) / (probs.sum(1) + target.sum(1) + 1)).mean()
    expected = reference(out["logits"]) + .4 * reference(out["aux_logits"]) + .3 * (
        torch.nn.functional.binary_cross_entropy_with_logits(out["cls_logits"], batch["label"]))
    torch.testing.assert_close(compute_loss(out, batch, .4), expected)
