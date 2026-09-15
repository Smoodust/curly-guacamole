import numpy as np
import pytest

from src.config import EvalConfig
from src.training.metric import AICAccumulator


def test_weighted_dice_changes_threshold_selection_and_survives_oof(tmp_path):
    gt = np.zeros((3, 10000), dtype=bool)
    gt[0, :300] = True
    gt[1, :3000] = True
    probs = np.zeros_like(gt, dtype=float)
    probs[0, :300] = .9
    probs[0, 300:700] = .5
    probs[1, :3000] = .5
    probs[1, :1000] = .9
    plain = AICAccumulator(n_bins=4)
    weighted = AICAccumulator(n_bins=4, small_mask_weight=1.6)
    for acc in (plain, weighted):
        acc.update(probs, gt)
    assert plain.best((.25, .75)).mask_threshold == .25
    result = weighted.best((.25, .75))
    assert result.mask_threshold == .75
    assert result.dice_pos == pytest.approx((1.6 + .5) / 2.6)
    assert result.fpr_neg == 0
    weighted.save(tmp_path / 'oof.npz')
    restored = AICAccumulator.load(tmp_path / 'oof.npz')
    assert restored.small_mask_weight == 1.6
    assert restored.best((.25, .75)).as_dict() == result.as_dict()


def test_weights_use_gt_fraction_with_exact_boundaries_and_leave_fpr_unweighted():
    gt = np.zeros((5, 10000), dtype=bool)
    for row, size in enumerate([100, 500, 501]):
        gt[row, :size] = True
    pred = np.zeros_like(gt, dtype=float)
    pred[1] = gt[1]
    pred[3, :100] = 1  # One false alarm among two negatives.
    acc = AICAccumulator(n_bins=4, small_mask_weight=1.6)
    acc.update(pred, gt)
    result = acc.evaluate()
    assert result.dice_pos == pytest.approx(1.6 / 3.6)
    assert result.fpr_neg == .5


@pytest.mark.parametrize('weight', [0, -1, float('nan'), float('inf'), True])
def test_rejects_invalid_weight(weight):
    with pytest.raises(ValueError, match='small_mask_weight'):
        EvalConfig(selection_small_mask_weight=weight)


def test_old_oof_defaults_to_official_metric(tmp_path):
    acc = AICAccumulator(n_bins=4)
    acc.update(np.array([[1., 0.], [0., 0.]]), np.array([[1, 0], [0, 0]]))
    acc.save(tmp_path / 'new.npz')
    with np.load(tmp_path / 'new.npz') as saved:
        np.savez(tmp_path / 'old.npz', **{k: saved[k] for k in saved.files if k != 'small_mask_weight'})
    loaded = AICAccumulator.load(tmp_path / 'old.npz')
    assert loaded.small_mask_weight == 1.0
    assert loaded.evaluate().aic == acc.evaluate().aic


def test_yaml_controls_validation_selection_and_keeps_official_report():
    from dataclasses import replace

    import pandas as pd
    import torch

    from src.config import load_experiment_config
    from src.eval.diagnostics import EvaluationReport
    from src.training.builders import build_amp
    from src.training.validation import validate

    config = load_experiment_config('configs/baseline.yaml')
    assert config.eval.selection_small_mask_weight == 1.6
    assert config.to_flat_dict()['selection_small_mask_weight'] == 1.6
    config = replace(config, train=replace(config.train, device='cpu', amp='off'),
                     eval=replace(config.eval, n_bins=4, mask_thresholds=(.25, .75), cls_thresholds=(0.,)))
    gt = torch.zeros(3, 1, 100, 100)
    gt[0].reshape(-1)[:300] = 1
    gt[1].reshape(-1)[:3000] = 1
    probability = torch.full_like(gt, .01)
    probability[0].reshape(-1)[:300] = .9
    probability[0].reshape(-1)[300:700] = .5
    probability[1].reshape(-1)[:3000] = .5
    probability[1].reshape(-1)[:1000] = .9

    class Model(torch.nn.Module):
        def forward(self, image, fmap=None, **kwargs):
            return dict(logits=image[:, :1], cls_logits=torch.ones(len(image), 1))

    batch = dict(image=torch.logit(probability).repeat(1, 3, 1, 1), mask=gt,
                 label=torch.tensor([[1.], [1.], [0.]]), valid_mask=torch.ones_like(gt),
                 original_mask=[mask[0] for mask in gt])
    result = validate(Model(), [batch], build_amp(config.train), config, torch.device('cpu'))
    assert result.tuned.mask_threshold == .75
    assert result.tuned.dice_pos == pytest.approx(2.1/2.6)
    rows = pd.DataFrame({'target_kind': ['provided'] * 3})
    report = EvaluationReport(result.accumulator, rows, result.tuned).summary()
    assert report['selection']['aic'] == result.tuned.aic
    assert report['combined']['dice_pos'] == pytest.approx(.75)


def test_resume_rejects_weight_change_before_writing_snapshot(tmp_path):
    from dataclasses import replace

    from src.config import load_experiment_config
    from src.training.engine import ExperimentRunner
    from src.training.runs import Run

    config = load_experiment_config('configs/baseline.yaml')
    config = replace(config, paths=replace(config.paths, runs_path=tmp_path),
                     train=replace(config.train, device='cpu', amp='off', resume=True))
    run = Run.create(tmp_path, config.run_name, tensorboard=False)
    snapshot = config.to_flat_dict()
    snapshot['selection_small_mask_weight'] = 1.0  # A historical run used the implicit weight 1.
    run.save_snapshot(snapshot)
    (run.dir/'ckpt/last.pt').touch()
    with pytest.raises(ValueError, match='evaluation settings'):
        ExperimentRunner(config)._check_resume_protocol()
    assert run.snapshot['selection_small_mask_weight'] == 1.0
