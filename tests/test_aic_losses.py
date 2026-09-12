import pytest
import torch

from src.losses import (
    AicHarmonicLoss,
    AicSurrogateLoss,
    MaskBatch,
    SegmentationLoss,
    build_criterion,
    harmonic_aic,
    soft_false_positive,
)

ARMS = ('jpeg576_pretrained_18ep_full_train_finetune_3ep_aic_surrogate',
        'jpeg576_pretrained_18ep_full_train_finetune_3ep_aic_harmonic')


def _batch(positive_first=True):
    """Two frames, the first edited and the second clean."""
    mask = torch.zeros(2, 1, 8, 8)
    if positive_first:
        mask[0, :, 2:6, 2:6] = 1
    logits = torch.zeros_like(mask, requires_grad=True)
    cls = torch.zeros(2, 1, requires_grad=True)
    return {'logits': logits, 'cls_logits': cls}, {'mask': mask}, logits, cls


def test_soft_false_positive_is_half_at_threshold_and_bounded():
    area = torch.tensor([0., .01, .5, 1.])
    for mode in ('sigmoid', 'rational'):
        value = soft_false_positive(area, threshold=.01, mode=mode)
        assert value[1] == pytest.approx(.5, abs=1e-5)
        assert value.min() >= 0 and value.max() <= 1
    # Hinge is exactly 1 at the threshold and unbounded above it.
    assert soft_false_positive(area, threshold=.01, mode='hinge')[1] == pytest.approx(1., abs=1e-5)
    assert soft_false_positive(area, threshold=.01, mode='hinge')[3] > 1
    # Below the threshold sigmoid nearly vanishes while rational still charges.
    tenth = torch.tensor([.001])
    assert soft_false_positive(tenth, threshold=.01, mode='sigmoid').item() < .02
    assert soft_false_positive(tenth, threshold=.01, mode='rational').item() > .05
    assert soft_false_positive(area, threshold=.01, mode='sigmoid')[0] < 1e-6
    assert soft_false_positive(area, threshold=.01, mode='rational')[0] == 0
    with pytest.raises(ValueError):
        soft_false_positive(area, threshold=.01, mode='linear')
    with pytest.raises(ValueError):
        soft_false_positive(area, threshold=0.)


def test_harmonic_aic_matches_the_metric_at_the_extremes():
    perfect = harmonic_aic(torch.tensor(1.), torch.tensor(0.))
    assert perfect == pytest.approx(1., abs=1e-5)
    assert harmonic_aic(torch.tensor(1.), torch.tensor(1.)) == pytest.approx(0., abs=1e-5)
    assert harmonic_aic(torch.tensor(0.), torch.tensor(0.)) == pytest.approx(0., abs=1e-5)


def test_surrogate_charges_dice_on_positives_and_area_on_negatives():
    out, batch, logits, cls = _batch()
    criterion = AicSurrogateLoss(area_threshold=.01, area_mode='sigmoid')
    result = criterion(out, batch)
    assert set(result.components) == {'bce', 'dice', 'fpr', 'cls'}

    # The negative frame pays through fpr only, so its Dice gradient is zero.
    dice_grad = torch.autograd.grad(result.components['dice'], logits, retain_graph=True)[0]
    assert dice_grad[0].abs().sum() > 0 and dice_grad[1].eq(0).all()
    fpr_grad = torch.autograd.grad(result.components['fpr'], logits, retain_graph=True)[0]
    assert fpr_grad[1].abs().sum() > 0 and fpr_grad[0].eq(0).all()

    result.total.backward()
    assert torch.isfinite(logits.grad).all() and cls.grad.abs().sum() > 0


def test_surrogate_dice_is_zero_without_positives_but_negatives_still_pay():
    out, batch, logits, _ = _batch(positive_first=False)
    result = AicSurrogateLoss()(out, batch)
    assert result.components['dice'].item() == 0
    assert result.components['fpr'].item() > 0
    result.total.backward()
    assert logits.grad.abs().sum() > 0


def test_harmonic_keeps_gradient_on_a_batch_without_positives():
    """The convolution is zero when Dice is zero, which would strand negatives."""
    out, batch, logits, _ = _batch(positive_first=False)
    criterion = AicHarmonicLoss(area_threshold=.01, area_mode='rational')
    result = criterion(out, batch)
    assert set(result.components) == {'bce', 'harmonic', 'cls'}

    main = MaskBatch(out['logits'], batch['mask'], None)
    expected = soft_false_positive(main.predicted_area, threshold=.01, mode='rational').mean()
    torch.testing.assert_close(result.components['harmonic'], expected)

    grad = torch.autograd.grad(result.components['harmonic'], logits, retain_graph=True)[0]
    assert grad.abs().sum() > 0


def test_harmonic_is_one_minus_aic_when_positives_are_present():
    out, batch, logits, _ = _batch()
    criterion = AicHarmonicLoss(area_threshold=.01, area_mode='rational')
    result = criterion(out, batch)

    main = MaskBatch(out['logits'], batch['mask'], None)
    positive = main.is_positive
    dice = main.mean_over(main.dice(1.), positive)
    fpr = main.mean_over(soft_false_positive(main.predicted_area, threshold=.01, mode='rational'),
                         1. - positive).clamp(0., 1.)
    torch.testing.assert_close(result.components['harmonic'], 1. - harmonic_aic(dice, fpr))
    result.total.backward()
    assert torch.isfinite(logits.grad).all()


def test_diagnostics_stay_the_plain_dice_loss_for_comparability():
    """dice_pos/dice_neg must mean the same thing as in a bce_dice run."""
    out, batch, _, _ = _batch()
    reference = SegmentationLoss()(out, batch).diagnostics
    for arm in (AicSurrogateLoss(), AicHarmonicLoss()):
        actual = arm(out, batch).diagnostics
        assert set(actual) == set(reference)
        for key in reference:
            torch.testing.assert_close(actual[key][0], reference[key][0])
            torch.testing.assert_close(actual[key][1].float(), reference[key][1].float())


def test_aux_loss_weight_zero_keeps_modules_but_drops_the_term():
    out, batch, _, cls = _batch()
    aux = torch.zeros_like(batch['mask'], requires_grad=True)
    out['aux_logits'] = aux
    result = AicSurrogateLoss(aux_weight=.4, aux_loss_weight=0)(out, batch)
    assert not any(key.startswith('aux_') for key in result.components)
    result.total.backward()
    assert aux.grad is None and cls.grad.abs().sum() > 0


def test_aux_head_learns_the_same_arm_term():
    out, batch, _, _ = _batch()
    out['aux_logits'] = torch.zeros_like(batch['mask'], requires_grad=True)
    result = AicSurrogateLoss(aux_weight=.4)(out, batch)
    assert {'aux_bce', 'aux_dice', 'aux_fpr'} <= set(result.components)


def test_dct_aux_respects_availability():
    out, batch, _, _ = _batch()
    out['dct_aux_logits'] = torch.zeros(2, 1, 4, 4, requires_grad=True)
    out['dct_aux_available'] = torch.tensor([True, False])
    result = AicSurrogateLoss(dct_aux_weight=.2)(out, batch)
    assert {'dct_aux_bce', 'dct_aux_dice', 'dct_aux_fpr'} <= set(result.components)
    # No frame available means no term at all, not a crash.
    out['dct_aux_available'] = torch.tensor([False, False])
    assert not any(key.startswith('dct_aux_') for key in AicSurrogateLoss(dct_aux_weight=.2)(out, batch).components)


@pytest.mark.parametrize('objective,expected', [('bce_dice', SegmentationLoss),
                                                ('aic_surrogate', AicSurrogateLoss),
                                                ('aic_harmonic', AicHarmonicLoss)])
def test_build_criterion_dispatches_and_passes_only_relevant_fields(objective, expected):
    from dataclasses import replace

    from src.config import LossConfig, load_experiment_config
    cfg = load_experiment_config('configs/baseline.yaml')
    criterion = build_criterion(replace(cfg, loss=LossConfig(objective=objective)),
                                aux_weight=.4, dct_aux_weight=.2)
    assert isinstance(criterion, expected)
    assert criterion.aux_weight == pytest.approx(.4)
    assert criterion.dct_aux_weight == pytest.approx(.2)


def test_build_criterion_omits_deep_supervision_for_validation():
    from dataclasses import replace

    from src.config import LossConfig, load_experiment_config
    cfg = load_experiment_config('configs/baseline.yaml')
    criterion = build_criterion(replace(cfg, loss=LossConfig(objective='aic_harmonic')))
    assert criterion.aux_weight == 0 and criterion.dct_aux_weight == 0


@pytest.mark.parametrize('name', ARMS)
def test_arms_retain_parent_architecture_and_roundtrip(name):
    from src.config import ExperimentConfig, load_experiment_config
    cfg = load_experiment_config(f'configs/{name}.yaml')
    parent = load_experiment_config('configs/jpeg576_pretrained_18ep_full_train_finetune_3ep.yaml')
    assert cfg.model == parent.model
    assert cfg.train == parent.train and cfg.augmentation == parent.augmentation
    assert cfg.dataset == parent.dataset and cfg.eval == parent.eval
    assert cfg.loss.objective == name.rsplit('_finetune_3ep_', 1)[1]
    assert cfg.loss.aux_loss_weight == 0
    assert cfg.loss.area_threshold == .01
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg
    # The arms must train the objective they name.
    assert isinstance(build_criterion(cfg), (AicSurrogateLoss, AicHarmonicLoss))


@pytest.mark.parametrize('values', [
    {'objective': 'aic_nope'},
    {'objective': 'aic_harmonic', 'area_mode': 'linear'},
    {'objective': 'aic_harmonic', 'area_threshold': 0.},
    {'objective': 'aic_harmonic', 'area_threshold': 1.},
    {'objective': 'aic_harmonic', 'area_softness': 0.},
    {'objective': 'aic_surrogate', 'fpr_weight': -1},
    {'objective': 'aic_surrogate', 'aic_weight': float('nan')},
])
def test_invalid_options(values):
    from src.config import LossConfig
    with pytest.raises(ValueError):
        LossConfig(**values)


@pytest.mark.parametrize('values', [
    {'objective': 'aic_harmonic', 'boundary_weight': 1.},
    {'objective': 'aic_harmonic', 'dice_weight': .5},
    {'objective': 'aic_surrogate', 'dice_scope': 'positive'},
    {'objective': 'aic_surrogate', 'pixel_loss': 'focal'},
    {'objective': 'bce_dice', 'fpr_weight': 2.},
    {'objective': 'bce_dice', 'area_mode': 'rational'},
])
def test_a_field_the_objective_ignores_is_rejected_not_dropped(values):
    """A silent no-op here costs a whole training run before anyone notices."""
    from src.config import LossConfig
    with pytest.raises(ValueError, match='ignores'):
        LossConfig(**values)


def test_defaults_leave_historical_runs_on_bce_dice():
    from src.config import LossConfig, load_experiment_config
    assert LossConfig().objective == 'bce_dice'
    assert load_experiment_config('configs/baseline.yaml').loss.objective == 'bce_dice'
    # Every pre-existing arm keeps its objective.
    assert load_experiment_config(
        'configs/jpeg576_pretrained_18ep_full_train_finetune_3ep_focal_boundary.yaml'
    ).loss.objective == 'bce_dice'


def test_old_snapshot_resumes_but_a_changed_objective_is_rejected(tmp_path, monkeypatch):
    from dataclasses import replace

    import src.training.engine as engine
    from src.config import LossConfig, load_experiment_config
    from src.training.engine import ExperimentRunner
    from src.training.runs import Run

    cfg = load_experiment_config('configs/baseline.yaml')
    cfg = replace(cfg, paths=replace(cfg.paths, runs_path=tmp_path),
                  train=replace(cfg.train, device='cpu', amp='off', resume=True))
    snapshot = cfg.to_flat_dict()
    # A snapshot written before the AIC arms existed carries none of their keys.
    for key in LossConfig().to_dict():
        if key not in {'dice_scope', 'dice_weight'}:
            snapshot.pop(key)
    run = Run.create(tmp_path, cfg.paths.run_name, tensorboard=False)
    run.save_snapshot(snapshot)
    (run.dir / 'ckpt/last.pt').touch()

    class Protocol:
        def verify_run(self, saved):
            pass

    monkeypatch.setattr(engine.EvaluationProtocol, 'load', lambda path: Protocol())
    ExperimentRunner(cfg)._check_resume_protocol()
    with pytest.raises(ValueError, match='loss.objective'):
        ExperimentRunner(replace(cfg, loss=LossConfig(objective='aic_harmonic')))._check_resume_protocol()
