import math

import numpy  # noqa: F401 - initialize NumPy before Torch on Windows.
import pytest
import torch
import torch.nn.functional as F


def test_selected_formula_and_self_exclusion():
    from src.modules.forensic_contrastive import ForensicIntraContrastiveLoss

    loss = ForensicIntraContrastiveLoss(temperature=1.)
    pos = torch.tensor([[1., 0.], [1., 0.]])
    neg = -pos
    torch.testing.assert_close(loss.selected_loss(pos, neg),
                               torch.tensor(math.log1p(2 * math.exp(-2))))
    torch.testing.assert_close(loss.selected_loss(pos, pos), torch.tensor(math.log(3)))
    pos = F.normalize(torch.tensor([[1., 2.], [-2., 1.], [3., -1.]]), dim=1)
    neg = F.normalize(torch.tensor([[-1., -2.], [1., 3.]]), dim=1)
    expected = []
    for i, query in enumerate(pos):
        positive = sum(float(query.dot(other)) for j, other in enumerate(pos) if j != i) / 2
        denominator = math.exp(positive) + sum(math.exp(float(query.dot(other))) for other in neg)
        expected.append(math.log(denominator) - positive)
    torch.testing.assert_close(loss.selected_loss(pos, neg), torch.tensor(sum(expected) / 3))


@pytest.mark.parametrize('kind', ['empty', 'full', 'single', 'invalid', 'unavailable'])
def test_ineligible_returns_differentiable_zero(kind):
    from src.modules.forensic_contrastive import ForensicIntraContrastiveLoss

    embeddings = torch.randn(1, 4, 2, 3, requires_grad=True)
    mask = torch.zeros(1, 1, 2, 3)
    mask.flatten()[:(6 if kind == 'full' else 1 if kind == 'single' else 2)] = 1
    if kind == 'empty':
        mask.zero_()
    valid = torch.zeros_like(mask) if kind == 'invalid' else None
    result = ForensicIntraContrastiveLoss()(embeddings, mask, torch.tensor([kind != 'unavailable']), valid)
    assert result.loss.item() == 0
    result.loss.backward()
    assert torch.isfinite(embeddings.grad).all()
    assert result.diagnostics['forensic_intra_raw'][1].item() == 0


def test_mask_cells_exclude_mixed_and_padding():
    from src.modules.forensic_contrastive import ForensicIntraContrastiveLoss

    mask = torch.tensor([[[[1., 1., 0., 0.], [1., 1., 0., 0.],
                            [1., 0., 0., 0.], [1., 0., 0., 0.]]]])
    valid = torch.ones_like(mask)
    valid[0, 0, 3, 3] = 0
    pos, neg = ForensicIntraContrastiveLoss().mask_cells(mask, valid, (2, 2))
    assert pos.flatten().tolist() == [True, False, False, False]
    assert neg.flatten().tolist() == [False, True, False, False]


def test_sampling_is_bounded_unique_and_rng_replayable():
    from src.modules.forensic_contrastive import ForensicIntraContrastiveLoss

    loss = ForensicIntraContrastiveLoss(max_samples=7)
    indices = torch.arange(100)
    state = torch.get_rng_state()
    selected = loss.sample(indices)
    torch.set_rng_state(state)
    assert torch.equal(selected, loss.sample(indices))
    assert selected.unique().numel() == 7
    assert torch.equal(loss.sample(indices[:2]), indices[:2])


def test_loss_is_image_mean_and_minimal_counts_work():
    from src.modules.forensic_contrastive import ForensicIntraContrastiveLoss

    loss = ForensicIntraContrastiveLoss()
    embeddings = torch.randn(2, 5, 1, 5, requires_grad=True)
    mask = torch.tensor([[[[1., 1., 0., .5, .5]]], [[[1., 1., 1., 0., 0.]]]])
    result = loss(embeddings, mask, torch.ones(2, dtype=torch.bool))
    individual = [loss(embeddings[i:i+1], mask[i:i+1], torch.tensor([True])).loss for i in range(2)]
    torch.testing.assert_close(result.loss, sum(individual) / 2)
    result.loss.backward()
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0


def test_config_defaults_and_joint_validation():
    from src.config import ExperimentConfig, LossConfig, ModelConfig

    assert ModelConfig().forensic_contrastive_dim == 0
    assert LossConfig().forensic_intra_weight == 0
    data = {'paths': {'run_name': 'test'}, 'model': {'forensic_contrastive_dim': 32}}
    with pytest.raises(ValueError, match='together'):
        ExperimentConfig.from_dict(data)
    data['loss'] = {'forensic_intra_weight': .01}
    config = ExperimentConfig.from_dict(data)
    assert ExperimentConfig.from_dict(config.to_dict()) == config


def test_weighted_loss_preserves_existing_components_and_eval_contract():
    from src.losses import SegmentationLoss

    out = {'logits': torch.randn(1, 1, 4, 4, requires_grad=True),
           'cls_logits': torch.randn(1, 1), 'forensic_embeddings': torch.randn(1, 4, 4, 4, requires_grad=True),
           'forensic_available': torch.tensor([True])}
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, :2] = 1
    batch = {'mask': mask}
    base = SegmentationLoss()(out, batch)
    criterion = SegmentationLoss(forensic_intra_weight=.01)
    active = criterion(out, batch)
    for key, value in base.components.items():
        torch.testing.assert_close(active.components[key], value, rtol=0, atol=0)
    torch.testing.assert_close(active.total, base.total + active.components['forensic_intra'])
    missing = {key: out[key] for key in ('logits', 'cls_logits')}
    with pytest.raises(ValueError, match='forensic'):
        criterion(missing, batch)
    torch.testing.assert_close(criterion.eval()(missing, batch).total, base.total)


def make_fusion(dim=8, mode='maps'):
    from src.modules.forensic_fusion import ForensicFusion
    return ForensicFusion([8, 16, 32], [4, 8, 16], (4, 8, 16),
                          forensic_mode=mode, forensic_contrastive_dim=dim)


def test_fusion_gradient_is_only_forensic_and_legacy_api_is_preserved():
    from src.modules.forensic_contrastive import ForensicIntraContrastiveLoss

    fusion = make_fusion()
    rgb = [torch.randn(2, c, s, s, requires_grad=True) for c, s in [(4, 8), (8, 4), (16, 2)]]
    fmap = torch.randn(2, 12, 6, 6)
    result = fusion.forward_training(list(rgb), fmap)
    assert result.embeddings.shape == (2, 8, 8, 8)
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, :32] = 1
    cl = ForensicIntraContrastiveLoss()(result.embeddings, mask, result.available)
    cl.loss.backward()
    for module in (fusion.branch, fusion.contrastive_head):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    assert all(p.grad is None for p in fusion.fusion_blocks.parameters())
    assert all(x.grad is None for x in rgb)
    assert len(fusion([x.detach() for x in rgb], fmap, return_aux=True)) == 2


def test_shared_initialization_eval_and_checkpoint_optimizer_ema():
    from src.config import TrainConfig
    from src.training.builders import build_ema, build_optimizer

    torch.manual_seed(23)
    baseline = make_fusion(0)
    next_baseline = torch.nn.Linear(4, 2)
    torch.manual_seed(23)
    active = make_fusion(8)
    next_active = torch.nn.Linear(4, 2)
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(active.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(next_active.weight, next_baseline.weight, rtol=0, atol=0)
    assert not any('contrastive_head' in key for key in baseline.state_dict())
    make_fusion(0).load_state_dict(baseline.state_dict(), strict=True)
    restored = make_fusion(8)
    restored.load_state_dict(active.state_dict(), strict=True)
    wrapper = torch.nn.Module()
    wrapper.forensic_fusion = active
    cfg = TrainConfig()
    optimizer = build_optimizer(cfg, wrapper)
    for parameter in active.contrastive_head.parameters():
        groups = [g for g in optimizer.param_groups if any(p is parameter for p in g['params'])]
        assert len(groups) == 1 and groups[0]['lr'] == cfg.fmap_lr
    ema = build_ema(cfg, wrapper)
    ema.update_parameters(wrapper)
    for actual, expected in zip(ema.module.forensic_fusion.contrastive_head.parameters(),
                                active.contrastive_head.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    calls = []
    active.contrastive_head.register_forward_hook(lambda *args: calls.append(1))
    rgb = [torch.randn(2, c, s, s) for c, s in [(4, 8), (8, 4), (16, 2)]]
    fmap = torch.randn(2, 12, 8, 8)
    active.eval()
    baseline.eval()
    state = torch.get_rng_state()
    with torch.no_grad():
        outputs = active(list(rgb), fmap)
        expected = baseline(list(rgb), fmap)
    assert torch.equal(state, torch.get_rng_state())
    assert not calls
    for value, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_jpeg_mixed_scatter_and_all_unavailable():
    fusion = make_fusion(mode='jpeg')

    class AlignedBranch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, samples, sizes):
            self.calls.append(samples)
            return {s: torch.stack([torch.full((c, *sizes[s]), float(x['id'])) for x in samples])
                    for s, c in [(8, 4), (16, 8), (32, 16)]}

    fusion.branch = AlignedBranch()
    rgb = [torch.randn(3, c, s, s) for c, s in [(4, 8), (8, 4), (16, 2)]]
    jpeg = [{'id': 1}, {'available': False}, {'id': 3}]
    result = fusion.forward_training(list(rgb), None, jpeg=jpeg)
    assert result.available.tolist() == [True, False, True]
    assert len(fusion.branch.calls) == 1 and len(fusion.branch.calls[0]) == 2
    expected = fusion.contrastive_head(torch.stack([torch.ones(4, 8, 8), torch.full((4, 8, 8), 3.)]))
    torch.testing.assert_close(result.embeddings[[0, 2]], expected)
    assert result.embeddings[1].count_nonzero() == 0
    for value, original in zip(result.features, rgb, strict=True):
        torch.testing.assert_close(value[1], original[1], rtol=0, atol=0)
    result = fusion.forward_training(list(rgb), None, jpeg=[{'available': False}] * 3)
    assert len(fusion.branch.calls) == 1
    assert not result.available.any()
    result.embeddings.sum().backward()


@pytest.mark.parametrize('key,value', [
    ('forensic_intra_weight', float('nan')), ('forensic_intra_weight', -.01),
    ('forensic_intra_temperature', 0), ('forensic_intra_temperature', float('inf')),
    ('forensic_intra_max_samples', True), ('forensic_intra_max_samples', 1),
    ('forensic_intra_max_samples', 2.5), ('forensic_intra_positive_fraction', 0),
    ('forensic_intra_positive_fraction', float('nan')), ('forensic_intra_positive_fraction', 1.1),
])
def test_invalid_loss_configuration(key, value):
    from src.config import LossConfig
    with pytest.raises(ValueError):
        LossConfig(**{key: value})


@pytest.mark.parametrize('value', [True, -1, 1.5])
def test_invalid_dimension(value):
    from src.config import ModelConfig
    with pytest.raises(ValueError):
        ModelConfig(forensic_contrastive_dim=value)


def test_experiment_changes_only_selected_parameters():
    from src.config import load_experiment_config
    parent = load_experiment_config('configs/jpeg576_fusion_local_weighted_val.yaml').to_dict()
    child = load_experiment_config('configs/jpeg576_fusion_local_weighted_val_intra_contrastive.yaml').to_dict()
    assert child['paths'].pop('run_name') != parent['paths'].pop('run_name')
    assert child['model'].pop('forensic_contrastive_dim') == 32
    assert parent['model'].pop('forensic_contrastive_dim') == 0
    assert child['loss'].pop('forensic_intra_weight') == .01
    assert parent['loss'].pop('forensic_intra_weight') == 0
    assert child == parent


@pytest.mark.parametrize('device,dtype', [('cpu', torch.bfloat16), ('cuda', torch.float16), ('cuda', torch.bfloat16)])
def test_autocast_fp32_and_finite_gradients(device, dtype):
    from src.modules.forensic_contrastive import ForensicIntraContrastiveLoss, ForensicProjectionHead
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if device == 'cuda' and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 unavailable')
    head = ForensicProjectionHead(8, 4).to(device)
    inputs = torch.randn(2, 8, 8, 8, device=device, requires_grad=True)
    mask = torch.zeros(2, 1, 8, 8, device=device)
    mask[:, :, :4] = 1
    criterion = ForensicIntraContrastiveLoss()
    available = torch.ones(2, dtype=torch.bool, device=device)
    with torch.autocast(device_type=device, dtype=dtype):
        embeddings = head(inputs)
        actual = criterion(embeddings, mask, available).loss
    expected = criterion(embeddings.float(), mask, available).loss
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert torch.isfinite(inputs.grad).all() and inputs.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in head.parameters())


def test_epoch_diagnostics_use_conditional_sums_and_report_counts():
    from src.losses import LossMeter, LossResult
    from src.modules.forensic_contrastive import ForensicIntraContrastiveLoss
    meter = LossMeter()
    raw_sums, eligible_counts = [], []
    for batch_size, n_eligible in [(3, 1), (2, 2), (1, 0)]:
        embeddings = torch.randn(batch_size, 4, 1, 3)
        mask = torch.zeros(batch_size, 1, 1, 3)
        mask[:n_eligible, :, :, :2] = 1
        result = ForensicIntraContrastiveLoss()(embeddings, mask, torch.ones(batch_size, dtype=torch.bool))
        meter.update(LossResult(result.loss, {'forensic_intra': .01 * result.loss}, result.diagnostics), batch_size)
        raw_sum, count = result.diagnostics['forensic_intra_raw']
        raw_sums.append(raw_sum)
        eligible_counts.append(count)
    metrics = meter.compute()
    assert metrics['forensic_intra_raw'] == pytest.approx(float(sum(raw_sums) / sum(eligible_counts)))
    assert metrics['forensic_intra_eligible_count'] == 3
    assert metrics['forensic_intra_total_count'] == 6


def test_torch_rng_state_roundtrip():
    from src.training.base import TorchRNGState
    state = TorchRNGState.capture()
    expected = torch.randperm(100)
    expected_cuda = torch.randperm(100, device='cuda') if torch.cuda.is_available() else None
    TorchRNGState.restore(state)
    assert torch.equal(torch.randperm(100), expected)
    if expected_cuda is not None:
        assert torch.equal(torch.randperm(100, device='cuda'), expected_cuda)


def test_segmenter_outputs_eval_equivalence_and_cl_gradient_isolation():
    from src.config import ModelConfig
    from src.modules.forensic_contrastive import ForensicIntraContrastiveLoss
    from src.training.builders import build_model

    def build(dim):
        return build_model(ModelConfig(encoder_name='pvt_v2_b0', forensic_channels=(4, 8, 16),
                                       aux_weight=0., forensic_contrastive_dim=dim), pretrained=False)

    torch.manual_seed(31)
    baseline = build(0)
    torch.manual_seed(31)
    active = build(8)
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(active.state_dict()[key], value, rtol=0, atol=0)
    images, fmap = torch.randn(2, 3, 64, 64), torch.randn(2, 12, 8, 8)
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, :32] = 1
    result = active(images, fmap)
    assert result['forensic_embeddings'].shape == (2, 8, 8, 8)
    ForensicIntraContrastiveLoss()(result['forensic_embeddings'], mask, result['forensic_available']).loss.backward()
    assert all(parameter.grad is None for name, parameter in active.named_parameters()
               if not name.startswith('forensic_fusion.branch.') and not name.startswith('forensic_fusion.contrastive_head.'))
    baseline.load_state_dict({key: active.state_dict()[key] for key in baseline.state_dict()}, strict=True)
    calls = []
    active.forensic_fusion.contrastive_head.register_forward_hook(lambda *args: calls.append(1))
    baseline.eval()
    active.eval()
    with torch.no_grad():
        actual, expected = active(images, fmap), baseline(images, fmap)
    assert set(actual) == set(expected) == {'logits', 'cls_logits'}
    assert not calls
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_training_checkpoint_restores_head_optimizer_ema_and_sampling(tmp_path):
    from dataclasses import replace

    from src.config import load_experiment_config
    from src.training.builders import build_amp, build_ema, build_optimizer, build_scheduler
    from src.training.engine import EpochTrainResult, ExperimentRunner, TrainingState
    from src.training.runs import Run

    cfg = load_experiment_config('configs/jpeg576_fusion_local_weighted_val_intra_contrastive.yaml')
    cfg = replace(cfg, train=replace(cfg.train, device='cpu', amp='off', resume=True))
    model = torch.nn.Module()
    model.forensic_fusion = make_fusion(8)
    optimizer = build_optimizer(cfg.train, model)
    scheduler = build_scheduler(cfg.train, optimizer, 1)
    scaler = build_amp(cfg.train).scaler()
    loss = sum(p.square().sum() for p in model.forensic_fusion.contrastive_head.parameters())
    loss.backward()
    optimizer.step()
    scheduler.step()
    ema = build_ema(cfg.train, model)
    ema.update_parameters(model)
    run = Run(tmp_path)
    (run.dir / 'ckpt').mkdir()
    state = TrainingState(pending_train_result=EpochTrainResult(loss=float(loss.detach()), skipped_steps=0, seen=2))
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    ExperimentRunner._save_training_checkpoint(run, model, ema, optimizer, scheduler, scaler,
                                               0, state, cfg.to_flat_dict(), validation_complete=True)
    expected_sampling = torch.randperm(100)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    restored = ExperimentRunner(cfg)._resume_if_needed(
        run=run, model=model, ema=ema, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
    assert restored.start_epoch == 1
    assert torch.equal(torch.randperm(100), expected_sampling)
    for key in expected:
        torch.testing.assert_close(model.state_dict()[key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(ema.module.state_dict()[key], expected[key], rtol=0, atol=0)
    assert optimizer.state


def test_active_resume_rejects_changed_settings_and_baseline_accepts_old_fields(tmp_path, monkeypatch):
    from dataclasses import replace

    from src.config import load_experiment_config
    from src.eval.protocol import EvaluationProtocol
    from src.training.engine import ExperimentRunner
    from src.training.runs import Run

    cfg = load_experiment_config('configs/jpeg576_fusion_local_weighted_val_intra_contrastive.yaml')
    cfg = replace(cfg, paths=replace(cfg.paths, runs_path=tmp_path), train=replace(cfg.train, resume=True))
    run = Run(tmp_path / cfg.paths.run_name)
    (run.dir / 'ckpt').mkdir(parents=True)
    (run.dir / 'ckpt' / 'last.pt').touch()
    run.save_snapshot(cfg.to_flat_dict())
    monkeypatch.setattr(EvaluationProtocol, 'load', lambda _: type('Protocol', (), {'verify_run': lambda self, snapshot: None})())
    ExperimentRunner(cfg)._check_resume_protocol()
    changed = replace(cfg, loss=replace(cfg.loss, forensic_intra_temperature=.2))
    with pytest.raises(ValueError, match='forensic_intra_temperature'):
        ExperimentRunner(changed)._check_resume_protocol()
    baseline = replace(cfg, model=replace(cfg.model, forensic_contrastive_dim=0),
                       loss=replace(cfg.loss, forensic_intra_weight=0))
    old_snapshot = {key: value for key, value in baseline.to_flat_dict().items()
                    if not key.startswith(('forensic_intra_', 'forensic_contrastive_'))}
    run.save_snapshot(old_snapshot)
    ExperimentRunner(baseline)._check_resume_protocol()


def test_loader_recreation_does_not_shift_contrastive_rng():
    from torch.utils.data.dataloader import _BaseDataLoaderIter

    from src.config import load_experiment_config
    from src.training.base import TorchRNGState
    from src.training.engine import ExperimentRunner

    runner = ExperimentRunner(load_experiment_config(
        'configs/jpeg576_fusion_local_weighted_val_intra_contrastive.yaml'))

    def loaders():
        pair = [torch.utils.data.DataLoader(torch.arange(4), num_workers=1, persistent_workers=True)
                for _ in range(2)]
        runner._isolate_loader_rng(*pair)
        return pair

    original = loaders()
    # Exercise real iterator RNG behavior without spawning worker processes.
    iterators = [_BaseDataLoaderIter(loader) for loader in original]
    state = TorchRNGState.capture()
    for iterator, loader in zip(iterators, original, strict=True):
        iterator._reset(loader)
    expected = torch.randperm(100)
    TorchRNGState.restore(state)
    recreated = loaders()
    for loader in recreated:
        _BaseDataLoaderIter(loader)
    assert torch.equal(torch.randperm(100), expected)
