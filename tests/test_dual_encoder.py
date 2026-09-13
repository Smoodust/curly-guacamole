from dataclasses import replace

import pytest
import torch


def test_guided_noise_constant_and_native_texture():
    from src.modules.dual_encoder import GuidedNoise
    prep = GuidedNoise()
    constant = torch.full((3, 35, 47), 127, dtype=torch.uint8)
    assert prep([constant], (32, 32)).abs().max() < 1e-5
    texture = torch.tensor([0, 255], dtype=torch.uint8).repeat(32).expand(3, 64, 64)
    result = prep([texture], (32, 32))
    assert result.shape == (1, 3, 32, 32)
    assert torch.isfinite(result).all() and result.mean() > .01


def test_dynamic_convolution_matches_explicit_mixed_kernel():
    from src.modules.dual_encoder import DynamicConvolution
    layer = DynamicConvolution(4, experts=3)
    x = torch.randn(2, 4, 7, 9)
    weights = layer.routing(x.mean((-2, -1))).softmax(-1)
    expected = []
    for i in range(2):
        kernel = sum(weights[i, j] * expert.weight for j, expert in enumerate(layer.experts))
        expected.append(torch.nn.functional.conv2d(x[i:i+1], kernel, padding=1))
    torch.testing.assert_close(layer(x), torch.cat(expected), atol=1e-6, rtol=1e-5)


def test_dual_encoder_gradients_and_scales():
    from src.modules.dual_encoder import DualEncoder
    torch.set_num_threads(1)
    model = DualEncoder('pvt_v2_b0', 'pvt_v2_b0', pretrained=False)
    rgb = torch.randn(2, 3, 64, 64)
    native = [torch.randint(256, (3, 65, 77), dtype=torch.uint8) for _ in range(2)]
    outputs = model(rgb, native)
    assert [tuple(y.shape) for y in outputs] == [
        (2, c, 64 // s, 64 // s) for c, s in zip(model.channels, model.strides)]
    sum(y.square().mean() for y in outputs).backward()
    for branch in (model.rgb, model.noise, model.fusions):
        grads = [p.grad for p in branch.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0


def test_dual_config_roundtrip_optimizer_and_model():
    from src.config import load_experiment_config, ExperimentConfig
    from src.training.builders import build_model, build_optimizer
    cfg = load_experiment_config('configs/jpeg576_wavelet_dual_b0.yaml')
    cfg = replace(cfg, model=replace(cfg.model, wavelet_image_size=64))
    assert ExperimentConfig.from_dict(cfg.to_dict()).model == cfg.model
    model = build_model(cfg.model, pretrained=False).eval()
    optimizer = build_optimizer(cfg.train, model)
    rates = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
    assert len(rates) == len(list(model.parameters()))
    assert rates[id(next(model.encoder.noise.parameters()))] == cfg.train.encoder_lr
    assert rates[id(next(model.encoder.fusions.parameters()))] == cfg.train.lr
    native = [torch.randint(256, (3, 65, 77), dtype=torch.uint8)]
    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64), native_rgb=native, jpeg=[{'available': False}])
    assert output['logits'].shape == (1, 1, 64, 64)
    assert torch.isfinite(output['logits']).all()


def test_dual_requires_native_wavelet_route():
    from src.config import ModelConfig
    with pytest.raises(ValueError, match='wavelet'):
        ModelConfig(noise_encoder_name='pvt_v2_b0')


@pytest.mark.parametrize('nested', [False, True])
def test_legacy_resume_and_dual_mismatch(tmp_path, monkeypatch, nested):
    from src.config import load_experiment_config
    from src.training.engine import ExperimentRunner, EvaluationProtocol
    from src.training.runs import Run
    cfg = load_experiment_config('configs/jpeg576_fusion_local_weighted_val_wavelet.yaml')
    cfg = replace(cfg, paths=replace(cfg.paths, runs_path=tmp_path))
    run = Run.create(tmp_path, cfg.paths.run_name, tensorboard=False)
    snapshot = cfg.to_dict() if nested else cfg.to_flat_dict()
    section = snapshot['model'] if nested else snapshot
    for key in ('noise_encoder_name', 'guided_radius', 'guided_epsilon', 'guided_scale', 'dual_fusion_width'):
        section.pop(key)
    run.save_snapshot(snapshot)
    (run.dir / 'ckpt' / 'last.pt').touch()
    class Protocol:
        def verify_run(self, saved):
            pass
    monkeypatch.setattr(EvaluationProtocol, 'load', lambda path: Protocol())
    ExperimentRunner(cfg)._check_resume_protocol()
    with pytest.raises(ValueError, match='noise_encoder_name'):
        ExperimentRunner(replace(cfg, model=replace(cfg.model, noise_encoder_name='pvt_v2_b0')))._check_resume_protocol()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_dual_cuda_amp_training_step_and_reload():
    from src.config import load_experiment_config
    from src.losses import SegmentationLoss
    from src.training.builders import build_model, build_optimizer, configure_memory_format, build_ema
    cfg = load_experiment_config('configs/jpeg576_wavelet_dual_b0.yaml')
    model_cfg = replace(cfg.model, wavelet_image_size=64)
    model = configure_memory_format(build_model(model_cfg, pretrained=False).cuda()).train()
    optimizer = build_optimizer(cfg.train, model)
    ema = build_ema(cfg.train, model)
    rgb = torch.randn(2, 3, 64, 64, device='cuda')
    native = [torch.randint(256, (3, 65, 77), dtype=torch.uint8, device='cuda') for _ in range(2)]
    jpeg = [dict(bins=torch.randint(21, (72, 80), dtype=torch.uint8, device='cuda'),
                 qtable=torch.ones(8, 8, device='cuda'), geometry=(0, 0, 65, 77, 0, 0, 0)) for _ in range(2)]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result = model(rgb, native_rgb=native, jpeg=jpeg)
        target = torch.zeros(2, 1, 64, 64, device='cuda')
        target[0, :, 10:30, 20:40] = 1
        loss = SegmentationLoss(aux_weight=0)(result, {'mask': target}).total
    loss.backward()
    for branch in (model.encoder.rgb, model.encoder.noise, model.encoder.fusions):
        grads = [p.grad for p in branch.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    optimizer.step()
    ema.update_parameters(model)
    restored = build_model(model_cfg, pretrained=False).cuda().eval()
    restored.load_state_dict(ema.module.state_dict(), strict=True)
    ema.module.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = ema.module(rgb, native_rgb=native, jpeg=jpeg)['logits']
        actual = restored(rgb, native_rgb=native, jpeg=jpeg)['logits']
    torch.testing.assert_close(actual, expected, rtol=.03, atol=.03)
