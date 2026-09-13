from dataclasses import replace

import pytest
import torch

from src.config import ModelConfig, load_experiment_config
from src.training.builders import build_model


@pytest.mark.parametrize('stride', [4, 8])
def test_early_fusion_reaches_decoder_once_and_has_gradients(stride):
    torch.set_num_threads(1)
    model = build_model(ModelConfig(wavelet_image_size=64, wavelet_fusion=f'stride{stride}'), pretrained=False).train()
    image = torch.randn(2, 3, 64, 64)
    native = [torch.randint(256, (3, 73, 91), dtype=torch.uint8) for _ in range(2)]
    branch_outputs, decoder_inputs = [], []
    h1 = model.wavelet_branch.register_forward_hook(lambda m, args, out: branch_outputs.append(out[0]))
    h2 = model.decoder.register_forward_pre_hook(lambda m, args: decoder_inputs.append(args[0][model.strides.index(stride)]))
    with torch.no_grad():
        model.wavelet_branch.gamma.fill_(.1)
    out = model(image, native_rgb=native)
    h1.remove(); h2.remove()
    assert len(branch_outputs) == 1
    assert branch_outputs[0] is decoder_inputs[0]
    assert branch_outputs[0].shape[-2:] == (64 // stride, 64 // stride)
    assert out['aux_logits'].shape == out['logits'].shape == (2, 1, 64, 64)
    out['logits'].square().mean().backward()
    assert model.wavelet_branch.stem[0].weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())


@pytest.mark.parametrize('stride', [4, 8])
def test_early_recipe_preserves_late_training_and_roundtrips(stride):
    from src.inference.submission import InferenceConfig
    from src.budget import count_gflops
    late = load_experiment_config('configs/jpeg576_fusion_local_weighted_val_wavelet.yaml')
    early = load_experiment_config(f'configs/jpeg576_fusion_local_weighted_val_wavelet_stride{stride}.yaml')
    assert early.model == replace(late.model, wavelet_fusion=f'stride{stride}')
    assert early.train == late.train and early.loss == late.loss
    assert early.dataset == late.dataset and early.augmentation == late.augmentation and early.eval == late.eval
    for snapshot in (early.to_dict(), early.to_flat_dict()):
        assert InferenceConfig.from_snapshot(snapshot).model == early.model
    with torch.device('meta'):
        model = build_model(early.model, pretrained=False)
    assert count_gflops(model, 576, native_size=(1024, 1024)) < 100


@pytest.mark.parametrize('kwargs', [dict(wavelet_fusion='unknown'), dict(wavelet_fusion='stride8'), dict(wavelet_fusion='stride4')])
def test_invalid_fusion_config(kwargs):
    with pytest.raises(ValueError, match='wavelet'):
        ModelConfig(**kwargs)


@pytest.mark.parametrize('stride', [4, 8])
def test_early_jpeg_zero_gate_matches_late_and_checkpoint_reloads(stride):
    cfg = load_experiment_config(f'configs/jpeg576_fusion_local_weighted_val_wavelet_stride{stride}.yaml')
    config = replace(cfg.model, wavelet_image_size=64)
    early = build_model(config, pretrained=False).eval()
    late = build_model(replace(config, wavelet_fusion='late'), pretrained=False).eval()
    common = {k: v for k, v in early.state_dict().items() if not k.startswith('wavelet_branch.')}
    loaded = late.load_state_dict(common, strict=False)
    assert not loaded.unexpected_keys
    assert all(k.startswith('wavelet_branch.') for k in loaded.missing_keys)
    image = torch.randn(1, 3, 64, 64)
    native = [torch.randint(256, (3, 80, 96), dtype=torch.uint8)]
    jpeg = [dict(bins=torch.randint(21, (80, 96), dtype=torch.uint8), qtable=torch.ones(8, 8),
                 geometry=(0, 0, 80, 96, 0, 0, 0))]
    with torch.no_grad():
        a = early(image, native_rgb=native, jpeg=jpeg)
        b = late(image, native_rgb=native, jpeg=jpeg)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        early.wavelet_branch.gamma.fill_(.2)
        expected = early(image, native_rgb=native, jpeg=jpeg)
    restored = build_model(config, pretrained=False).eval()
    restored.load_state_dict(early.state_dict(), strict=True)
    with torch.no_grad():
        actual = restored(image, native_rgb=native, jpeg=jpeg)
        torch.testing.assert_close(expected['logits'], actual['logits'])
        assert not torch.equal(a['logits'], actual['logits'])


@pytest.mark.parametrize('nested', [False, True])
def test_legacy_wavelet_resume_defaults_to_late(tmp_path, monkeypatch, nested):
    from src.training.engine import ExperimentRunner, EvaluationProtocol
    from src.training.runs import Run
    base = load_experiment_config('configs/jpeg576_fusion_local_weighted_val_wavelet.yaml')
    config = replace(base, paths=replace(base.paths, runs_path=tmp_path), train=replace(base.train, resume=True))
    run = Run.create(tmp_path, config.paths.run_name, tensorboard=False)
    snapshot = config.to_dict() if nested else config.to_flat_dict()
    (snapshot['model'] if nested else snapshot).pop('wavelet_fusion')
    run.save_snapshot(snapshot)
    (run.dir / 'ckpt' / 'last.pt').touch()
    class Protocol:
        def verify_run(self, saved):
            pass
    monkeypatch.setattr(EvaluationProtocol, 'load', lambda path: Protocol())
    ExperimentRunner(config)._check_resume_protocol()
    with pytest.raises(ValueError, match='wavelet_fusion'):
        ExperimentRunner(replace(config, model=replace(config.model, wavelet_fusion='stride8')))._check_resume_protocol()
