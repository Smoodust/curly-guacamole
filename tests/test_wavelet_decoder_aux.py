from dataclasses import replace

import pytest
import torch

from src.config import ModelConfig
from src.training.builders import build_model


def test_decoder_aux_wavelet_preserves_initialization_and_gradient_paths():
    config = ModelConfig(use_forensics=False, aux_weight=.4)
    torch.manual_seed(42)
    baseline = build_model(config, pretrained=False)
    torch.manual_seed(42)
    model = build_model(replace(config, wavelet_image_size=64,
                                wavelet_aux_source='decoder'), pretrained=False)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    assert model.decoder.aux_head is not None
    assert model.wavelet_branch.aux_head is None
    x = torch.randn(2, 3, 64, 64)
    native = [torch.randint(0, 256, (3, 65, 77), dtype=torch.uint8) for _ in range(2)]
    baseline.eval()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x, native_rgb=native)['logits'], baseline(x)['logits'])
    model.train()
    out = model(x, native_rgb=native)
    out['aux_logits'].square().mean().backward()
    assert model.decoder.aux_head.weight.grad.abs().sum() > 0
    assert model.wavelet_branch.stem[0].weight.grad is None
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.wavelet_branch.gamma.fill_(.1)
    model(x, native_rgb=native)['logits'].square().mean().backward()
    assert model.wavelet_branch.stem[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('kwargs', [dict(wavelet_aux_source='unknown'),
                                  dict(wavelet_aux_source='decoder'),
                                  dict(wavelet_aux_source='decoder', wavelet_image_size=64,
                                       wavelet_fusion='stride8')])
def test_invalid_aux_source(kwargs):
    with pytest.raises(ValueError, match='wavelet_aux_source'):
        ModelConfig(**kwargs)


def test_experiment_recipes_and_snapshot_roundtrip():
    from src.config import ExperimentConfig, load_experiment_config
    from src.budget import count_gflops

    base = load_experiment_config('configs/jpeg640_fusion_local_weighted_val_syncbn.yaml')
    wavelet = load_experiment_config('configs/jpeg640_fusion_local_weighted_val_syncbn_wavelet_decoder_aux.yaml')
    long = load_experiment_config('configs/jpeg640_fusion_local_weighted_val_syncbn_18ep_full_train.yaml')
    assert wavelet.model == replace(base.model, wavelet_image_size=1024, wavelet_aux_source='decoder')
    assert wavelet.train == base.train
    assert long.model == base.model
    assert long.train == replace(base.train, epochs=18, epoch_size=24000, full_train_epochs=3)
    assert long.augmentation == replace(base.augmentation, final_full_frame_epochs=3)
    assert wavelet.eval == long.eval == base.eval
    assert wavelet.loss == long.loss == base.loss
    restored = ExperimentConfig.from_dict(wavelet.to_dict())
    assert restored.model == wavelet.model
    with torch.device('meta'):
        model = build_model(restored.model, pretrained=False)
        reference = build_model(wavelet.model, pretrained=False)
    model.load_state_dict(reference.state_dict(), strict=True)
    assert isinstance(model.decoder.aux_head, torch.nn.Conv2d)
    assert any(isinstance(m, torch.nn.SyncBatchNorm) for m in model.decoder.modules())
    assert model.wavelet_branch.aux_head is None
    assert count_gflops(model, 640, native_size=(1080, 1920)) < 100


@pytest.mark.parametrize('nested', [False, True])
def test_legacy_resume_and_changed_aux_guard(tmp_path, monkeypatch, nested):
    from src.config import load_experiment_config
    from src.training.engine import ExperimentRunner, EvaluationProtocol
    from src.training.runs import Run

    base = load_experiment_config('configs/jpeg640_fusion_local_weighted_val_syncbn_wavelet.yaml')
    config = replace(base, paths=replace(base.paths, runs_path=tmp_path))
    run = Run.create(tmp_path, config.paths.run_name, tensorboard=False)
    snapshot = config.to_dict() if nested else config.to_flat_dict()
    (snapshot['model'] if nested else snapshot).pop('wavelet_aux_source')
    run.save_snapshot(snapshot)
    (run.dir / 'ckpt' / 'last.pt').touch()

    class Protocol:
        def verify_run(self, saved):
            pass

    monkeypatch.setattr(EvaluationProtocol, 'load', lambda path: Protocol())
    ExperimentRunner(config)._check_resume_protocol()
    with pytest.raises(ValueError, match='wavelet_aux_source'):
        ExperimentRunner(replace(config, model=replace(config.model, wavelet_aux_source='decoder')))._check_resume_protocol()
