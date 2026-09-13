from dataclasses import replace

import torch
import pytest

from src.config import load_experiment_config
from src.training.builders import build_model
from src.inference.submission import InferenceConfig
from src.config import ModelConfig


def test_multi_recipe_and_shared_extraction_gradients():
    torch.set_num_threads(1)
    base = load_experiment_config('configs/jpeg576_fusion_local_weighted_val_wavelet.yaml')
    cfg = load_experiment_config('configs/jpeg576_fusion_local_weighted_val_wavelet_multi.yaml')
    assert cfg.model == replace(base.model, wavelet_fusion='stride4_stride8_late')
    assert cfg.train == base.train and cfg.loss == base.loss
    assert cfg.dataset == base.dataset and cfg.augmentation == base.augmentation and cfg.eval == base.eval
    for snapshot in (cfg.to_dict(), cfg.to_flat_dict()):
        assert InferenceConfig.from_snapshot(snapshot).model == cfg.model
    model = build_model(replace(cfg.model, wavelet_image_size=64, aux_weight=.1), pretrained=False).train()
    branch = model.wavelet_branch
    events, early_outputs = [], {}
    handles = [branch.preprocess.register_forward_hook(lambda *args: events.append('haar')),
               branch.stem.register_forward_hook(lambda *args: events.append('stem')),
               branch.aux_head.register_forward_hook(lambda *args: events.append('aux')),
               model.decoder.register_forward_hook(lambda *args: events.append('decoder')),
               branch.fusion.register_forward_hook(lambda *args: events.append('late'))]
    for name, fusion in branch.early_fusions.items():
        def capture(module, args, output, name=name):
            early_outputs[name] = output
            events.append(name)
        handles.append(fusion.register_forward_hook(capture))
    def check_inputs(module, args):
        for stride in (4, 8):
            assert args[0][model.strides.index(stride)] is early_outputs[f'stride{stride}']
    handles.append(model.decoder.register_forward_pre_hook(check_inputs))
    with torch.no_grad():
        branch.gamma.fill_(.2)
        for fusion in branch.early_fusions.values():
            fusion.gamma.fill_(.2)
    image = torch.randn(2, 3, 64, 64)
    native = [torch.randint(256, (3, 80, 96), dtype=torch.uint8) for _ in range(2)]
    jpeg = [dict(bins=torch.randint(21, (80, 96), dtype=torch.uint8), qtable=torch.ones(8, 8),
                 geometry=(0, 0, 80, 96, 0, 0, 0)) for _ in range(2)]
    result = model(image, native_rgb=native, jpeg=jpeg)
    assert events == ['haar', 'stem', 'aux', 'stride4', 'stride8', 'decoder', 'late']
    assert result['aux_logits'].shape == result['logits'].shape == (2, 1, 64, 64)
    result['logits'].square().mean().backward()
    assert branch.stem[0].weight.grad.abs().sum() > 0
    for fusion in (branch, *branch.early_fusions.values()):
        assert fusion.gamma.grad.abs() > 0
        assert fusion.fusion[-1].weight.grad.abs().sum() > 0
    for handle in handles:
        handle.remove()
    model.eval()
    restored = build_model(replace(cfg.model, wavelet_image_size=64, aux_weight=.1), pretrained=False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        expected = model(image, native_rgb=native, jpeg=jpeg)
        actual = restored(image, native_rgb=native, jpeg=jpeg)
    torch.testing.assert_close(expected['logits'], actual['logits'])
    assert 'aux_logits' not in actual


def test_multi_zero_gates_preserve_late_outputs_and_budget():
    from src.budget import count_gflops
    cfg = load_experiment_config('configs/jpeg576_fusion_local_weighted_val_wavelet_multi.yaml')
    with torch.device('meta'):
        meta_model = build_model(cfg.model, pretrained=False)
    assert count_gflops(meta_model, 576, native_size=(1024, 1024)) < 100
    config = replace(cfg.model, wavelet_image_size=64)
    multi = build_model(config, pretrained=False).eval()
    late = build_model(replace(config, wavelet_fusion='late'), pretrained=False).eval()
    loaded = multi.load_state_dict(late.state_dict(), strict=False)
    assert not loaded.unexpected_keys
    assert all(key.startswith('wavelet_branch.early_fusions.') for key in loaded.missing_keys)
    assert len({id(f.gamma) for f in (multi.wavelet_branch, *multi.wavelet_branch.early_fusions.values())}) == 3
    image = torch.randn(1, 3, 64, 64)
    native = [torch.randint(256, (3, 79, 91), dtype=torch.uint8)]
    jpeg = [dict(bins=torch.randint(21, (80, 96), dtype=torch.uint8), qtable=torch.ones(8, 8),
                 geometry=(0, 0, 79, 91, 0, 0, 0))]
    with torch.no_grad():
        actual = multi(image, native_rgb=native, jpeg=jpeg)
        expected = late(image, native_rgb=native, jpeg=jpeg)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_multi_requires_native_wavelet_branch():
    with pytest.raises(ValueError, match='requires wavelet_image_size'):
        ModelConfig(wavelet_fusion='stride4_stride8_late')
