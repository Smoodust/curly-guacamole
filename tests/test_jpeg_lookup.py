import numpy as np  # noqa: F401 - initialize the conda runtime before torch
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode

from src.modules.jpeg_branch import JPEGArtifactModule


@pytest.fixture(autouse=True)
def ieee_convolution_reference():
    # A TF32 convolution rounds FP32 weights; lookup retains their full precision.
    with torch.backends.cudnn.flags(allow_tf32=False):
        yield


@pytest.mark.parametrize('shape', [(1, 8, 8), (2, 24, 40)])
@pytest.mark.parametrize('constant', [None, 0, 20])
def test_lookup_matches_dense_outputs_and_parameter_gradients(shape, constant):
    torch.manual_seed(19)
    layer = JPEGArtifactModule().dc_layer0_dil[0].double()
    reference = nn.Conv2d(21, 64, 3, dilation=8, padding=8).double()
    reference.load_state_dict(layer.state_dict(), strict=True)
    bins = torch.randint(0, 21, shape, dtype=torch.uint8)
    if constant is not None:
        bins.fill_(constant)
    volume = F.one_hot(bins.long(), 21).permute(0, 3, 1, 2).double()
    expected = reference(volume)
    actual = layer(bins)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    grad = torch.randn_like(expected)
    actual.backward(grad)
    expected.backward(grad)
    for name in ['weight', 'bias']:
        torch.testing.assert_close(getattr(layer, name).grad, getattr(reference, name).grad,
                                   rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_lookup_matches_dense_and_backpropagates_with_amp(device, dtype):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(23)
    layer = JPEGArtifactModule().dc_layer0_dil[0].to(device)
    reference = nn.Conv2d(21, 64, 3, dilation=8, padding=8).to(device)
    reference.load_state_dict(layer.state_dict())
    bins = torch.randint(0, 21, (2, 24, 40), device=device, dtype=torch.uint8)
    volume = F.one_hot(bins.long(), 21).permute(0, 3, 1, 2).float()
    with torch.autocast(device, dtype=dtype, enabled=dtype != torch.float32):
        actual, expected = layer(bins), reference(volume)
    assert actual.dtype == expected.dtype == dtype
    tolerance = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=0.01, atol=0.002)
    torch.testing.assert_close(actual, expected, **tolerance)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for name in ['weight', 'bias']:
        actual_grad, expected_grad = getattr(layer, name).grad, getattr(reference, name).grad
        assert torch.isfinite(actual_grad).all() and actual_grad.abs().sum() > 0
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0.02, atol=0.5)
        assert (actual_grad - expected_grad).norm() / expected_grad.norm() < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('shape', [(2, 8, 8), (1, 17, 31), (1, 40, 24)])
@pytest.mark.parametrize('category', [0, 20])
def test_cuda_lookup_padding_and_noncontiguous_inputs(shape, category):
    layer = JPEGArtifactModule().dc_layer0_dil[0].cuda()
    bins = torch.full(shape, category, device='cuda', dtype=torch.uint8).transpose(1, 2)
    volume = F.one_hot(bins.long(), 21).permute(0, 3, 1, 2).float()
    expected = F.conv2d(volume, layer.weight, layer.bias, padding=8, dilation=8)
    torch.testing.assert_close(layer(bins), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_uses_fused_kernel_instead_of_embedding_fallback(monkeypatch):
    pytest.importorskip('triton')

    def unexpected_embedding(*args, **kwargs):
        pytest.fail('CUDA lookup unexpectedly used the eager embedding fallback')

    monkeypatch.setattr(F, 'embedding', unexpected_embedding)
    module = JPEGArtifactModule().cuda()
    output = module(torch.zeros(1, 16, 24, device='cuda', dtype=torch.uint8),
                    torch.ones(1, 8, 8, device='cuda'))
    output.square().mean().backward()
    assert torch.isfinite(module.dc_layer0_dil[0].weight.grad).all()


def test_artifact_removes_dense_first_convolution_from_flop_budget():
    with torch.device('meta'):
        module = JPEGArtifactModule().eval()
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            module(torch.zeros(1, 1024, 1024, dtype=torch.uint8), torch.ones(1, 8, 8))
    assert counter.get_total_flops() / 1e9 == pytest.approx(0.536870912)


def test_baseline_fits_full_hd_budget_and_requires_native_size():
    from src.budget import count_gflops
    from src.config import load_experiment_config
    from src.training.builders import build_model

    config = load_experiment_config('configs/baseline.yaml')
    assert config.dataset.image_size == 640
    with torch.device('meta'):
        model = build_model(config.model, pretrained=False)
        with pytest.raises(ValueError, match='native_size'):
            count_gflops(model, 640)
        small = count_gflops(model, 640, native_size=(1024, 1024))
        large = count_gflops(model, 640, native_size=(1080, 1920))
    assert 0 < small < large < 100
