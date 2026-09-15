import numpy as np
import torch

from src.config import load_experiment_config, ModelConfig, DatasetConfig
from src.data.data_sample import DataSample
from src.data.preprocess import SamplePreprocessor, imagenet_normalize
from src.modules.jpeg_branch import JPEGPointwiseConv2d
from src.training.builders import build_model


def test_b2_li768_disables_optimizations_only_for_this_experiment():
    config = load_experiment_config('configs/experiments/b2_li768.yaml')
    assert not config.model.jpeg_specialize_width
    assert not config.model.jpeg_pointwise_matmul
    assert not config.dataset.rgb_uint8_transport
    assert ModelConfig().jpeg_specialize_width
    assert ModelConfig().jpeg_pointwise_matmul
    assert DatasetConfig().rgb_uint8_transport
    model = build_model(config.model, pretrained=False)
    stem = model.forensic_fusion.branch.artifact
    assert not stem.dc_layer0_dil[0].specialize_width
    assert not stem.dc_layer1_tail[0].use_matmul


def test_legacy_rgb_transport_returns_normalized_float():
    sample = DataSample(np.random.default_rng(42).integers(0, 256, (16, 16, 3), dtype=np.uint8))
    result = SamplePreprocessor(16, rgb_uint8_transport=False).to_output(sample)['image']
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, imagenet_normalize(sample).image)


def test_legacy_pointwise_forward_and_backward():
    layer = JPEGPointwiseConv2d()
    layer.use_matmul = False
    reference = torch.nn.Conv2d(64, 4, 1, bias=False)
    reference.load_state_dict(layer.state_dict())
    x = torch.randn(1, 64, 8, 8, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    actual, expected = layer(x), reference(other)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, other.grad)
    torch.testing.assert_close(layer.weight.grad, reference.weight.grad)
