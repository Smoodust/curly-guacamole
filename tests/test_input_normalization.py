import numpy as np
import pytest
import torch

from src.data.data_sample import DataSample
from src.data.preprocess import SamplePreprocessor, imagenet_normalize
from src.modules.segmenter import Segmenter
from src.modules.input_normalization import ImageNetInputNormalization


def test_preprocessor_transports_uint8_without_changing_pixels():
    image = np.random.default_rng(42).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    output = SamplePreprocessor(32).to_output(DataSample(image))
    assert output['image'].dtype == torch.uint8
    np.testing.assert_array_equal(output['image'].numpy(), image.transpose(2, 0, 1))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_uint8_model_matches_legacy_normalized_input(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = Segmenter(pretrained=False).to(device).eval()
    image = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    raw = torch.from_numpy(image.transpose(2, 0, 1).copy())[None].to(device)
    normalized = imagenet_normalize(DataSample(image)).image[None].to(device)
    jpeg = [{'available': False}]
    with torch.inference_mode():
        expected = model(normalized, jpeg=jpeg)
        actual = model(raw, jpeg=jpeg)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_normalization_all_pixel_values_in_amp_preserves_fp32_and_layout(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    image = np.repeat(np.arange(256, dtype=np.uint8).reshape(16, 16, 1), 3, axis=2)
    raw = torch.from_numpy(image.transpose(2, 0, 1).copy())[None].to(
        device, memory_format=torch.channels_last)
    reference = imagenet_normalize(DataSample(image)).image[None].to(device)
    layer = ImageNetInputNormalization().to(device).half()
    with torch.autocast(device, dtype=torch.bfloat16):
        actual = layer(raw)
    assert actual.dtype == torch.float32
    assert actual.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(actual, reference, rtol=0, atol=3e-7)
    assert layer(reference) is reference
    assert not layer.state_dict()
