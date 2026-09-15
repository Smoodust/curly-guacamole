import io

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from src.forensic.jpeg_input import JPEGInput
from src.modules.jpeg_branch import JPEGArtifactModule


@pytest.mark.parametrize('subsampling', [0, 1, 2])
@pytest.mark.parametrize('progressive', [False, True])
def test_known_quantized_coefficients_and_block_order(subsampling, progressive):
    # For a constant 8x8 block, DC=8*(pixel-128). Q_DC=8 gives pixel-128.
    values = np.array([[127, 128, 129], [120, 140, 160]], np.uint8)
    pixels = np.repeat(np.repeat(values, 8, axis=0), 8, axis=1)
    rgb = np.repeat(pixels[..., None], 3, axis=2)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format='JPEG', qtables=[[8]*64, [8]*64],
                             subsampling=subsampling, progressive=progressive)
    native = JPEGInput.read(buffer.getvalue())
    expected = np.zeros((16, 24), np.uint8)
    expected[::8, ::8] = np.minimum(np.abs(values.astype(int)-128), 20)
    np.testing.assert_array_equal(native.bins, expected)
    np.testing.assert_array_equal(native.qtable, np.full((8, 8), 8))


def test_artifact_matches_independent_pixel_unshuffle_reference():
    torch.manual_seed(12)
    module = JPEGArtifactModule().eval()
    signed = torch.randint(-40, 41, (2, 24, 40))
    bins = signed.abs().clamp_max(20).to(torch.uint8)
    q = torch.arange(1, 129).reshape(2, 8, 8).float()
    # Reference categorical construction and channel ordering from CAT-Net,
    # independently expressed using one_hot + pixel_unshuffle.
    volume = F.one_hot(signed.abs().clamp_max(20), 21).permute(0, 3, 1, 2).float().contiguous()
    features = module.dc_layer1_tail(module.dc_layer0_dil(volume))
    raw = F.pixel_unshuffle(features, 8)
    expected = torch.cat((raw, raw*q.flatten(1).repeat(1, 4)[:, :, None, None]), dim=1)
    actual = module(bins, q)
    # Lookup changes summation order; quantization-table weighting amplifies
    # the resulting FP32 roundoff, without changing the channel ordering.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=3e-6)


@pytest.mark.parametrize('orientation', range(1, 9))
def test_composed_exif_crop_rotation_and_flips(orientation):
    from PIL import ImageOps

    from src.modules.jpeg_branch import JPEGBranch

    blocks = np.arange(24, dtype=np.uint8).reshape(4, 6)
    image = Image.fromarray(blocks)
    image.getexif()[274] = orientation
    displayed = np.array(ImageOps.exif_transpose(image))[1:3, 1:4]
    feature = torch.tensor(blocks, dtype=torch.float32)[None, None]
    for rotations in range(4):
        for horizontal in (False, True):
            for vertical in (False, True):
                expected = np.rot90(displayed, rotations)
                if horizontal:
                    expected = np.flip(expected, 1)
                if vertical:
                    expected = np.flip(expected, 0)
                actual = JPEGBranch.align(feature, (8, 8, 16, 24, rotations, horizontal, vertical),
                                          8, expected.shape, orientation=orientation, source_size=(32, 48))
                torch.testing.assert_close(actual[0, 0], torch.tensor(expected.copy(), dtype=torch.float32))
