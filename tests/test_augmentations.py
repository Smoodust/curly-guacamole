import numpy as np
import pytest

from src.data.augmentation.base import AugmentationConfig, AugmentationStage
from src.data.augmentation.pipeline import AugmentationPipeline
from src.data.augmentation.transforms.random_dct_aligned_crop import RandomDCTAlignedCrop
from src.data.augmentation.transforms.random_rotate_flip import RandomRotateFlip
from src.data.data_sample import DataSample
from src.forensic.jpeg_input import JPEGInput


@pytest.mark.parametrize("seed", range(10))
def test_crop_and_flip_keep_image_mask_and_dct_blocks_aligned(seed):
    blocks = np.arange(16 * 20, dtype=np.float32).reshape(16, 20)
    pixels = blocks.repeat(8, axis=0).repeat(8, axis=1)
    sample = DataSample(
        image=np.repeat(pixels[..., None], 3, axis=2),
        mask=pixels.copy(),
        jpeg=JPEGInput(np.zeros(pixels.shape, np.uint8), np.ones((8, 8), np.float32),
                       (0, 0, *pixels.shape, 0, 0, 0)),
    )
    rng = np.random.default_rng(seed)
    cropped = RandomDCTAlignedCrop((0.25, 0.5), full_frame=False).apply(sample, rng)
    transformed = RandomRotateFlip(full_frame=False).apply(cropped, rng)

    assert cropped.image.shape[0] < sample.image.shape[0]
    np.testing.assert_array_equal(transformed.image[..., 0], transformed.mask)
    top, left, height, width, rotations, horizontal, vertical = transformed.jpeg.geometry
    expected = pixels[top:top+height, left:left+width]
    expected = RandomRotateFlip.rotate_flip(expected, rotations, horizontal, vertical, (0, 1))
    np.testing.assert_array_equal(expected, transformed.mask)
    assert transformed.image.flags.c_contiguous


def test_default_augmentation_schedule():
    config = AugmentationConfig()
    assert config.crop_scale_range == (0.35, 1.0)
    assert config.jpeg_recompression_probability == 0.3
    assert config.jpeg_recompression_quality_range == (60, 100)
    assert config.foreground_crop_probability == 0.5
    pipeline = AugmentationPipeline(config)
    for epoch, expected in enumerate([0.5, 0.5, 0.5, 0.5, 1.0, 1.0]):
        pipeline.set_epoch(epoch)
        assert pipeline.full_frame_probability == expected
    pipeline.set_epoch(0)
    assert pipeline.full_frame_probability == 0.5


def test_full_frame_config_switch_is_removed():
    with pytest.raises(TypeError):
        AugmentationConfig(full_frame=True)


@pytest.mark.parametrize("total_epochs,final_epochs,expected", [
    (4, 2, [0, 0, 1, 1]), (3, 0, [0, 0, 0]), (2, 3, [1, 1]),
])
def test_crop_and_final_full_frame_geometry(total_epochs, final_epochs, expected):
    blocks = np.arange(16 * 20, dtype=np.float32).reshape(16, 20)
    pixels = blocks.repeat(8, axis=0).repeat(8, axis=1)
    sample = DataSample(image=np.repeat(pixels[..., None], 3, axis=2),
                        mask=pixels.copy())
    pipeline = AugmentationPipeline(AugmentationConfig(
        crop_scale_range=(0.25, 0.5), full_frame_probability=0,
        final_full_frame_epochs=final_epochs), total_epochs=total_epochs)
    for epoch, full_frame in enumerate(expected):
        pipeline.set_epoch(epoch)
        output = pipeline.apply(AugmentationStage.AFTER_FORENSICS, sample,
                                np.random.default_rng(42))
        if full_frame:
            assert output.image.shape == sample.image.shape
        else:
            assert output.image.size < sample.image.size
        np.testing.assert_array_equal(output.image[..., 0], output.mask)
