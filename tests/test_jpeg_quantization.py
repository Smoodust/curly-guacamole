import io

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.augmentation.base import AugmentationConfig, AugmentationStage
from src.data.augmentation.pipeline import AugmentationPipeline
from src.data.augmentation.transforms.random_jpeg_recompression import RandomJPEGRecompression
from src.data.data_sample import DataSample
from src.data.data_workspace import DataWorkspace
from src.data.dataset import AIIJCDataset
from src.forensic.jpeg import luma_qtable


def jpeg_bytes():
    buffer = io.BytesIO()
    rgb = np.random.default_rng(4).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(buffer, format="JPEG", qtables=[list(range(1, 65))])
    return buffer.getvalue()


def test_known_quantization_table_defaults_to_natural_order(tmp_path):
    data = jpeg_bytes()
    path = tmp_path / "image.jpg"
    path.write_bytes(data)
    expected = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
    for source in (data, bytearray(data), memoryview(data), path, str(path)):
        table = luma_qtable(source)
        np.testing.assert_array_equal(table, expected)
        assert table.dtype == np.float32


@pytest.mark.parametrize("order", ["natural", "legacy_zigzag", "typo"])
def test_removed_qtable_options_are_rejected(tmp_path, order):
    with pytest.raises(TypeError):
        luma_qtable(jpeg_bytes(), order=order)
    with pytest.raises(TypeError):
        RandomJPEGRecompression((70, 71), 1, jpeg_qtable_order=order)
    with pytest.raises(TypeError):
        AugmentationPipeline({}, jpeg_qtable_order=order)
    with pytest.raises(TypeError):
        AIIJCDataset(DataWorkspace(tmp_path), pd.DataFrame(), False, 32, 42,
                    jpeg_qtable_order=order)


def test_unreadable_and_non_jpeg_sources_have_no_table(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    for source in (b"invalid", buffer.getvalue(), tmp_path / "missing.jpg"):
        assert luma_qtable(source) is None


def test_pipeline_recompression_uses_natural_table():
    rgb = np.asarray(Image.open(io.BytesIO(jpeg_bytes())).convert("RGB"))
    pipeline = AugmentationPipeline(AugmentationConfig(
        jpeg_recompression_quality_range=(70, 71), jpeg_recompression_probability=1))
    sample = pipeline.apply(AugmentationStage.BEFORE_FORENSICS,
                            DataSample(image=rgb), np.random.default_rng(42))
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok
    with Image.open(io.BytesIO(encoded.tobytes())) as image:
        expected = np.asarray(image.quantization[0]).reshape(8, 8)
    expected_rgb = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(sample.qtable, expected)
    np.testing.assert_array_equal(sample.image, expected_rgb)


def test_dataset_uses_natural_table_for_native_jpeg(tmp_path):
    workspace = DataWorkspace(tmp_path)
    workspace.train_root.mkdir()
    image_path = workspace.train_root / "image.jpg"
    image_path.write_bytes(jpeg_bytes())
    Image.new("L", (48, 32), 255).save(workspace.train_root / "mask.png")
    rows = pd.DataFrame({"chng_img_path": ["image.jpg"], "gt_path": ["mask.png"]})
    dataset = AIIJCDataset(workspace, rows, False, 32, 42, mode="val")
    expected = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
    np.testing.assert_array_equal(dataset[0]["jpeg"]["qtable"].numpy(), expected)
