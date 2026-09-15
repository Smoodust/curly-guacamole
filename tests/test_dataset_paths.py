import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.collation import ValidationCollator
from src.data.data_workspace import DataWorkspace
from src.data.dataset import AIIJCDataset


@pytest.mark.parametrize("mode", ["train", "val", "test"])
def test_dataset_reads_paths_and_preserves_targets(tmp_path, mode):
    workspace = DataWorkspace(tmp_path)
    root = workspace.test_root if mode == "test" else workspace.train_root
    (root / "images").mkdir(parents=True)
    image = np.zeros((24, 40, 3), dtype=np.uint8)
    image[..., 2] = 255
    cv2.imencode(".png", image)[1].tofile(root / "images" / "РѕР±СЂР°Р·РµС†.png")
    row = {"img_path" if mode == "test" else "chng_img_path": r"images\РѕР±СЂР°Р·РµС†.png"}
    if mode != "test":
        row["gt_path"] = "mask.png"
        cv2.imencode(".png", np.full((24, 40), 255, dtype=np.uint8))[1].tofile(root / "mask.png")
    dataset = AIIJCDataset(workspace, pd.DataFrame([row]), mode == "train", 32, 42, mode=mode)

    output = dataset[0]

    assert ("original_mask" in output) == (mode == "val")
    if mode == "val":
        assert output["original_mask"].shape == (24, 40)
        assert output["original_mask"].all()
    assert output["image"].shape == (3, 32, 32)
    assert output["jpeg"]["available"] is False
    assert "fmap" not in output
    assert output["image"][0].mean() > output["image"][2].mean()  # RGB conversion
    if mode == "test":
        assert "mask" not in output
        assert "label" not in output
        assert output["original_size"].tolist() == [24, 40]
        assert output["image_path"] == row["img_path"]
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=ValidationCollator())))
        assert batch["original_size"].shape == (1, 2)
    else:
        assert output["mask"].shape == (1, 32, 32)
        assert torch.all(output["mask"] == 1)
        assert output["label"].tolist() == [1.0]


@pytest.mark.parametrize("mode, row", [
    ("train", {"chng_img_path": "x.png"}),
    ("val", {"gt_path": "x.png"}),
    ("test", {"chng_img_path": "x.png"}),
])
def test_dataset_rejects_missing_columns_before_iteration(tmp_path, mode, row):
    with pytest.raises(ValueError, match="columns"):
        AIIJCDataset(DataWorkspace(tmp_path), pd.DataFrame([row]), False, 32, 42, mode=mode)


@pytest.mark.parametrize("path", [None, "", "  "])
def test_dataset_rejects_empty_image_paths(tmp_path, path):
    with pytest.raises(ValueError, match="img_path"):
        AIIJCDataset(DataWorkspace(tmp_path), pd.DataFrame([{"img_path": path}]), False, 32, 42)


@pytest.mark.parametrize("positive", [False, True])
def test_original_validation_aligns_smaller_mask_to_image(tmp_path, positive):
    workspace = DataWorkspace(tmp_path)
    root = workspace.train_root
    root.mkdir(parents=True)
    image = np.zeros((48, 80, 3), dtype=np.uint8)
    mask = np.zeros((24, 40), dtype=np.uint8)
    if positive:
        mask[:, 20:] = 255
    cv2.imencode(".png", image)[1].tofile(root / "image.png")
    cv2.imencode(".png", mask)[1].tofile(root / "mask.png")
    dataset = AIIJCDataset(
        workspace, pd.DataFrame([{"chng_img_path": "image.png", "gt_path": "mask.png"}]),
        False, 32, 42, mode="val",
    )

    output = dataset[0]

    expected = torch.zeros((48, 80), dtype=torch.bool)
    if positive:
        expected[:, 40:] = True
    assert torch.equal(output["original_mask"], expected)
    assert output["mask"].shape == (1, 32, 32)
    assert output["label"].tolist() == [float(positive)]


@pytest.mark.parametrize("original_targets", [False, True])
def test_validation_original_targets_can_be_overridden(tmp_path, original_targets):
    workspace = DataWorkspace(tmp_path)
    workspace.train_root.mkdir(parents=True)
    cv2.imencode(".png", np.zeros((24, 40, 3), dtype=np.uint8))[1].tofile(workspace.train_root / "image.png")
    cv2.imencode(".png", np.zeros((24, 40), dtype=np.uint8))[1].tofile(workspace.train_root / "mask.png")
    rows = pd.DataFrame([{"chng_img_path": "image.png", "gt_path": "mask.png"}])
    dataset = AIIJCDataset(workspace, rows, False, 32, 42, mode="val",
                          original_targets=original_targets)
    assert ("original_mask" in dataset[0]) == original_targets
