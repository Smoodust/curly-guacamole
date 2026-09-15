from pathlib import Path

import pandas as pd
import pytest


def test_stage0_public_imports():
    import src.data.augmentation.pipeline  # noqa: F401
    import src.modules.forensic_fusion  # noqa: F401
    import src.training.builders  # noqa: F401


def test_dataset_modes_resolve_paths():
    from src.data.data_workspace import DataWorkspace
    from src.data.dataset import AIIJCDataset

    workspace = DataWorkspace(Path("D:/data"))
    train_row = {
        "chng_img_path": r"stage1\train\img\changed.jpg",
        "gt_path": r"stage1\train\mask\changed.png",
        "is_negative": False,
    }
    test_row = {
        "img_path": r"test_stage1_img\sample.jpg",
    }

    train_ds = AIIJCDataset(
        workspace,
        pd.DataFrame([train_row]),
        train=True,
        image_size=640,
        seed=42,
        mode="train",
    )
    val_ds = AIIJCDataset(
        workspace,
        pd.DataFrame([train_row]),
        train=False,
        image_size=640,
        seed=42,
        mode="val",
    )
    test_ds = AIIJCDataset(
        workspace,
        pd.DataFrame([test_row]),
        train=False,
        image_size=640,
        seed=42,
        mode="test",
    )

    assert train_ds._image_path(train_ds.df.iloc[0]) == workspace.train_root / "stage1/train/img/changed.jpg"
    assert train_ds._mask_path(train_ds.df.iloc[0]) == workspace.train_root / "stage1/train/mask/changed.png"
    assert val_ds._image_path(val_ds.df.iloc[0]) == workspace.train_root / "stage1/train/img/changed.jpg"
    assert val_ds._mask_path(val_ds.df.iloc[0]) == workspace.train_root / "stage1/train/mask/changed.png"
    assert test_ds._image_path(test_ds.df.iloc[0]) == workspace.test_root / "test_stage1_img/sample.jpg"
    assert train_ds.use_augmentations is False
    assert val_ds.has_targets is True
    assert val_ds.use_augmentations is False
    assert test_ds.has_targets is False

    with pytest.raises(ValueError):
        test_ds._mask_path(test_ds.df.iloc[0])


def test_build_datasets_uses_typed_config():
    from src.config import load_experiment_config
    from src.data.data_workspace import DataWorkspace
    from src.training.builders import build_datasets

    workspace = DataWorkspace(Path("D:/data"))
    config = load_experiment_config("configs/baseline.yaml")
    df = pd.DataFrame(
        [
            {
                "chng_img_path": "stage1/train/img/changed.jpg",
                "gt_path": "stage1/train/mask/changed.png",
                "is_negative": False,
            },
            {
                "chng_img_path": "stage1/train/img/negative.jpg",
                "gt_path": "stage1/train/mask/negative.png",
                "is_negative": True,
            },
        ]
    )

    train_ds, val_ds = build_datasets(config, workspace, df.iloc[:1], df.iloc[1:])

    assert train_ds.mode == "train"
    assert train_ds.image_size == 640
    assert train_ds.use_augmentations is True
    assert val_ds.mode == "val"
    assert val_ds.has_targets is True
    assert val_ds.use_augmentations is False


def test_forensic_fusion_constructs():
    from src.modules.forensic_fusion import ForensicFusion

    fusion = ForensicFusion(
        encoder_strides=[4, 8, 16, 32],
        encoder_channels=[8, 16, 32, 64],
        jpeg_channels=(8, 16, 32),
    )

    assert set(fusion.fusion_blocks.keys()) == {"8", "16", "32"}


def test_segmenter_dummy_forward(monkeypatch):
    torch = pytest.importorskip("torch")

    import src.modules.utils as module_utils
    from src.modules.segmenter import Segmenter

    class FakeFeatureInfo:
        @staticmethod
        def reduction():
            return [4, 8, 16, 32]

        @staticmethod
        def channels():
            return [8, 16, 32, 64]

    class FakeEncoder(torch.nn.Module):
        feature_info = FakeFeatureInfo()

        def forward(self, image):
            batch, _, height, width = image.shape
            return [
                image.new_zeros(batch, 8, height // 4, width // 4),
                image.new_zeros(batch, 16, height // 8, width // 8),
                image.new_zeros(batch, 32, height // 16, width // 16),
                image.new_zeros(batch, 64, height // 32, width // 32),
            ]

    def fake_create_model(*args, **kwargs):
        return FakeEncoder()

    monkeypatch.setattr(module_utils.timm, "create_model", fake_create_model)

    model = Segmenter(
        encoder="fake_encoder",
        jpeg_channels=(8, 16, 32),
        aux_weight=0.4,
    )
    model.train()

    image = torch.zeros(2, 3, 64, 64)
    out = model(image, jpeg=[{'available': False}] * 2)

    assert out["logits"].shape == (2, 1, 64, 64)
    assert out["cls_logits"].shape == (2, 1)
    assert out["aux_logits"].shape == (2, 1, 64, 64)
