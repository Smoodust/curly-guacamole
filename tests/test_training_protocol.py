from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import ExperimentConfig, load_experiment_config
from src.data.augmentation.base import AugmentationConfig, AugmentationStage
from src.data.augmentation.pipeline import AugmentationPipeline
from src.data.augmentation.transforms.random_dct_aligned_crop import RandomDCTAlignedCrop
from src.data.data_sample import DataSample
from src.data.data_workspace import DataWorkspace
from src.data.dataset import AIIJCDataset


def make_dataset(tmp_path):
    rows = pd.DataFrame([{"chng_img_path": "image.jpg", "gt_path": "mask.png"}])
    return AIIJCDataset(DataWorkspace(tmp_path), rows, True, 32, 42)


def test_repeated_samples_get_new_reproducible_augmentations(tmp_path):
    first, second = make_dataset(tmp_path), make_dataset(tmp_path)
    draws = [first._make_rng(0).random(8) for _ in range(3)]
    assert not np.array_equal(draws[0], draws[1])
    for expected in draws:
        np.testing.assert_array_equal(expected, second._make_rng(0).random(8))


def test_epoch_changes_rng_and_can_be_reproduced_on_resume(tmp_path):
    dataset = make_dataset(tmp_path)
    initial = dataset._make_rng(0).random(8)
    dataset.set_epoch(4)
    later = dataset._make_rng(0).random(8)
    resumed = make_dataset(tmp_path)
    resumed.set_epoch(4)
    assert not np.array_equal(initial, later)
    np.testing.assert_array_equal(later, resumed._make_rng(0).random(8))


def sample_with_small_mask():
    blocks = np.arange(24 * 32, dtype=np.float32).reshape(24, 32)
    pixels = blocks.repeat(8, 0).repeat(8, 1)
    mask = np.zeros(pixels.shape, np.float32)
    mask[170:172, 230:232] = 1.0  # SampleIO returns binary float targets.
    return DataSample(np.repeat(pixels[..., None], 3, 2), mask=mask)


def test_foreground_crop_keeps_small_gt_and_dct_alignment():
    crop = RandomDCTAlignedCrop((0.15, 0.2), False, foreground_probability=1.0)
    for seed in range(20):
        result = crop.apply(sample_with_small_mask(), np.random.default_rng(seed))
        assert result.mask.max() == 1.0
        assert result.image.shape[0] % 8 == result.image.shape[1] % 8 == 0
        block = result.image[::8, ::8, 0]
        np.testing.assert_array_equal(result.image[..., 0], block.repeat(8, 0).repeat(8, 1))


def test_mixed_frames_and_final_full_frame_epochs():
    config = AugmentationConfig((0.15, 0.2), 0.0, (60, 100),
                                full_frame_probability=0.5, final_full_frame_epochs=2)
    pipeline = AugmentationPipeline(config, total_epochs=8)
    sample = sample_with_small_mask()
    rng = np.random.default_rng(42)
    shapes = [pipeline.apply(AugmentationStage.AFTER_FORENSICS, sample, rng).image.shape
              for _ in range(40)]
    assert sample.image.shape in shapes
    assert any(shape != sample.image.shape for shape in shapes)
    pipeline.set_epoch(6)
    for _ in range(10):
        assert pipeline.apply(AugmentationStage.AFTER_FORENSICS, sample, rng).image.shape == sample.image.shape


def test_mixed_defaults_and_overrides_roundtrip():
    config = load_experiment_config("configs/baseline.yaml")
    assert config.augmentation.full_frame_probability == 0.5
    raw = config.to_dict()
    raw["augmentation"].update(full_frame_probability=0.5, foreground_crop_probability=0.5,
                               final_full_frame_epochs=2)
    updated = ExperimentConfig.from_dict(raw)
    assert ExperimentConfig.from_dict(updated.to_dict()) == updated


def test_resume_rejects_changed_validation_before_overwriting_run(tmp_path):
    from src.training.engine import ExperimentRunner
    from src.training.runs import Run

    config = load_experiment_config("configs/baseline.yaml")
    config = replace(config, paths=replace(config.paths, runs_path=tmp_path),
                     train=replace(config.train, resume=True))
    run = Run.create(tmp_path, config.run_name, tensorboard=False)
    run.save_snapshot({"resolution": "resized"})
    (run.dir / "ckpt" / "last.pt").touch()
    with pytest.raises(ValueError, match="pipeline"):
        ExperimentRunner(config)._check_resume_protocol()
    assert run.snapshot == {"resolution": "resized"}


def test_original_validation_matches_submission_for_mixed_sizes():
    from src.inference.predict import Predictor, ThresholdConfig
    from src.training.builders import build_amp
    from src.training.metric import score_masks
    from src.training.validation import validate

    class Model(torch.nn.Module):
        def forward(self, image, fmap=None):
            return {"logits": image[:, :1], "cls_logits": torch.full((len(image), 1), 8.0)}

    config = load_experiment_config("configs/baseline.yaml")
    config = replace(config, train=replace(config.train, device="cpu", amp="off"),
                     eval=replace(config.eval, mask_thresholds=(0.5,),
                                  cls_thresholds=(0.0,), min_areas=(0.0,)))
    amp = build_amp(config.train)
    image = torch.tensor([[[[-2., 2.], [-2., 2.]]]]).repeat(2, 3, 1, 1)
    masks = [torch.zeros(3, 7, dtype=torch.bool), torch.zeros(9, 4, dtype=torch.bool)]
    masks[0][1, 4] = True  # Tiny GT must remain positive even if resized GT disappears.
    batch = {"image": image, "mask": torch.zeros(2, 1, 2, 2), "original_mask": masks}
    result = validate(Model(), [batch], amp, config, torch.device("cpu"))
    predictor = Predictor(Model(), ThresholdConfig(), amp)
    predictions = [predictor.binary_mask(image[i:i+1, :1].sigmoid(), 1., tuple(mask.shape))
                   for i, mask in enumerate(masks)]
    direct = score_masks(predictions, [mask.numpy() for mask in masks])
    assert result.accumulator.n_pixels == [21, 36]
    assert result.tuned.n_pos == 1
    assert result.tuned.n_neg == 1
    assert result.tuned.dice_pos == pytest.approx(direct.dice_pos)
    assert result.tuned.fpr_neg == direct.fpr_neg


def test_dataset_preserves_original_gt_and_collates_variable_sizes(tmp_path):
    import cv2

    from src.data.collation import ValidationCollator

    workspace = DataWorkspace(tmp_path)
    workspace.train_root.mkdir(parents=True)
    rows = []
    for i, (h, w) in enumerate([(24, 40), (40, 24)]):
        cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8))[1].tofile(workspace.train_root / f"{i}.jpg")
        mask = np.zeros((h, w), np.uint8)
        mask[3, 5] = 255
        cv2.imencode(".png", mask)[1].tofile(workspace.train_root / f"{i}.png")
        rows.append({"chng_img_path": f"{i}.jpg", "gt_path": f"{i}.png"})
    dataset = AIIJCDataset(workspace, pd.DataFrame(rows), False, 16, 42,
                           mode="val", original_targets=True)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=ValidationCollator())))
    assert batch["image"].shape == (2, 3, 16, 16)
    assert [tuple(mask.shape) for mask in batch["original_mask"]] == [(24, 40), (40, 24)]
    assert [int(mask.sum()) for mask in batch["original_mask"]] == [1, 1]

    # Supplied GT can be smaller than the image; evaluation uses the image resolution.
    cv2.imencode(".png", np.ones((12, 20), np.uint8) * 255)[1].tofile(workspace.train_root / "0.png")
    aligned = dataset[0]["original_mask"]
    assert aligned.shape == (24, 40)
    assert aligned.all()


class RandomDrawDataset(AIIJCDataset):
    def __getitem__(self, index):
        return self._make_rng(index).random(8)


def test_epoch_propagates_to_persistent_spawn_workers(tmp_path):
    rows = pd.DataFrame([{"chng_img_path": "unused.jpg", "gt_path": "unused.png"}])
    dataset = RandomDrawDataset(DataWorkspace(tmp_path), rows, True, 32, 42)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=[0, 0],
                                        num_workers=1, persistent_workers=True,
                                        multiprocessing_context="spawn")
    try:
        first_epoch = list(loader)
        assert not torch.equal(first_epoch[0], first_epoch[1])
        dataset.set_epoch(1)
        second_epoch = list(loader)
        assert not torch.equal(first_epoch[0], second_epoch[0])
        reference = RandomDrawDataset(DataWorkspace(tmp_path), rows, True, 32, 42)
        reference.set_epoch(1)
        np.testing.assert_array_equal(second_epoch[0][0].numpy(), reference[0])
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
