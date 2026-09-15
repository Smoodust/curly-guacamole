from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info

from src.data.augmentation.base import AugmentationStage
from src.data.augmentation.pipeline import AugmentationPipeline
from src.data.data_sample import DataSample
from src.data.data_workspace import DataWorkspace
from src.data.preprocess import SamplePreprocessor
from src.data.profiling import SampleTimer
from src.data.sample_io import SampleIO
from src.forensic.jpeg_input import JPEGInput


class AIIJCDataset(Dataset):
    """Read native JPEG inputs and prepare aligned image and mask tensors.

    Validation can retain original-resolution targets. Test samples also carry
    original_size and the CSV image_path for restoring and naming predictions.
    """

    def __init__(
            self,
            data_workspace: DataWorkspace,
            folded_df: pd.DataFrame,
            train: bool,
            image_size: int,
            seed: int,
            augmentations: AugmentationPipeline | None = None,
            mode: Literal["train", "val", "test"] | None = None,
            original_targets: bool | None = None,
    ):
        super().__init__()
        self.data_workspace = data_workspace
        self.mode = self._resolve_mode(train, mode)
        self.train = self.mode == "train"
        self.has_targets = self.mode in {"train", "val"}
        JPEGInput.prepare_decoder()
        self.profile_data = False
        self.image_size = image_size
        self.seed = seed
        self.original_targets = self.mode == "val" if original_targets is None else original_targets
        if self.original_targets and self.mode != "val":
            raise ValueError("original_targets is only supported for validation")
        # Shared tensor propagates epochs to persistent DataLoader workers on Windows too.
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self._rng_key = None
        self._rng = None
        self.augmentations = augmentations
        self.use_augmentations = self.train and augmentations is not None
        self.root = data_workspace.train_root if self.has_targets else data_workspace.test_root
        self.sample_io = SampleIO(self.root, self.has_targets)
        self.preprocessor = SamplePreprocessor(image_size)
        self.sample_io.validate_dataframe(folded_df)
        self.df = folded_df.reset_index(drop=True)
        self.is_negative = (
            self.df["is_negative"].to_numpy(dtype=bool)
            if "is_negative" in self.df.columns
            else None
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        timer = SampleTimer() if self.profile_data else None
        row = self.df.iloc[index]
        image_path = self._image_path(row)
        image = self.load_image(image_path)
        original_size = image.shape[:2]
        if timer:
            timer.mark('read_image')
        mask = self.load_mask(row, original_size) if self.has_targets else None
        if timer:
            timer.mark('read_mask')
        jpeg = JPEGInput.read(image_path)
        if timer:
            timer.mark('qtable')
        sample = DataSample(
            image=image,
            mask=mask,
            qtable=jpeg.qtable,
            jpeg=jpeg,
        )
        del mask
        rng = self._make_rng(index)
        original_mask = sample.mask if self.original_targets else None
        if self.use_augmentations:
            self.augmentations.set_epoch(int(self._epoch.item()))

        # Recompression updates native coefficients before crop and rotation.
        if timer:
            timer.mark('setup')
        sample = self._augment(AugmentationStage.BEFORE_FORENSICS, sample, rng)
        if timer:
            timer.mark('jpeg')
        sample = self._augment(AugmentationStage.AFTER_FORENSICS, sample, rng)
        if timer:
            timer.mark('geometry')
        sample = self.preprocessor.resize(sample)
        if timer:
            timer.mark('resize')
        sample = self._augment(AugmentationStage.FINAL, sample, rng)
        if timer:
            timer.mark('photometric')
        output = self.preprocessor.to_output(sample)
        output['jpeg'] = sample.jpeg.tensors()
        if original_mask is not None:
            if original_mask.shape != original_size:
                raise ValueError("original validation mask must match the image size")
            output["original_mask"] = torch.from_numpy(original_mask > 0.5)

        if self.mode == "test":
            output["original_size"] = torch.tensor(original_size, dtype=torch.int64)
            output["image_path"] = str(row["img_path"])
        if timer:
            timer.mark('tensorize')
            output['_worker_profile'] = timer.tensor()
        return output

    def _augment(
            self, stage: AugmentationStage, sample: DataSample, rng: np.random.Generator,
    ) -> DataSample:
        if self.use_augmentations:
            return self.augmentations.apply(stage, sample, rng)
        return sample

    @staticmethod
    def _resolve_mode(train: bool, mode: Literal["train", "val", "test"] | None) -> str:
        if mode is None:
            return "train" if train else "test"
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"unknown dataset mode: {mode}")
        return mode

    # Preserve existing path/loading entry points while keeping IO in one component.
    def _image_path(self, row: pd.Series) -> Path:
        return self.sample_io.image_path(row)

    def _mask_path(self, row: pd.Series) -> Path:
        return self.sample_io.mask_path(row)

    def _row_path(self, row: pd.Series, columns: tuple[str, ...]) -> Path:
        return self.sample_io.row_path(row, columns)

    def load_image(self, path: Path) -> np.ndarray:
        return self.sample_io.load_image(path)

    def load_mask(self, row: pd.Series, image_shape: tuple[int, int]) -> np.ndarray:
        # Some supplied masks are smaller than their upscaled images. Align them
        # as in training; original_targets retains the image resolution, not the file size.
        return self.sample_io.load_mask(row, image_shape)

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(epoch)

    def _make_rng(self, index: int) -> np.random.Generator:
        worker = get_worker_info()
        key = (self.seed, int(self._epoch.item()), worker.id if worker else 0)
        if key != self._rng_key:
            self._rng = np.random.default_rng(key)
            self._rng_key = key
        return self._rng
