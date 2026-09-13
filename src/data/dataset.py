from collections.abc import Sequence
from dataclasses import replace
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
from src.data.sample_io import SampleIO
from src.data.targets import SignedDistanceTarget
from src.data.local_preprocess import LocalPreprocessor
from src.data.profiling import SampleTimer
from src.forensic.dct import forensic_maps, luma_qtable
from src.forensic.jpeg_input import JPEGInput


class AIIJCDataset(Dataset):
    """Read, augment, compute DCT maps, and prepare model inputs.

    Test samples additionally contain original_size (int64 [height, width]) and
    image_path (the original CSV value). Default collation yields Bx2 sizes.
    Forensic maps are computed per sample: no cache is justified without profiling,
    and JPEG augmentation changes both pixels and quantization tables each time.
    """

    def __init__(
            self,
            data_workspace: DataWorkspace,
            folded_df: pd.DataFrame,
            train: bool,
            image_size: int,
            seed: int,
            augmentations: AugmentationPipeline | None = None,
            fmap_channels: Sequence[int] | None = None,
            mode: Literal["train", "val", "test"] | None = None,
            original_targets: bool | None = None,
            resize_mode: str = "stretch",
            use_forensics: bool = True,
            forensic_mode: str = 'maps',
            jpeg_variant: str = 'baseline',
            local_image_size: int = 0,
            luma_image_size: int = 0,
            wavelet_image_size: int = 0,
            local_dtype: torch.dtype = torch.float32,
            strided_resize: bool = False,
            boundary_targets: bool = False,
    ):
        super().__init__()
        self.data_workspace = data_workspace
        self.mode = self._resolve_mode(train, mode)
        self.train = self.mode == "train"
        self.boundary_target = SignedDistanceTarget() if boundary_targets else None
        self.has_targets = self.mode in {"train", "val"}
        self.use_forensics = use_forensics
        if type(wavelet_image_size) is not int or wavelet_image_size < 0 or wavelet_image_size % 32:
            raise ValueError('wavelet_image_size must be 0 or a positive multiple of 32')
        if wavelet_image_size and (resize_mode != 'stretch' or local_image_size or luma_image_size or strided_resize):
            raise ValueError('wavelet_image_size requires stretch geometry and no other detail/resize branches')
        self.wavelet_image_size = wavelet_image_size
        if strided_resize and (use_forensics or local_image_size or luma_image_size):
            raise ValueError('strided_resize requires RGB-only without local/luma branches')
        self.strided_resize = strided_resize
        if forensic_mode not in {'maps', 'jpeg'}:
            raise ValueError('unknown forensic_mode')
        self.forensic_mode = forensic_mode
        self.jpeg_coefficients = jpeg_variant in {'signed', 'subblock4'}
        if forensic_mode == 'jpeg':
            if not use_forensics or resize_mode != 'stretch':
                raise ValueError('jpeg mode requires forensics and stretch geometry')
            JPEGInput.prepare_decoder()
        if local_image_size and resize_mode != 'stretch':
            raise ValueError('local_image_size currently requires stretch geometry')
        self.local_preprocessor = LocalPreprocessor(local_image_size) if local_image_size else None
        if luma_image_size and (resize_mode != 'stretch' or local_image_size):
            raise ValueError('luma_image_size requires stretch geometry and no local_image_size')
        self.luma_image_size = luma_image_size
        self.local_dtype = local_dtype
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
        self.fmap_channels = fmap_channels
        self.root = data_workspace.train_root if self.has_targets else data_workspace.test_root
        self.sample_io = SampleIO(self.root, self.has_targets)
        self.preprocessor = SamplePreprocessor(image_size, fmap_channels, resize_mode,
                                              strided_resize=strided_resize)
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
        jpeg = JPEGInput.read(image_path, include_coefficients=self.jpeg_coefficients) if self.forensic_mode == 'jpeg' else None
        qtable = jpeg.qtable if jpeg is not None else (luma_qtable(image_path) if self.use_forensics else None)
        if timer:
            timer.mark('qtable')
        sample = DataSample(
            image=image,
            mask=mask,
            qtable=qtable,
            jpeg=jpeg,
        )
        del mask, qtable
        rng = self._make_rng(index)
        original_mask = sample.mask if self.original_targets else None
        if self.use_augmentations:
            self.augmentations.set_epoch(int(self._epoch.item()))

        # Recompression must precede DCT extraction; maps use the full frame before crop.
        if timer:
            timer.mark('setup')
        sample = self._augment(AugmentationStage.BEFORE_FORENSICS, sample, rng)
        if timer:
            timer.mark('jpeg')
        if self.use_forensics and self.forensic_mode == 'maps':
            fmap = forensic_maps(sample.image, sample.qtable)
        else:
            # Keep the same crop/resize geometry without extracting DCT features.
            h, w = sample.image.shape[:2]
            fmap = np.zeros((1, h // 8, w // 8), dtype=np.float32)
        sample = replace(sample, fmap=fmap)
        if timer:
            timer.mark('dct')
        sample = self._augment(AugmentationStage.AFTER_FORENSICS, sample, rng)
        if timer:
            timer.mark('geometry')
        local_input = None
        native_rgb = None
        if self.local_preprocessor is not None or self.luma_image_size or self.wavelet_image_size:
            # One appearance draw on native geometry shared by both views.
            sample = self._augment(AugmentationStage.FINAL, sample, rng)
            if timer:
                timer.mark('photometric')
            # Match the first local convolution's AMP cast before IPC/H2D.
            # Native residual extraction and resize always remain float32.
            if self.local_preprocessor is not None:
                local_input = self.local_preprocessor(sample.image, dtype=self.local_dtype)
            elif self.wavelet_image_size:
                # Keep decoded HWC storage; permute is a view, not a full CHW copy.
                # Negative-stride augmentation views are materialized only if needed.
                native_rgb = torch.from_numpy(np.ascontiguousarray(sample.image)).permute(2, 0, 1)
            else:
                native_rgb = torch.from_numpy(np.ascontiguousarray(sample.image.transpose(2, 0, 1)))
            if timer:
                timer.mark('local_features')
            # Conversion is now included in local_features; local_cast stays zero.
            sample = self.preprocessor.resize(sample)
            if timer:
                timer.mark('resize')
        else:
            sample = self.preprocessor.resize(sample)
            if timer:
                timer.mark('resize')
            sample = self._augment(AugmentationStage.FINAL, sample, rng)
            if timer:
                timer.mark('photometric')
        output = self.preprocessor.to_output(sample)
        if self.boundary_target is not None and self.has_targets:
            output['boundary_distance'] = self.boundary_target(output['mask'], output.get('valid_mask'))
        if sample.jpeg is not None:
            output['jpeg'] = sample.jpeg.tensors()
        if local_input is not None:
            output['local_input'] = local_input
        if native_rgb is not None:
            output['native_rgb'] = native_rgb
        if not self.use_forensics or self.forensic_mode == 'jpeg':
            output.pop("fmap")
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
