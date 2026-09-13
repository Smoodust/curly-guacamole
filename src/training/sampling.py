import numpy as np
import torch
from torch.utils.data import Sampler


class UniformMaskAreaSampler(torch.utils.data.WeightedRandomSampler):
    """Equal probability for empty masks and five positive area buckets."""

    def __init__(self, mask_area, num_samples):
        area = np.asarray(mask_area, dtype=float)
        if area.ndim != 1 or not np.isfinite(area).all() or ((area < 0) | (area > 1)).any():
            raise ValueError('mask_area must contain finite fractions in [0, 1]')
        buckets = np.digitize(area, [.01, .03, .08, .20]) + 1
        buckets[area == 0] = 0
        counts = np.bincount(buckets, minlength=6)
        if (counts == 0).any():
            raise ValueError('uniform_mask_area requires all six mask area categories')
        weights = 1.0 / counts[buckets]
        super().__init__(torch.as_tensor(weights, dtype=torch.double), num_samples, replacement=True)


class FinalFullTrainSampler(Sampler):
    """Weighted draws initially; one shuffled permutation per final epoch."""

    def __init__(self, weighted_sampler, dataset_size, first_full_epoch):
        self.weighted_sampler = weighted_sampler
        self.dataset_size = dataset_size
        self.first_full_epoch = first_full_epoch
        self.epoch = 0
        self.generator = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.dataset_size if self.epoch >= self.first_full_epoch else len(self.weighted_sampler)

    def __iter__(self):
        if self.epoch >= self.first_full_epoch:
            return iter(torch.randperm(self.dataset_size, generator=self.generator).tolist())
        self.weighted_sampler.generator = self.generator
        return iter(self.weighted_sampler)
