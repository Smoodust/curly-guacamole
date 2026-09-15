import math

import torch
from torch.utils.data import Sampler


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


class DistributedValidationSampler(Sampler):
    """Contiguous, disjoint groups of whole batches; never pad or drop rows.

    Keeping the single-process batch boundaries also preserves conditional loss
    weighting. Concatenating rank outputs restores the original dataset order.
    """

    def __init__(self, dataset, batch_size, rank, world_size):
        batches = math.ceil(len(dataset) / batch_size)
        self.start = min(len(dataset), (batches * rank // world_size) * batch_size)
        self.stop = min(len(dataset), (batches * (rank + 1) // world_size) * batch_size)

    def __iter__(self):
        return iter(range(self.start, self.stop))

    def __len__(self):
        return self.stop - self.start


class DistributedBatchSampler(Sampler):
    """Partition one global draw stream without changing its sampling weights.

    Non-dropping epochs balance local batches so every rank performs the same
    number of backwards, even when the dataset size is not divisible by GPUs.
    No padding examples are introduced into full-train epochs.
    """

    def __init__(self, sampler, batch_size, rank, world_size, *, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.seed = 0

    def set_epoch(self, epoch, *, seed):
        self.seed = seed + epoch
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

    def __len__(self):
        return self.batch_count(len(self.sampler), self.batch_size, self.world_size, self.drop_last)

    @staticmethod
    def batch_count(size, batch_size, world_size, drop_last):
        if drop_last:
            count = size // (batch_size * world_size)
        else:
            count = math.ceil(math.ceil(size / world_size) / batch_size)
            if size // world_size < count:
                raise ValueError('Too few samples for equal DDP steps without padding; '
                                 'increase batch_size or use fewer devices')
        if count == 0:
            raise ValueError('epoch_size must contain at least one global distributed batch')
        return count

    def __iter__(self):
        self.sampler.generator = torch.Generator().manual_seed(self.seed)
        indices = list(self.sampler)
        count = len(self)
        if self.drop_last:
            indices = indices[:count * self.batch_size * self.world_size]
        local = indices[self.rank::self.world_size]
        # Consecutive balanced slices, all nonempty and at most batch_size.
        return iter(local[i * len(local) // count:(i + 1) * len(local) // count]
                    for i in range(count))
