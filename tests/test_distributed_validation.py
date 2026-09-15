from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from src.data.collation import ValidationCollator
from src.training.distributed import TrainingRuntime
from tests.test_distributed_training import _run_process_test
from tests.test_engine import _cpu_config


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, size, fail=False):
        self.size = size
        self.fail = fail

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.fail:
            raise ValueError('injected validation data failure')
        height, width = 5 + index, 6 + index
        mask = torch.zeros(1, 4, 4)
        original = torch.zeros(height, width, dtype=torch.bool)
        if index % 2:
            mask[:, :2, :2] = 1
            if index % 4 == 1:
                original[0, 0] = 1
            else:
                original[:height // 2, :width // 2] = 1
        return {'image': torch.full((3, 4, 4), float(index - 2)), 'mask': mask,
                'label': torch.tensor([float(index % 2)]), 'original_mask': original}


class PredictionModel(torch.nn.Module):
    def forward(self, images, fmap=None):
        return {'logits': images[:, :1], 'cls_logits': images[:, :1].mean((2, 3))}


@pytest.mark.parametrize('size,batch_size,world_size', [(13, 4, 3), (1, 4, 3), (8, 4, 2), (0, 4, 2)])
def test_validation_partitions_preserve_global_batches(size, batch_size, world_size):
    from src.training.sampling import DistributedValidationSampler

    actual = []
    for rank in range(world_size):
        sampler = DistributedValidationSampler(range(size), batch_size, rank, world_size)
        indices = list(sampler)
        assert len(indices) == len(sampler)
        actual.extend(indices[i:i + batch_size] for i in range(0, len(indices), batch_size))
    assert actual == [list(range(i, min(i + batch_size, size))) for i in range(0, size, batch_size)]


def _validation_worker(rank, folder, device='cpu'):
    from src.training.builders import build_amp
    from src.training.sampling import DistributedValidationSampler
    from src.training.validation import validate

    folder = Path(folder)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(folder / 'store').as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=40))
    try:
        runtime = TrainingRuntime(device, rank, 2)
        config = _cpu_config()
        config = replace(config, train=replace(config.train, device=device),
                         eval=replace(config.eval, selection_small_mask_weight=3.,
                                      mask_thresholds=(.25, .5, .75), cls_thresholds=(0., .5)))
        amp = build_amp(config.train)
        model = PredictionModel().to(device)
        for size in (13, 1):  # One empty rank in the second case.
            dataset = ValidationDataset(size)
            sampler = DistributedValidationSampler(dataset, 4, rank, 2)
            loader = DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=ValidationCollator())
            actual = validate(model, loader, amp, config, torch.device(device), runtime=runtime)
            reference = validate(model, DataLoader(dataset, batch_size=4, collate_fn=ValidationCollator()),
                                 amp, config, torch.device(device))
            assert actual.tuned.as_dict() == reference.tuned.as_dict()
            assert actual.fixed.as_dict() == reference.fixed.as_dict()
            # Float32 partial sums are combined in a different order across ranks.
            assert actual.loss_components == pytest.approx(reference.loss_components, rel=1e-6, abs=1e-7), (
                actual.loss_components, reference.loss_components)
            if rank == 0:
                for field in ('hist_all', 'hist_gt', 'gt_sum', 'n_pixels', 'cls_prob'):
                    np.testing.assert_array_equal(getattr(actual.accumulator, field),
                                                  getattr(reference.accumulator, field))
            else:
                assert len(actual.accumulator) == 0  # Full OOF lives only on rank zero.
            frozen = validate(model, loader, amp, config, torch.device(device),
                              runtime=runtime, thresholds=reference.tuned)
            assert frozen.tuned.as_dict() == reference.tuned.as_dict()

        dataset = ValidationDataset(13, fail=rank == 1)
        loader = DataLoader(dataset, batch_size=4,
                            sampler=DistributedValidationSampler(dataset, 4, rank, 2),
                            collate_fn=ValidationCollator())
        with pytest.raises(RuntimeError, match='rank 1.*injected validation data failure'):
            validate(model, loader, amp, config, torch.device(device), runtime=runtime)
    finally:
        dist.destroy_process_group()


def test_distributed_validation_matches_single_process_and_propagates_errors(tmp_path):
    _run_process_test('_validation_worker', tmp_path, 'tests.test_distributed_validation')


def _cuda_validation_worker(rank, folder):
    _validation_worker(rank, folder, 'cuda:0')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_two_cuda_processes_validate_on_one_physical_gpu(tmp_path):
    _run_process_test('_cuda_validation_worker', tmp_path, 'tests.test_distributed_validation')
