import numpy as np
import pytest
import torch

from src.training import validation
from src.training.metric import AICAccumulator


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='Requires CUDA'))])
def test_buffered_histograms_preserve_order_counts_and_thresholds(device):
    # Different original sizes, empty/full GT, exact bin boundaries and a tail.
    expected = AICAccumulator(n_bins=4)
    actual = AICAccumulator(n_bins=4)
    assert hasattr(validation, 'DeviceHistogramAccumulator')
    buffer = validation.DeviceHistogramAccumulator(actual, device, capacity=2)
    fixtures = [
        ([0., .25, .5, 1.], [0, 1, 1, 0], .8),
        ([.75, .1, .9], [0, 0, 0], .2),
        ([.25, .49], [1, 1], .6),
    ]
    for index, (values, targets, cls) in enumerate(fixtures):
        probs = torch.tensor(values, device=device).reshape(1, 1, 1, -1)
        masks = torch.tensor(targets, device=device).reshape_as(probs)
        confidence = torch.tensor([cls], device=device)
        expected.update(probs.cpu(), masks.cpu(), confidence.cpu())
        buffer.update(probs, masks, confidence)
        assert len(actual) == (2 if index >= 1 else 0)
    buffer.flush()
    buffer.flush()  # No duplicated rows when empty.
    for left, right in zip(actual.tables(), expected.tables(), strict=True):
        np.testing.assert_array_equal(left, right)
    assert actual.sweep([0., .25, .5, .75], [0., .5], [0., .1]) == expected.sweep(
        [0., .25, .5, .75], [0., .5], [0., .1])


def test_batch_crosses_chunk_boundary_without_losing_rows(monkeypatch):
    expected = AICAccumulator(n_bins=256)
    actual = AICAccumulator(n_bins=256)
    buffer = validation.DeviceHistogramAccumulator(actual, 'cpu', capacity=3)
    rng = torch.Generator().manual_seed(42)
    probs = torch.rand((8, 1, 7, 9), generator=rng)
    masks = torch.rand((8, 1, 7, 9), generator=rng) > .5
    cls = torch.rand(8, generator=rng)
    expected.update(probs, masks, cls)
    transfers = []
    original_cpu = torch.Tensor.cpu

    def record_cpu(tensor, *args, **kwargs):
        transfers.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'cpu', record_cpu)
    buffer.update(probs[:2], masks[:2], cls[:2])
    assert transfers == []
    buffer.update(probs[2:], masks[2:], cls[2:])
    buffer.flush()
    assert transfers == [(3, 514), (3,), (3, 514), (3,), (2, 514), (2,)]
    for left, right in zip(actual.tables(), expected.tables(), strict=True):
        np.testing.assert_array_equal(left, right)
