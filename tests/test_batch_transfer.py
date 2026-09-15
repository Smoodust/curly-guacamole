import pytest
import torch


def test_batch_transfer_cpu_preserves_non_tensor_metadata():
    from src.training.transfer import BatchTransfer
    batch = {'image': torch.randn(2, 3, 8, 8), 'image_path': ['a', 'b']}
    actual = BatchTransfer('cpu')(batch)
    assert actual['image'] is batch['image']
    assert actual['image_path'] is batch['image_path']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA stream integration requires GPU')
def test_cuda_copy_stream_values_and_lifetime():
    from src.training.transfer import BatchTransfer
    transfer = BatchTransfer('cuda')
    outputs = []
    for i in range(20):
        batch = {'image': torch.full((2, 3, 64, 64), float(i), dtype=torch.float16).pin_memory()}
        moved = transfer(batch)
        outputs.append(moved['image'].float().sum())
        del batch, moved
    torch.cuda.synchronize()
    assert [value.item() for value in outputs] == [float(i * 2 * 3 * 64 * 64) for i in range(20)]
