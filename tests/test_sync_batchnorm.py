import numpy as np  # noqa: F401 - initialize the conda runtime before torch
import pytest
import torch
from torch import nn

from src.config import ModelConfig
from src.modules.jpeg_branch import JPEGFrameBatchNorm2d
from src.training.builders import build_model
from src.training.distributed import TrainingRuntime


def _small_fusion():
    from types import SimpleNamespace

    from src.modules.forensic_fusion import ForensicFusion
    from src.modules.sync_batchnorm import SynchronizedBatchNorm

    fusion = ForensicFusion((4, 8, 16, 32), (4, 8, 16, 32), (8, 16, 32))
    SynchronizedBatchNorm.apply(SimpleNamespace(encoder=nn.Identity(), decoder=nn.Identity(), forensic_fusion=fusion))
    features = [torch.randn(2, c, 64 // s, 64 // s, requires_grad=True)
                for s, c in zip((4, 8, 16, 32), (4, 8, 16, 32), strict=True)]
    return fusion, features


def test_empty_and_mixed_jpeg_batches_need_no_fusion_collectives(monkeypatch):
    fusion, features = _small_fusion()

    def unexpected_collective(*args, **kwargs):
        pytest.fail('Local fusion must not synchronize variable native JPEG batches')

    monkeypatch.setattr(torch.distributed, 'all_reduce', unexpected_collective)
    output = fusion(features, jpeg=[{'available': False}, {'available': False}])
    for original, actual in zip(features, output, strict=True):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    sample = dict(bins=torch.zeros(64, 64, dtype=torch.uint8), qtable=torch.ones(8, 8),
                  geometry=(0, 0, 64, 64, 0, 0, 0))
    output = fusion(features, jpeg=[{'available': False}, sample])
    sum(value.square().mean() for value in output).backward()
    for original, actual in zip(features, output, strict=True):
        torch.testing.assert_close(original[0], actual[0], rtol=0, atol=0)


def test_sync_conversion_preserves_checkpoint_structure_and_native_normalization():
    from src.modules.segmenter import Segmenter
    from src.modules.sync_batchnorm import SynchronizedBatchNorm

    with torch.device('meta'):
        reference = Segmenter(pretrained=False)
        model = SynchronizedBatchNorm.apply(Segmenter(pretrained=False))
    assert reference.state_dict().keys() == model.state_dict().keys()
    assert any(isinstance(layer, nn.SyncBatchNorm) for layer in model.decoder.modules())
    assert all(not isinstance(layer, nn.SyncBatchNorm)
               for layer in model.forensic_fusion.branch.modules())
    assert any(isinstance(layer, JPEGFrameBatchNorm2d)
               for layer in model.forensic_fusion.branch.modules())
    assert not any(isinstance(layer, nn.SyncBatchNorm)
                   for layer in model.forensic_fusion.fusion_blocks.modules())


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_sync_checkpoint_inference_parity_and_single_device_training(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    from src.modules.segmenter import Segmenter
    reference = Segmenter(pretrained=False).to(device).eval()
    model = build_model(ModelConfig(), pretrained=False).to(device).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    image = torch.randn(2, 3, 64, 64, device=device)
    jpeg = [dict(bins=torch.randint(0, 21, (64, 64), device=device, dtype=torch.uint8),
                 qtable=torch.ones(8, 8, device=device), geometry=(0, 0, 64, 64, 0, 0, 0)) for _ in range(2)]
    with torch.no_grad():
        expected = reference(image, jpeg=jpeg)
        actual = model(image, jpeg=jpeg)
    torch.testing.assert_close(actual['logits'], expected['logits'])
    del reference
    model.train()
    for block in model.forensic_fusion.fusion_blocks.values():
        with torch.no_grad():
            block.channel_gate.fill_(0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    output = model(image, jpeg=jpeg)
    (output['logits'].square().mean() + output['aux_logits'].square().mean()).backward()
    grad = model.forensic_fusion.branch.artifact.dc_layer0_dil[0].weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    optimizer.step()


@pytest.mark.parametrize('device,world_size,backend,allowed', [
    ('cpu', 1, None, True), ('cuda', 1, None, True),
    ('cpu', 2, 'gloo', False), ('cuda', 2, 'gloo', False),
    ('cuda', 2, 'nccl', True),
])
def test_sync_runtime_rejects_unsupported_collectives(monkeypatch, device, world_size, backend, allowed):
    monkeypatch.setattr(torch.distributed, 'get_backend', lambda: backend)
    runtime = TrainingRuntime(device, world_size=world_size)
    runtime.validate_sync_batchnorm(False)
    if allowed:
        runtime.validate_sync_batchnorm(True)
    else:
        with pytest.raises(ValueError, match='NCCL'):
            runtime.validate_sync_batchnorm(True)


class _FusionProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion, _ = _small_fusion()
        self.scale = nn.Parameter(torch.tensor(1.0))
        for block in self.fusion.fusion_blocks.values():
            with torch.no_grad():
                block.channel_gate.fill_(0.1)

    def forward(self, features, jpeg):
        return self.fusion([value * self.scale for value in features], jpeg=jpeg)


def _nccl_sync_worker(rank, folder):
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', init_method=(Path(folder) / 'store').as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=40))
    try:
        runtime = TrainingRuntime(device, rank, 2)
        # A split batch must match ordinary BN on the concatenated batch.
        torch.manual_seed(42)
        full = torch.randn(4, 3, 4, 4, device=device)
        reference = nn.BatchNorm2d(3).to(device)
        model = nn.SyncBatchNorm(3).to(device)
        ddp = runtime.wrap(model)
        actual = ddp(full.chunk(2)[rank])
        expected = reference(full)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(actual, expected.chunk(2)[rank])
        torch.testing.assert_close(model.weight.grad, reference.weight.grad)
        torch.testing.assert_close(model.running_mean, reference.running_mean)
        torch.testing.assert_close(model.running_var, reference.running_var)
        del ddp, model, reference

        probe = _FusionProbe().to(device)
        ddp = runtime.wrap(probe)
        features = [torch.randn(2, c, 64 // s, 64 // s, device=device)
                    for s, c in zip((4, 8, 16, 32), (4, 8, 16, 32), strict=True)]
        sample = dict(bins=torch.randint(0, 21, (64, 64), device=device, dtype=torch.uint8),
                      qtable=torch.ones(8, 8, device=device), geometry=(0, 0, 64, 64, 0, 0, 0))
        # No JPEG on rank 0, unequal nonzero counts, then no JPEG anywhere.
        for counts in ((0, 2), (1, 2), (0, 0)):
            available = counts[rank]
            jpeg = [sample] * available + [{'available': False}] * (2 - available)
            probe.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = ddp(features, jpeg)
                loss = sum(value.float().square().mean() for value in output)
            loss.backward()
            assert torch.isfinite(probe.scale.grad).all()
            for parameter in probe.parameters():
                if parameter.grad is not None:
                    assert torch.isfinite(parameter.grad).all()
            for layer in probe.fusion.fusion_blocks.modules():
                if isinstance(layer, nn.SyncBatchNorm):
                    mean = layer.running_mean.clone()
                    dist.broadcast(mean, src=0)
                    torch.testing.assert_close(layer.running_mean, mean)
        del ddp, probe

        # Exercise actual decoder/fusion backward ordering, not just the probe.
        model = build_model(ModelConfig(), pretrained=False).to(device)
        for block in model.forensic_fusion.fusion_blocks.values():
            with torch.no_grad():
                block.channel_gate.fill_(0.1)
        ddp = runtime.wrap(model)
        jpeg = [{'available': False}] * 2 if rank == 0 else [sample] * 2
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = ddp(torch.randn(2, 3, 64, 64, device=device), jpeg=jpeg)
            loss = sum(output[key].float().square().mean() for key in ('logits', 'aux_logits', 'cls_logits'))
        loss.backward()
        gradient = model.forensic_fusion.branch.artifact.dc_layer0_dil[0].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        (Path(folder) / f'sync-{rank}.txt').write_text('passed', encoding='utf-8')
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2 or not torch.distributed.is_nccl_available(),
                    reason='Requires two CUDA GPUs with NCCL')
def test_two_gpu_syncbn_matches_global_batch_and_handles_empty_amp_rank(tmp_path):
    from tests.test_distributed_training import _run_process_test

    _run_process_test('_nccl_sync_worker', tmp_path, module='tests.test_sync_batchnorm')
    assert all((tmp_path / f'sync-{rank}.txt').read_text(encoding='utf-8') == 'passed' for rank in range(2))
