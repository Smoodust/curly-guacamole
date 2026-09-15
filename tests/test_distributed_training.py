from dataclasses import replace

import numpy as np  # noqa: F401 -- Load the numerical runtime before torch on Windows.
import pytest
import torch

from src.config import TrainConfig


def test_devices_config():
    assert TrainConfig().devices == 'auto'
    assert TrainConfig(devices=[0, 2]).devices == (0, 2)
    for devices in ([], [0, 0], [-1], [True], '0,1', 2):
        with pytest.raises(ValueError, match='devices'):
            TrainConfig(devices=devices)


def test_device_selection(monkeypatch):
    from src.training.distributed import TrainingRuntime

    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 3)
    assert TrainingRuntime.selected_devices(TrainConfig()) == (0, 1, 2)
    assert TrainingRuntime.selected_devices(TrainConfig(device='cuda:1')) == (1,)
    assert TrainingRuntime.selected_devices(TrainConfig(devices=[2, 0])) == (2, 0)
    assert TrainingRuntime.selected_devices(TrainConfig(device='cpu')) == ()
    with pytest.raises(ValueError, match='visible'):
        TrainingRuntime.selected_devices(TrainConfig(devices=[3]))


@pytest.mark.parametrize('size,batch_size,world_size', [(25, 4, 3), (9, 2, 2), (5, 2, 4), (24, 4, 2)])
def test_full_train_distributed_batches_cover_exactly_once(size, batch_size, world_size):
    from src.training.sampling import DistributedBatchSampler, FinalFullTrainSampler

    outputs = []
    for rank in range(world_size):
        base = FinalFullTrainSampler(torch.utils.data.RandomSampler(range(size)), size, 0)
        batches = DistributedBatchSampler(base, batch_size, rank, world_size, drop_last=False)
        batches.set_epoch(0, seed=42)
        outputs.append(list(batches))
    assert len({len(batches) for batches in outputs}) == 1
    assert all(0 < len(batch) <= batch_size for batches in outputs for batch in batches)
    assert sorted(i for batches in outputs for batch in batches for i in batch) == list(range(size))


def test_distributed_weighted_sampling_matches_global_draws():
    from src.training.sampling import DistributedBatchSampler

    weights = torch.tensor([1., 4., 2.])
    expected = list(torch.utils.data.WeightedRandomSampler(weights, 24, True,
                    generator=torch.Generator().manual_seed(43)))
    for rank in range(2):
        base = torch.utils.data.WeightedRandomSampler(weights, 24, True)
        batches = DistributedBatchSampler(base, 3, rank, 2, drop_last=True)
        batches.set_epoch(1, seed=42)
        assert [i for batch in batches for i in batch] == expected[rank::2]
        batches.set_epoch(1, seed=42)
        assert [i for batch in batches for i in batch] == expected[rank::2]


class TinyDistributedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))
        self.unused = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, image, fmap=None):
        logits = self.weight * image[:, :1]
        return {'logits': logits, 'cls_logits': logits.mean((2, 3))}


def _distributed_epoch_worker(rank, folder, device='cpu'):
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist

    from src.config import ExperimentConfig, PathsConfig
    from src.training.builders import build_amp, build_ema
    from src.training.distributed import TrainingRuntime
    from src.training.engine import train_one_epoch

    folder = Path(folder)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(folder / 'store').as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=40))
    try:
        runtime = TrainingRuntime(device, rank, 2)
        config = ExperimentConfig(run_name='test', paths=PathsConfig(folder, folder),
                                  train=TrainConfig(device=device, workers=0, amp='off',
                                                    grad_accum_steps=2, grad_clip=0))
        model = TinyDistributedModel().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        ema = build_ema(config.train, model)
        amp = build_amp(config.train)
        batches = [{'image': torch.full((rank + 1, 3, 2, 2), float(rank + 1)),
                    'mask': torch.ones(rank + 1, 1, 2, 2),
                    'label': torch.ones(rank + 1, 1)} for _ in range(3)]
        result = train_one_epoch(model=runtime.wrap(model), loader=batches,
                                 optimizer=optimizer,
                                 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.),
                                 scaler=amp.scaler(), ema=ema, amp=amp, config=config,
                                 device=torch.device(device), runtime=runtime)
        marker = folder / 'main.txt'
        runtime.main_call(lambda: marker.write_text('written once', encoding='utf-8'))
        with pytest.raises(RuntimeError, match='intentional'):
            runtime.main_call(lambda: (_ for _ in ()).throw(ValueError('intentional')))
        if runtime.cpu_collectives:
            # A branch used on only one rank still needs a gradient on both.
            model.unused.grad = torch.tensor(2., device=device) if rank == 0 else None
            runtime.synchronize_gradients()
            assert model.unused.grad.item() == pytest.approx(1.)
            assert model.weight.grad is None  # Globally unused for this reduction.
        torch.save({'weight': model.weight.detach().cpu(), 'ema': ema.module.weight.detach().cpu(),
                    'seen': result.seen, 'loss': result.loss, 'keys': list(model.state_dict())},
                   folder / f'{rank}.pt')
    finally:
        dist.destroy_process_group()


def test_two_process_training_synchronizes_gradients_ema_and_metrics(tmp_path):
    _run_process_test('_distributed_epoch_worker', tmp_path)
    first, second = [torch.load(tmp_path / f'{rank}.pt', weights_only=True) for rank in range(2)]
    assert first == second
    assert first['seen'] == 9
    assert first['weight'].item() != pytest.approx(.2)
    assert first['keys'] == ['weight', 'unused']
    assert (tmp_path / 'main.txt').read_text(encoding='utf-8') == 'written once'
    _assert_reference_weight(first['weight'])


def _assert_reference_weight(weight):

    # Unequal local batch sizes and the final partial accumulation must agree
    # with two updates on their concatenated global batches.
    from src.losses import SegmentationLoss

    reference = TinyDistributedModel()
    optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
    for _ in range(2):
        batch = {'image': torch.tensor([1., 2., 2.]).view(3, 1, 1, 1).expand(3, 3, 2, 2),
                 'mask': torch.ones(3, 1, 2, 2), 'label': torch.ones(3, 1)}
        SegmentationLoss()(reference(batch['image']), batch).total.backward()
        optimizer.step()
        optimizer.zero_grad()
    assert weight.item() == pytest.approx(reference.weight.item(), abs=1e-6)


def _run_process_test(worker, folder, module='tests.test_distributed_training'):
    import subprocess
    import sys
    from pathlib import Path

    processes = []
    logs = [Path(folder) / f'worker-{rank}.log' for rank in range(2)]
    for rank in range(2):
        with logs[rank].open('w', encoding='utf-8') as output:
            processes.append(subprocess.Popen([sys.executable, '-c',
                 f'import numpy; from {module} import {worker}; '
                 f'import sys; {worker}(int(sys.argv[1]), sys.argv[2])',
                 str(rank), str(folder)],
                 stdout=output, stderr=subprocess.STDOUT,
                 creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0))
    try:
        codes = [process.wait(timeout=55) for process in processes]
        assert codes == [0, 0], '\n'.join(path.read_text(encoding='utf-8', errors='replace') for path in logs)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()


def _distributed_runner_worker(rank, folder):
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist

    import src.training.engine as engine
    from src.training.distributed import TrainingRuntime
    from tests.test_epoch_completion import configure_tiny_run

    folder = Path(folder)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(folder / 'store').as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=40))
    try:
        with pytest.MonkeyPatch.context() as patch:
            config = configure_tiny_run(folder, patch)
            config = replace(config, train=replace(config.train, resume=True, epochs=2, full_pass_epochs=2),
                             augmentation=replace(config.augmentation, final_full_frame_epochs=2))
            runtime = TrainingRuntime('cpu', rank, 2)
            validation = engine.validate
            calls = []

            def failing_validation(*args, **kwargs):
                calls.append(rank)
                raise RuntimeError('intentional validation interruption')

            patch.setattr(engine, 'validate', failing_validation)
            with pytest.raises(RuntimeError, match='intentional validation interruption'):
                engine.ExperimentRunner(config, runtime=runtime).run()
            checkpoint = torch.load(folder / config.run_name / 'ckpt/last.pt', weights_only=True)
            assert checkpoint['samples'] == 2
            assert checkpoint['validation_complete'] is False
            assert not any(key.startswith('module.') for key in checkpoint['model'])
            patch.setattr(engine, 'validate', validation)
            run = engine.ExperimentRunner(config, runtime=runtime).run()
            assert calls == [rank]
            final = run.load_state('last.pt', map_location='cpu')
            assert final['samples'] == 4
            assert final['validation_complete'] is True
            assert run.summary['training_complete'] is True
            assert len(run.jsonl_path.read_text(encoding='utf-8').splitlines()) == 2
            with pytest.raises(ValueError, match='world_size'):
                engine.ExperimentRunner(config)._check_resume_protocol()
    finally:
        dist.destroy_process_group()


def test_distributed_runner_checkpoints_validation_and_resume(tmp_path):
    _run_process_test('_distributed_runner_worker', tmp_path)


def _launch_test_worker(payload_path, rank):
    from pathlib import Path

    from src.training.distributed import _worker
    from tests.test_epoch_completion import configure_tiny_run

    with pytest.MonkeyPatch.context() as patch:
        configure_tiny_run(Path(payload_path).parent, patch)
        _worker(payload_path, rank)


def test_automatic_launcher_preserves_paths_and_returns_run(tmp_path, monkeypatch):
    import subprocess
    import sys

    from src.training.distributed import TrainingRuntime
    from tests.test_epoch_completion import configure_tiny_run

    config = configure_tiny_run(tmp_path, monkeypatch)
    popen = subprocess.Popen

    def test_worker(command, **kwargs):
        payload = command[command.index('--worker') + 1]
        rank = command[command.index('--rank') + 1]
        command = [sys.executable, '-c',
                   'import numpy; from tests.test_distributed_training import _launch_test_worker; '
                   'import sys; _launch_test_worker(sys.argv[1], int(sys.argv[2]))', payload, rank]
        return popen(command, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', test_worker)
    from src.training.engine import ExperimentRunner

    monkeypatch.setattr(TrainingRuntime, 'selected_devices', lambda config: (0, 1))
    run = ExperimentRunner(config).run()
    assert run.dir == tmp_path / config.run_name
    assert run.summary['training_complete'] is True
    assert run.load_state()['cfg']['world_size'] == 2
    assert not list(tmp_path.glob('.ddp-*'))




def test_launcher_terminates_sibling_on_worker_failure(tmp_path, monkeypatch):
    import subprocess
    import sys

    from src.config import ExperimentConfig, PathsConfig
    from src.training.distributed import TrainingRuntime

    config = ExperimentConfig(run_name='failure', paths=PathsConfig(tmp_path, tmp_path),
                               train=TrainConfig(device='cpu'))
    popen = subprocess.Popen
    processes = []

    def failing_worker(command, **kwargs):
        rank = int(command[command.index('--rank') + 1])
        program = 'import sys; sys.exit(7)' if rank == 0 else 'import time; time.sleep(30)'
        process = popen([sys.executable, '-c', program], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, 'Popen', failing_worker)
    with pytest.raises(RuntimeError, match='worker 0 exited with code 7'):
        TrainingRuntime.launch(config, 'test', (0, 1))
    assert all(process.poll() is not None for process in processes)
    assert not list(tmp_path.glob('.ddp-*'))


@pytest.mark.parametrize('precision', ['fp16', 'bf16'])
def test_cuda_training_runtime_with_amp(precision):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 unavailable')
    from src.training.builders import build_amp, build_ema
    from src.training.distributed import TrainingRuntime
    from src.training.engine import train_one_epoch
    from tests.test_engine import CountingScheduler, _cpu_config

    config = _cpu_config(grad_accum_steps=2)
    config = replace(config, train=replace(config.train, device='cuda:0', amp=precision))
    model = TinyDistributedModel().cuda()
    ema = build_ema(config.train, model)
    amp = build_amp(config.train)
    batch = {'image': torch.ones(2, 3, 8, 8), 'mask': torch.ones(2, 1, 8, 8),
             'label': torch.ones(2, 1)}
    result = train_one_epoch(model=model, loader=[batch, batch, batch],
                             optimizer=torch.optim.SGD(model.parameters(), lr=.1),
                             scheduler=CountingScheduler(), scaler=amp.scaler(), ema=ema, amp=amp,
                             config=config, device=torch.device('cuda:0'), runtime=TrainingRuntime('cuda:0'))
    assert result.seen == 6 and result.skipped_steps == 0
    assert model.weight.item() > .2


def _cuda_gloo_worker(rank, folder):
    # Two independent processes share the single available GPU for this test.
    # This exercises CUDA computation with CPU Gloo exchange, not multi-card speed.
    _distributed_epoch_worker(rank, folder, 'cuda:0')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_gloo_gradient_exchange(tmp_path):
    _run_process_test('_cuda_gloo_worker', tmp_path)
    first, second = [torch.load(tmp_path / f'{rank}.pt', weights_only=True) for rank in range(2)]
    assert first == second
    assert first['seen'] == 9
    assert first['weight'].item() > .2
    _assert_reference_weight(first['weight'])
