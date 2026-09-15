"""Device selection, DDP collectives and notebook-safe local process launching."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np  # noqa: F401 -- Load the numerical runtime before torch on Windows.
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.config import ExperimentConfig, PathsConfig


class TrainingRuntime:
    """One process's training device and collective operations."""

    def __init__(self, device, rank=0, world_size=1):
        self.device = torch.device(device)
        self.rank = rank
        self.world_size = world_size

    @property
    def is_main(self):
        return self.rank == 0

    @property
    def distributed(self):
        return self.world_size > 1

    @property
    def cpu_collectives(self):
        # Windows Gloo builds may broadcast CUDA tensors but crash on all_reduce.
        # Keep all Gloo CUDA communication on CPU, including unused parameters.
        return self.distributed and self.device.type == 'cuda' and dist.get_backend() == 'gloo'

    @property
    def communication_device(self):
        return torch.device('cpu') if self.cpu_collectives else self.device

    @staticmethod
    def selected_devices(config):
        device = torch.device(config.device)
        if device.type != 'cuda':
            return ()
        count = torch.cuda.device_count()
        devices = (tuple(range(count)) if device.index is None else (device.index,)) if config.devices == 'auto' else config.devices
        if not devices or any(index >= count for index in devices):
            raise ValueError(f'Requested CUDA devices {devices}; only {count} GPU(s) are visible')
        return tuple(devices)

    @staticmethod
    def backend(config):
        backend = config.distributed_backend
        if backend == 'auto':
            backend = 'nccl' if config.device.startswith('cuda') and dist.is_nccl_available() else 'gloo'
        available = dist.is_nccl_available() if backend == 'nccl' else dist.is_gloo_available()
        if not available:
            raise ValueError(f'Distributed backend {backend} is unavailable in this PyTorch build')
        return backend

    def validate_sync_batchnorm(self, enabled):
        if enabled and self.distributed and (self.device.type != 'cuda' or dist.get_backend() != 'nccl'):
            raise ValueError('Multi-GPU SyncBatchNorm requires CUDA with NCCL; Gloo is unsupported')

    def wrap(self, model):
        self.validate_sync_batchnorm(any(isinstance(module, torch.nn.SyncBatchNorm) for module in model.modules()))
        if not self.distributed:
            return model
        if self.cpu_collectives:
            self._gradient_reducer = CpuGradientReducer(model, self.world_size)
            for parameter in model.parameters():
                self._broadcast(parameter)
            self.synchronize_buffers(model)
            return model
        return DistributedDataParallel(
            model, device_ids=[self.device.index] if self.device.type == 'cuda' else None,
            find_unused_parameters=True,
        )

    @staticmethod
    def unwrap(model):
        return model.module if isinstance(model, DistributedDataParallel) else model

    def main_call(self, function, *args, **kwargs):
        """Run on rank zero; propagate a small result or error to every rank."""
        if not self.distributed:
            return function(*args, **kwargs)
        message = [None]
        if self.is_main:
            try:
                message[0] = (True, function(*args, **kwargs))
            except Exception as exc:
                message[0] = (False, f'{type(exc).__name__}: {exc}')
        dist.broadcast_object_list(message, src=0)
        success, result = message[0]
        if not success:
            raise RuntimeError(f'Rank zero failed: {result}')
        return result

    def gather_objects(self, value):
        if not self.distributed:
            return [value]
        result = [None] * self.world_size
        dist.all_gather_object(result, value)
        return result

    def gather_to_main(self, value):
        """Gather CPU statistics without retaining full copies on every rank."""
        if not self.distributed:
            return [value]
        result = [None] * self.world_size if self.is_main else None
        dist.gather_object(value, object_gather_list=result, dst=0)
        return result

    def sum(self, value):
        dtype = torch.float32 if self.device.type == 'mps' else torch.float64
        tensor = torch.as_tensor(value, dtype=dtype, device=self.communication_device).clone()
        if self.distributed:
            dist.all_reduce(tensor)
        return tensor.to(self.device)

    def reduce_meter(self, meter):
        if not self.distributed:
            return
        keys = sorted({key for keys in self.gather_objects(list(meter.sums)) for key in keys})
        if not keys:
            return
        values = torch.stack([torch.as_tensor(value, device=self.communication_device, dtype=torch.float64)
                              for key in keys for value in (meter.sums.get(key, 0), meter.counts.get(key, 0))])
        dist.all_reduce(values)
        for index, key in enumerate(keys):
            meter.sums[key], meter.counts[key] = values[2 * index:2 * index + 2]

    def synchronize_buffers(self, model):
        if self.distributed:
            for buffer in model.buffers():
                self._broadcast(buffer)

    @torch.no_grad()
    def _broadcast(self, tensor):
        communicated = tensor.detach().to(self.communication_device).contiguous()
        dist.broadcast(communicated, src=0)
        tensor.copy_(communicated)

    def before_forward(self, model):
        if self.cpu_collectives:
            self.synchronize_buffers(model)

    def synchronize_gradients(self):
        if self.cpu_collectives:
            self._gradient_reducer.synchronize()

    @classmethod
    def launch(cls, config, arm, devices):
        """Use importable workers instead of multiprocessing's notebook __main__."""
        from src.training.runs import Run

        backend = cls.backend(config.train)
        config = replace(config, paths=replace(config.paths, data_path=config.paths.data_path.resolve(),
                                               runs_path=config.paths.runs_path.resolve()))
        config.paths.runs_path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.ddp-', dir=config.paths.runs_path) as folder:
            folder = Path(folder).resolve()
            payload = folder / 'config.json'
            payload.write_text(json.dumps({'config': config.to_dict(), 'arm': arm,
                                           'devices': devices, 'backend': backend}), encoding='utf-8')
            processes = []
            try:
                for rank in range(len(devices)):
                    processes.append(subprocess.Popen(
                        [sys.executable, '-m', 'src.training.distributed', '--worker', str(payload),
                         '--rank', str(rank)],
                        cwd=Path(__file__).resolve().parents[2],
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                    ))
                while any(process.poll() is None for process in processes):
                    for rank, process in enumerate(processes):
                        if process.poll() not in (None, 0):
                            raise RuntimeError(f'DDP worker {rank} exited with code {process.returncode}')
                    time.sleep(.1)
                for rank, process in enumerate(processes):
                    if process.returncode:
                        raise RuntimeError(f'DDP worker {rank} exited with code {process.returncode}')
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            return Run.open((folder / 'result.txt').read_text(encoding='utf-8'))


class CpuGradientReducer:
    """Portable CUDA training over Gloo, with bounded CPU gradient buckets.

    Reduce once per optimizer update, after accumulation and before AMP unscale.
    Globally unused parameters keep grad=None (and therefore no weight decay).
    """

    def __init__(self, model, world_size, bucket_bytes=25 * 1024 * 1024):
        self.parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.world_size = world_size
        self.buckets = []
        bucket, size = [], 0
        for index, parameter in enumerate(self.parameters):
            if bucket and (size >= bucket_bytes or parameter.dtype != self.parameters[bucket[0]].dtype):
                self.buckets.append(bucket)
                bucket, size = [], 0
            bucket.append(index)
            size += parameter.numel() * max(4, parameter.element_size())
        if bucket:
            self.buckets.append(bucket)

    @torch.no_grad()
    def synchronize(self):
        used = torch.tensor([parameter.grad is not None for parameter in self.parameters], dtype=torch.int32)
        dist.all_reduce(used)
        for bucket in self.buckets:
            active = [self.parameters[index] for index in bucket if used[index]]
            if not active:
                continue
            dtype = torch.float64 if active[0].dtype == torch.float64 else torch.float32
            flat = torch.cat([parameter.grad.detach().to(device='cpu', dtype=dtype).reshape(-1)
                              if parameter.grad is not None else torch.zeros(parameter.numel(), dtype=dtype)
                              for parameter in active])
            dist.all_reduce(flat)
            flat.div_(self.world_size)
            offset = 0
            for parameter in active:
                gradient = flat[offset:offset + parameter.numel()].view(parameter.shape).to(parameter)
                if parameter.grad is None:
                    parameter.grad = torch.empty_like(parameter)
                parameter.grad.copy_(gradient)
                offset += parameter.numel()


def _worker(payload_path, rank):
    from src.training.engine import ExperimentRunner

    payload_path = Path(payload_path)
    payload = json.loads(payload_path.read_text(encoding='utf-8'))
    config = ExperimentConfig.from_dict(payload['config'])
    paths = payload['config']['paths']
    config = replace(config, paths=PathsConfig(Path(paths['data_path']), Path(paths['runs_path'])))
    device = f"cuda:{payload['devices'][rank]}" if config.train.device.startswith('cuda') else config.train.device
    if device.startswith('cuda'):
        torch.cuda.set_device(device)
    runtime = TrainingRuntime(device, rank, len(payload['devices']))
    dist.init_process_group(payload['backend'],
                            init_method=(payload_path.parent / 'store').as_uri(),
                            rank=rank, world_size=runtime.world_size, timeout=timedelta(minutes=60))
    try:
        run = ExperimentRunner(config, arm=payload['arm'], runtime=runtime).run()
        if runtime.is_main:
            (payload_path.parent / 'result.txt').write_text(str(run.dir.resolve()), encoding='utf-8')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    _worker(args.worker, args.rank)
