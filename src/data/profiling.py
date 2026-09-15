"""Opt-in worker wall-time measurements returned with samples, never printed."""

import time

import numpy as np
import torch


class SampleTimer:
    STAGES = ('read_image', 'read_mask', 'qtable', 'setup', 'jpeg',
              'geometry', 'resize', 'photometric', 'tensorize')

    def __init__(self):
        self.times = dict.fromkeys(self.STAGES, 0.)
        self.last = time.perf_counter()

    def mark(self, stage):
        now = time.perf_counter()
        self.times[stage] += (now - self.last) * 1000
        self.last = now

    def tensor(self):
        return torch.tensor([self.times[key] for key in self.STAGES], dtype=torch.float64)


class WorkerProfileSummary:
    def __init__(self):
        self.rows = []

    def consume(self, batch, *, include=True):
        timing = batch.pop('_worker_profile', None)
        if timing is not None and include:
            self.rows.append(timing.cpu().numpy())
        return batch

    def report(self):
        if not self.rows:
            return {'samples': 0, 'stages': {}}
        values = np.concatenate(self.rows, axis=0)
        total = float(values.sum(axis=1).mean())
        stages = {}
        for index, name in enumerate(SampleTimer.STAGES):
            column = values[:, index]
            stages[name] = {'mean_ms': float(column.mean()),
                            'p50_ms': float(np.median(column)),
                            'p95_ms': float(np.percentile(column, 95)),
                            'share_percent': float(column.mean()) / max(total, 1e-9) * 100}
        return {'samples': len(values), 'mean_sample_ms': total, 'stages': stages,
                'note': 'Per-image worker wall time, including scheduling/preemption. Excludes collation, IPC, pinning, queues and CUDA. Parallel-worker times must not be added to GPU time.'}
