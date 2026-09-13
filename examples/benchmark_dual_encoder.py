"""Synthetic full-pipeline FLOPs and optional CUDA inference benchmark.

Run from project root: python -m examples.benchmark_dual_encoder [--cuda]
No pretrained downloads, competition data, or training. GPU timing excludes I/O.
"""

import argparse
import json
from pathlib import Path

import torch

from src.budget import count_gflops
from src.config import load_experiment_config
from src.training.builders import build_model, configure_memory_format


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--output', default='runs/dual_encoder_setup/benchmark.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    report = {'torch': torch.__version__, 'pretrained': False, 'batch_size': 1,
              'rgb_size': 576, 'gpu': torch.cuda.get_device_name() if args.cuda else None,
              'timing': 'GPU resident full forward; BF16; 3 warmups, 10 repeats; no H2D or I/O',
              'recipes': {}}
    recipes = ['jpeg576_fusion_local_weighted_val_wavelet', 'jpeg576_wavelet_dual_b0', 'jpeg576_wavelet_dual_b1']
    for recipe in recipes:
        cfg = load_experiment_config(f'configs/{recipe}.yaml')
        with torch.device('meta'):
            model = build_model(cfg.model, pretrained=False)
        record = {'parameters': sum(p.numel() for p in model.parameters()), 'gflops': {}}
        for h, w in [(1024, 1024), (769, 1153), (1536, 2048)]:
            record['gflops'][f'{h}x{w}'] = count_gflops(model, 576, native_size=(h, w))
        del model
        if args.cuda:
            model = configure_memory_format(build_model(cfg.model, pretrained=False).cuda()).eval()
            rgb = torch.randn(1, 3, 576, 576, device='cuda').contiguous(memory_format=torch.channels_last)
            native = [torch.randint(256, (3, 1024, 1024), dtype=torch.uint8, device='cuda')]
            jpeg = [dict(bins=torch.randint(21, (1024, 1024), dtype=torch.uint8, device='cuda'),
                         qtable=torch.ones(8, 8, device='cuda'), geometry=(0, 0, 1024, 1024, 0, 0, 0))]
            times = []
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for step in range(13):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    output = model(rgb, native_rgb=native, jpeg=jpeg)
                    end.record()
                    end.synchronize()
                    if step >= 3:
                        times.append(start.elapsed_time(end))
                assert torch.isfinite(output['logits']).all()
            record['cuda_ms_median'] = float(torch.tensor(times).median())
            record['cuda_peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
            del model, rgb, native, jpeg, output
            torch.cuda.empty_cache()
        report['recipes'][recipe] = record
        print(recipe, json.dumps(record), flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
