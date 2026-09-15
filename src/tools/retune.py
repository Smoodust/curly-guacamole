"""Re-select the operating point of a finished run from its saved histograms.

`oof/val.npz` holds a per-frame 256-bin probability histogram for every
development frame, so any grid of mask threshold, classifier threshold,
minimum area and area cap can be scored without the model, the images or a
GPU. That is the whole point of the accumulator: a run that was tuned over a
narrower grid can be re-tuned later for the cost of a few seconds of numpy.

Deliberately torch-free — it runs anywhere the run directory does.

    python -m src.tools.retune --run runs/<name>

Reports the re-selected point against the one in `summary.json`. The command
prints, and changes nothing, unless `--write` is given.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

# Import the metric module by path. src.training.metric is itself torch-free,
# but importing it as a package member pulls src/__init__ and its siblings,
# which are not.
_spec = importlib.util.spec_from_file_location(
    'aic_metric', Path(__file__).resolve().parents[1] / 'training' / 'metric.py')
_metric = importlib.util.module_from_spec(_spec)
# @dataclass resolves annotations through sys.modules[cls.__module__]; register
# before executing or every dataclass in the module raises.
sys.modules[_spec.name] = _metric
_spec.loader.exec_module(_metric)

AICAccumulator = _metric.AICAccumulator
DEFAULT_MASK_GRID = _metric.DEFAULT_MASK_GRID
DEFAULT_CAP_GRID = _metric.DEFAULT_CAP_GRID

# A refinement of DEFAULT_CLS_GRID around the region where the gate actually
# trades Dice against false alarms. A cap makes a stricter gate affordable, so
# the best classifier threshold moves up and the coarse grid can miss it.
CLS_GRID = (0.0, 0.5, 0.6, 0.65, 0.7, 0.74, 0.78, 0.8, 0.82, 0.84, 0.86, 0.9, 0.95)

# min_area is fixed at zero on purpose. A frame counts as a false alarm at 1% of
# its area, so any min_area <= 0.01 can only delete predicted pixels in frames
# that were already below the alarm threshold: it cannot improve FPR and can
# only cost Dice. Above 0.01 it starts deleting true positives instead.
MIN_AREAS = (0.0,)


def _grid(text, fallback):
    return fallback if not text else tuple(float(x) for x in text.replace(',', ' ').split())


def _load(run_dir: Path, name: str) -> AICAccumulator:
    for candidate in (run_dir / 'oof' / f'{name}.npz', run_dir / name / 'predictions.npz'):
        if candidate.exists():
            print(f'histograms: {candidate}')
            return AICAccumulator.load(candidate)
    raise FileNotFoundError(
        f'no saved histograms for {name!r} under {run_dir}; expected oof/{name}.npz '
        f'or {name}/predictions.npz')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, help='run directory')
    parser.add_argument('--split', default='val',
                        help="oof/<split>.npz, or a directory like 'holdout' holding predictions.npz")
    parser.add_argument('--mask-thresholds', default='', help='space or comma separated; default is the full grid')
    parser.add_argument('--cls-thresholds', default='')
    parser.add_argument('--min-areas', default='')
    parser.add_argument('--area-caps', default='', help='0 disables the cap; see DEFAULT_CAP_GRID')
    parser.add_argument('--top', type=int, default=8, help='how many operating points to list')
    parser.add_argument('--write', action='store_true',
                        help="patch summary.json's best/best_aic with the re-selected point")
    args = parser.parse_args()

    run_dir = Path(args.run)
    acc = _load(run_dir, args.split)
    masks = list(_grid(args.mask_thresholds, DEFAULT_MASK_GRID))
    cls = list(_grid(args.cls_thresholds, CLS_GRID))
    areas = list(_grid(args.min_areas, MIN_AREAS))
    caps = list(_grid(args.area_caps, DEFAULT_CAP_GRID))
    print(f'frames={len(acc)} bins={acc.n_bins} small_mask_weight={acc.small_mask_weight}')
    print(f'grid: {len(masks)} x {len(cls)} x {len(areas)} x {len(caps)} '
          f'= {len(masks) * len(cls) * len(areas) * len(caps)} points')

    ranked = acc.sweep(masks, cls, areas, caps)
    best = ranked[0]
    # The same grid with the cap disabled: the honest baseline for the cap's
    # contribution, since a refined classifier grid alone can also gain.
    uncapped = acc.best(masks, cls, areas, [0.0])

    summary_path = run_dir / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    shipped = summary.get('best')

    print('\ntop operating points')
    for result in ranked[:max(args.top, 1)]:
        print(f'  {result}')
    if shipped:
        current = acc.evaluate(float(shipped['mask_threshold']), float(shipped['cls_threshold']),
                               float(shipped['min_area']), float(shipped.get('area_cap', 0.0)))
        print(f'\nsummary.json best   {current}')
        if abs(current.aic - float(shipped.get('aic', current.aic))) > 1e-9:
            print(f'  note: recomputed AIC differs from the stored {shipped.get("aic")!r}; '
                  'the histograms and the summary may come from different runs')
    print(f'\nno cap              AIC={uncapped.aic:.6f}  cls={uncapped.cls_threshold}')
    print(f'with cap            AIC={best.aic:.6f}  cls={best.cls_threshold} cap={best.area_cap}')
    print(f'cap contributes     {best.aic - uncapped.aic:+.5f}')
    if shipped:
        print(f'total over summary  {best.aic - current.aic:+.5f}')

    if not args.write:
        print('\nnothing written. To use this point directly in a submission:\n'
              f'  python -m src.inference {args.run} <output_dir> '
              f'--mask-threshold {best.mask_threshold} --cls-threshold {best.cls_threshold} '
              f'--min-area {best.min_area} --area-cap {best.area_cap}')
        return

    summary['best'] = best.as_dict()
    summary['best_aic'] = best.aic
    summary['retuned'] = dict(split=args.split, previous=shipped,
                              grid=dict(mask=masks, cls=cls, min_area=areas, area_cap=caps))
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nwrote {summary_path}')
    print('  src.inference now picks this point up with no flags.')
    print('  src.eval (holdout) will refuse until the checkpoint is re-stamped: it compares '
          'summary best against the operating_point saved inside best.pt, on purpose. '
          'Pass the point explicitly to src.inference instead of writing, if that check matters.')


if __name__ == '__main__':
    main()
