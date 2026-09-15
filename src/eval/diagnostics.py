"""Per-image and cohort metrics at a frozen operating point; never tune here."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.training.metric import harmonic_aic, operating_bins


class EvaluationReport:
    @staticmethod
    def add_jpeg_metadata(rows, root, workers=4):
        rows = rows.copy()

        def probe(path):
            with Image.open(Path(root) / path) as image:
                q = getattr(image, 'quantization', {}).get(0)
            return ('unit' if min(q) == max(q) == 1 else 'nonunit') if q else 'unknown'

        missing = rows.q_kind.isna() if 'q_kind' in rows else pd.Series(True, index=rows.index)
        if 'q_kind' not in rows:
            rows['q_kind'] = 'unknown'
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            rows.loc[missing, 'q_kind'] = list(pool.map(probe, rows.loc[missing, 'chng_img_path']))
        return rows

    def __init__(self, accumulator, rows, thresholds):
        if len(accumulator) != len(rows):
            raise ValueError('Metric rows do not align with accumulator')
        if thresholds.mask_threshold >= 1 or thresholds.mask_threshold * accumulator.n_bins != int(thresholds.mask_threshold * accumulator.n_bins):
            raise ValueError('Frozen threshold must match histogram boundary')
        self.accumulator, self.rows, self.thresholds = accumulator, rows.reset_index(drop=True), thresholds

    def per_image(self):
        p, i, g, n, c = self.accumulator.tables()
        k = min(int(self.thresholds.mask_threshold * self.accumulator.n_bins), self.accumulator.n_bins - 1)
        area_cap = float(getattr(self.thresholds, 'area_cap', 0.0))
        # Same rule as the sweep and as inference. A private copy here would make
        # the per-image report describe a different operating point than the one
        # the checkpoint was selected under, which is exactly the drift the
        # shared helper exists to prevent.
        rows_index = np.arange(p.shape[0])
        bins, blank = operating_bins(p, n, c, k, self.thresholds.cls_threshold,
                                     self.thresholds.min_area, area_cap)
        # "no gate" keeps the area rule but ignores the classifier, so the pair
        # stays comparable however the gate is configured.
        _, blank_area = operating_bins(p, n, c, k, 0.0, self.thresholds.min_area, 0.0)
        keep, keep_area = ~blank, ~blank_area
        p_no_gate, i_no_gate = p[:, k], i[:, k]
        p, i = np.where(keep, p[rows_index, bins], 0.0), np.where(keep, i[rows_index, bins], 0.0)
        area, area_no_gate = p / n, p_no_gate / n
        frame = self.rows.copy()
        if 'target_kind' not in frame:
            frame['target_kind'] = 'provided'
        if 'q_kind' not in frame:
            frame['q_kind'] = 'unknown'
        frame['is_positive'] = g > 0
        frame['gt_fraction'] = g / n
        frame['cls_probability'] = c
        frame['pred_fraction'] = area
        frame['dice'] = np.where(g > 0, 2 * i / (p + g + 1e-6), np.nan)
        frame['false_positive'] = (g == 0) & (area >= .01)
        frame['dice_no_gate'] = np.where(
            g > 0, 2 * i_no_gate * keep_area / (p_no_gate * keep_area + g + 1e-6), np.nan)
        frame['false_positive_no_gate'] = (g == 0) & (area_no_gate >= .01) & keep_area
        frame['area_bin'] = pd.cut(frame.gt_fraction, [-1, 0, .01, .05, .15, 1.],
                                    labels=['negative', '(0,1%]', '(1,5%]', '(5,15%]', '(15,100%]'])
        return frame

    @staticmethod
    def aggregate(frame):
        pos = frame.is_positive
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        dice = float(frame.loc[pos, 'dice'].mean()) if n_pos else None
        fp = int(frame.loc[~pos, 'false_positive'].sum())
        fpr = fp / n_neg if n_neg else None
        return dict(n_pos=n_pos, n_neg=n_neg, dice_pos=dice, false_positives=fp, fpr_neg=fpr,
                    aic=harmonic_aic(dice, fpr) if n_pos and n_neg else None,
                    dice_no_gate=float(frame.loc[pos, 'dice_no_gate'].mean()) if n_pos else None,
                    fpr_no_gate=float(frame.loc[~pos, 'false_positive_no_gate'].mean()) if n_neg else None)

    def summary(self):
        frame = self.per_image()
        summary = {name: self.aggregate(subset) for name, subset in (
            ('combined', frame), ('provided', frame.loc[frame.target_kind != 'original_zero']),
            ('originals', frame.loc[frame.target_kind == 'original_zero']))}
        if self.accumulator.small_mask_weight != 1.0 and frame.is_positive.any() and (~frame.is_positive).any():
            result = self.accumulator.evaluate(self.thresholds.mask_threshold, self.thresholds.cls_threshold,
                                               self.thresholds.min_area, area_cap=float(
                                                   getattr(self.thresholds, 'area_cap', 0.0)))
            summary['selection'] = result.as_dict()
        return summary

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        frame = self.per_image()
        frame.to_parquet(directory / 'per_image.parquet', index=False)
        (directory / 'metrics.json').write_text(json.dumps(self.summary(), indent=2, allow_nan=False), encoding='utf-8')
        slices = []
        for column in ('domain', 'q_kind', 'area_bin'):
            for (kind, value), subset in frame.groupby(['target_kind', column], observed=True, dropna=False):
                slices.append(dict(target_kind=kind, slice=column, value=str(value), **self.aggregate(subset)))
        pd.DataFrame(slices).to_csv(directory / 'slices.csv', index=False)
