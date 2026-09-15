import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.collation import ValidationCollator
from src.data.data_workspace import DataWorkspace
from src.data.dataset import AIIJCDataset


@pytest.mark.parametrize('workers', [0, 2])
def test_profile_preserves_sample_and_reports_stage_times(tmp_path, workers):
    workspace = DataWorkspace(tmp_path)
    workspace.train_root.mkdir(parents=True)
    cv2.imwrite(str(workspace.train_root / 'a.png'), np.full((64, 96, 3), 127, np.uint8))
    cv2.imwrite(str(workspace.train_root / 'mask.png'), np.zeros((64, 96), np.uint8))
    rows = pd.DataFrame({'chng_img_path': ['a.png'], 'gt_path': ['mask.png']})
    dataset = AIIJCDataset(workspace, rows, True, 32, 42)
    expected = dataset[0]
    dataset.profile_data = True
    actual = dataset[0]
    timing = actual.pop('_worker_profile')
    assert torch.isfinite(timing).all() and (timing >= 0).all() and timing.sum() > 0
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=workers,
                                                   collate_fn=ValidationCollator())))
    assert batch['_worker_profile'].shape == (1, len(timing))
    assert torch.isfinite(batch['_worker_profile']).all()


def test_worker_summary_excludes_warmup_and_strips_metadata():
    from src.data.profiling import SampleTimer, WorkerProfileSummary
    from src.training.profiling import BenchmarkBatches
    summary = WorkerProfileSummary()
    samples = [{'image': torch.zeros(2, 3, 4, 4),
                '_worker_profile': torch.full((2, len(SampleTimer.STAGES)), float(i))}
               for i in (100, 2, 4)]
    batches = list(BenchmarkBatches(samples, 3, worker_profile=summary, warmup=1))
    assert all('_worker_profile' not in batch for batch in batches)
    report = summary.report()
    assert report['samples'] == 4
    assert report['stages']['jpeg']['mean_ms'] == 3
