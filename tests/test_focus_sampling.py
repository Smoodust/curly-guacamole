from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import TrainConfig
from src.training.builders import build_sampler


def test_focus_coverage_and_replay(tmp_path):
    path = tmp_path / 'focus.parquet'
    pd.DataFrame({'chng_img_path': ['a', 'b', 'c']}).to_parquet(path)
    ds = SimpleNamespace(df=pd.DataFrame({'chng_img_path': list('abcdef')}),
                         is_negative=np.array([False] * 4 + [True] * 2))
    cfg = TrainConfig(sampling_strategy='focus_coverage', focus_manifest=str(path),
                      epoch_size=8, negative_fraction=.25)
    sampler = build_sampler(cfg, ds)
    draws = []
    for seed in (42, 43, 42):
        sampler.generator = torch.Generator().manual_seed(seed)
        indices = list(sampler)
        assert len(indices) == 8
        assert set(range(3)) <= set(indices)
        assert sum(i < 3 for i in indices) == 4
        assert sum(i == 3 for i in indices) == 2
        assert sum(i >= 4 for i in indices) == 2
        draws.append(indices)
    assert draws[0] == draws[2]
    assert draws[0] != draws[1]
    pd.DataFrame({'chng_img_path': ['unknown']}).to_parquet(path)
    with pytest.raises(ValueError, match='train'):
        build_sampler(cfg, ds)


def test_focus_budget_must_cover_every_target(tmp_path):
    path = tmp_path / 'focus.parquet'
    pd.DataFrame({'chng_img_path': ['a', 'b', 'c']}).to_parquet(path)
    ds = SimpleNamespace(df=pd.DataFrame({'chng_img_path': list('abcde')}),
                         is_negative=np.array([False] * 4 + [True]))
    cfg = TrainConfig(sampling_strategy='focus_coverage', focus_manifest=str(path), epoch_size=4)
    with pytest.raises(ValueError, match='cover'):
        build_sampler(cfg, ds)
