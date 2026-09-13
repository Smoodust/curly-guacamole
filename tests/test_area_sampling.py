from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import TrainConfig
from src.training.builders import build_sampler


def dataset(areas):
    areas = np.asarray(areas)
    return SimpleNamespace(df=pd.DataFrame({'mask_area': areas}), is_negative=areas == 0)


def test_uniform_area_gives_equal_mass_to_unequal_categories():
    ds = dataset([0, 0, .001, .009, .01, .029, .03, .079, .08, .199, .20, .5, 1.])
    sampler = build_sampler(TrainConfig(sampling_strategy='uniform_mask_area'), ds)
    probabilities = sampler.weights.numpy() / sampler.weights.sum().item()
    for indices in ([0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11, 12]):
        assert probabilities[indices].sum() == pytest.approx(1 / 6)
    assert sampler.replacement
    assert len(sampler) == 24000


def test_uniform_area_rejects_missing_category():
    with pytest.raises(ValueError, match='categories'):
        build_sampler(TrainConfig(sampling_strategy='uniform_mask_area'), dataset([0, .02]))


@pytest.mark.parametrize('area', [float('nan'), -.1, 1.1])
def test_uniform_area_rejects_invalid_area(area):
    with pytest.raises(ValueError, match='mask_area'):
        build_sampler(TrainConfig(sampling_strategy='uniform_mask_area'), dataset([0, area]))


def test_default_sampling_keeps_negative_fraction():
    sampler = build_sampler(TrainConfig(), dataset([0, 0, .01, .5, .9]))
    assert (sampler.weights[:2].sum() / sampler.weights.sum()).item() == pytest.approx(.25)
