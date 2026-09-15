"""Public pipeline modules remain importable without starting an experiment."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "src.config",
        "src.data.dataset",
        "src.data.sample_io",
        "src.data.preprocess",
        "src.forensic.jpeg",
        "src.forensic.jpeg_input",
        "src.modules.segmenter",
        "src.training.builders",
        "src.training.engine",
        "src.training.validation",
        "src.training.runs",
        "src.inference.predict",
        "src.inference.submission",
    ],
)
def test_public_module_import(module):
    importlib.import_module(module)
