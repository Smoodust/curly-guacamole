"""The project must import and run on Python 3.10, not just on the 3.11 of a
particular workstation.

The pod and the workstation do not always agree on a minor version, and a
`from enum import StrEnum` at import time fails the whole run before the first
batch — after the dataset is already unpacked and the GPU is already booked.
Both tests below are cheap enough to keep that from happening twice.
"""

import ast
import pathlib

import pytest

from src.data.augmentation.base import AugmentationStage

MINIMUM = (3, 10)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("src", "tests")


def _sources():
    """Every module the project owns, including `global_config.py` at the root."""
    nested = [path for directory in SOURCE_DIRS
              for path in (PROJECT_ROOT / directory).rglob("*.py")]
    return sorted(nested + list(PROJECT_ROOT.glob("*.py")))


@pytest.mark.parametrize("path", _sources(), ids=lambda path: str(path.relative_to(PROJECT_ROOT)))
def test_every_module_parses_on_the_oldest_supported_python(path):
    """Syntax only — `feature_version` cannot see stdlib members like StrEnum."""
    ast.parse(path.read_text(encoding="utf-8"), feature_version=MINIMUM)


def test_the_augmentation_stage_behaves_like_the_string_it_replaces():
    """`str, Enum` stands in for 3.11's StrEnum, so it must format the same way.

    Without the explicit `__str__`/`__format__` this passes on one minor version
    and fails on the next: a plain mixin renders as `AugmentationStage.FINAL`.
    """
    stage = AugmentationStage.FINAL

    assert isinstance(stage, str)
    assert str(stage) == "final"
    assert f"{stage}" == "final"
    assert "%s" % stage == "final"
    assert stage == "final"
    assert AugmentationStage("final") is stage
