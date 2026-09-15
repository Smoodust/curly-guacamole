"""JPEG metadata helpers."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def luma_qtable(
    source: str | Path | bytes | bytearray | memoryview,
) -> np.ndarray | None:
    """Read JPEG luminance quantization table as an 8x8 array.

    Pillow already returns natural frequency order; reshape it without permutation.
    Returns None for non-JPEG inputs or when the table cannot be read.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        source = io.BytesIO(bytes(source))

    try:
        with Image.open(source) as image:
            tables = getattr(image, "quantization", None)

        if not tables:
            return None

        table = np.asarray(tables[0], dtype=np.float32)
        return table.reshape(8, 8)

    except Exception:
        return None
