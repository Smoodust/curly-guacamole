import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.data.data_workspace import DataWorkspace
from src.data.utils import read_image
from src.progress import ConsoleProgress

HASH_PREFIX = re.compile(r"^[0-9a-f]{12}_")
DATASET_PREFIX = re.compile(r"^(coco|raise|openimages)_")

METHOD_TOKENS = {
    "powerpaint", "brushnet", "hdpainter", "removeanything",
    "inpaintanything", "inpainted", "lama", "sd2",
}
CUT_TOKENS = METHOD_TOKENS | {"small", "medium", "large"}
GT_THRESHOLD = 128


class MetadataCache:
    """Общий кеш датасета; увеличьте VERSION при изменении расчёта метаданных."""

    VERSION = 1

    def __init__(self, workspace: DataWorkspace):
        self.path = workspace.train_root / ".cache" / f"metadata_v{self.VERSION}.parquet"

    def load(self, source: pd.DataFrame) -> pd.DataFrame | None:
        if not self.path.exists():
            return None
        cached = load_metadata(self.path)
        if not set(source.columns).issubset(cached.columns):
            return None
        if not cached[source.columns].equals(source):
            return None
        ConsoleProgress.info(f"Метаданные загружены из кеша: {self.path}")
        return cached

    def save(self, metadata: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a complete file, including when runs start concurrently.
        with NamedTemporaryFile(dir=self.path.parent, suffix=".parquet", delete=False) as file:
            temporary_path = Path(file.name)
        try:
            save_metadata(metadata, temporary_path)
            temporary_path.replace(self.path)
        finally:
            temporary_path.unlink(missing_ok=True)
        ConsoleProgress.info(f"Метаданные сохранены в кеш: {self.path}")


def build_metadata(
        data_workspace: DataWorkspace,
        workers: int = 12,
        output_path: str | Path | None = None,
        *,
        force_rebuild: bool = False,
) -> pd.DataFrame:
    """Возвращает общие метаданные, пересчитывая их при изменении train.csv.

    После замены изображений/масок по прежним путям используйте force_rebuild.
    Фолды в кеш не входят и строятся отдельно для каждого эксперимента.
    """
    df = data_workspace.train_csv
    cache = MetadataCache(data_workspace)
    cached = None if force_rebuild else cache.load(df)
    if cached is None:
        df = _add_filename_metadata(df)
        df = _add_image_metadata(df, data_workspace.train_root, workers)
        cache.save(df)
    else:
        df = cached

    if output_path is not None:
        save_metadata(df, output_path)

    return df


def save_metadata(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_metadata(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Не найден index: {path}")
    return pd.read_parquet(path)


def _add_filename_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stem"] = df["chng_img_path"].map(_stem)
    df["domain"] = df["stem"].map(_parse_domain)
    df["generator"] = df["stem"].map(_parse_generator)
    df["group_id"] = [
        _parse_group_id(stem, original_path)
        for stem, original_path in zip(
            df["stem"], df["orgl_img_path"], strict=True
        )
    ]
    return df


def _add_image_metadata(
        df: pd.DataFrame,
        dataset_root: str | Path,
        workers: int,
) -> pd.DataFrame:
    tasks = [
        (i, gt_path, image_path, str(dataset_root))
        for i, (gt_path, image_path) in enumerate(
            zip(df["gt_path"], df["chng_img_path"], strict=True)
        )
    ]

    if workers <= 1:
        results = [_probe_sample(task) for task in ConsoleProgress.iterate(tasks, "Метаданные, изображения и маски")]
    else:
        results = [None] * len(df)
        chunksize = max(64, len(tasks) // max(workers * 16, 1) or 1)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in ConsoleProgress.iterate(
                pool.map(_probe_sample, tasks, chunksize=chunksize),
                f"Метаданные, изображения и маски ({len(tasks)} шт., workers={workers})",
            ):
                results[result[0]] = result

    metadata = pd.DataFrame(
        results,
        columns=["row_id", "height", "width", "mask_area", "img_h", "img_w"],
    ).set_index("row_id")

    out = df.copy()
    for column in metadata.columns:
        out[column] = metadata[column].to_numpy()

    out["size_mismatch"] = (
            (out["height"] != out["img_h"]) | (out["width"] != out["img_w"])
    )
    out["is_negative"] = out["mask_area"] == 0.0
    out["broken"] = out["mask_area"] < 0.0
    return out


def _probe_sample(args):
    row_id, gt_path, image_path, dataset_root = args
    root = Path(dataset_root)

    mask = read_image(
        root / str(gt_path).replace("\\", "/"),
        cv2.IMREAD_GRAYSCALE,
    )
    if mask is None:
        return row_id, 0, 0, -1.0, 0, 0

    height, width = mask.shape[:2]
    mask_area = float(np.mean(mask >= GT_THRESHOLD))

    try:
        with Image.open(root / str(image_path).replace("\\", "/")) as image:
            img_w, img_h = image.size
    except Exception:
        img_h = img_w = 0

    return row_id, height, width, mask_area, img_h, img_w


def _stem(path: str) -> str:
    name = HASH_PREFIX.sub("", Path(str(path)).name)
    return name.rsplit(".", 1)[0]


def _parse_domain(stem: str) -> str:
    if stem.startswith("coco_"):
        return "coco"
    if stem.startswith("raise_"):
        return "raise"
    if stem.startswith("openimages_"):
        return "openimages"
    if re.match(r"^D\d{2}_", stem):
        return "vision"
    if re.match(r"^\d+$", stem):
        return "plain"
    if re.match(r"^\d+_", stem):
        return "numid"
    return "other"


def _parse_generator(stem: str) -> str:
    for token in stem.split("_"):
        method = token.split("-")[0]
        if method in METHOD_TOKENS:
            return method
    return "none"


def _parse_group_id(stem: str, original_path) -> str:
    """Общий id для всех манипуляций одного исходного изображения."""
    if isinstance(original_path, str) and original_path:
        return DATASET_PREFIX.sub("", _stem(original_path))

    tokens = stem.split("_")
    for i, token in enumerate(tokens):
        if token.split("-")[0] in CUT_TOKENS:
            tokens = tokens[:i]
            break

    return DATASET_PREFIX.sub("", "_".join(tokens) or stem)
