"""Shared PinSoRo label and manifest conventions."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from src.acm_pipeline.io import read_stream_matrix


LABEL_MAPS = {
    "task_engagement": {
        "goaloriented": 0,
        "aimless": 1,
        "adultseeking": 2,
        "noplay": 3,
    },
    "social_engagement": {
        "solitary": 0,
        "onlooker": 1,
        "parallel": 2,
        "associative": 3,
        "cooperative": 4,
    },
}
ROLE_ORDER = ("purple", "yellow")
LABEL_RATE_HZ = 30.0


def read_class_labels(path: Path, head: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one classification target into integer labels and a validity mask."""

    label_map = LABEL_MAPS[head]
    labels: list[int] = []
    mask: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            label = label_map.get(line.strip().lower(), -1)
            labels.append(label)
            mask.append(float(label >= 0))
    return np.asarray(labels, dtype=np.int64), np.asarray(mask, dtype=np.float32)


def pinsoro_stream_metadata(header_path: Path) -> tuple[float, int, int]:
    """Return rate, dimension, and declared frame count from a PinSoRo header."""

    text = header_path.read_text(encoding="utf-8", errors="ignore")
    rate_match = re.search(r'sr="([^" ]+)"', text)
    dim_match = re.search(r'dim="([^" ]+)"', text)
    counts = [int(value) for value in re.findall(r'num="(\d+)"', text)]
    if rate_match is None or dim_match is None or not counts:
        raise ValueError(f"Incomplete PinSoRo stream header: {header_path}")
    return float(rate_match.group(1)), int(float(dim_match.group(1))), sum(counts)


def read_pinsoro_stream(
    header_path: Path, binary_path: Path
) -> tuple[np.ndarray, float, int, int]:
    """Read a PinSoRo stream while respecting the header-declared frame count."""

    matrix, rate, dim = read_stream_matrix(header_path, binary_path)
    _, _, declared_frames = pinsoro_stream_metadata(header_path)
    if len(matrix) < declared_frames:
        raise ValueError(f"Binary shorter than declared frame count: {binary_path}")
    return matrix[:declared_frames], rate, dim, declared_frames


def resample_class_labels(
    labels: np.ndarray,
    mask: np.ndarray,
    target_len: int,
    source_rate_hz: float = LABEL_RATE_HZ,
    target_rate_hz: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor resample discrete labels and masks onto a target grid."""

    indices = np.rint(
        np.arange(target_len, dtype=np.float64) * source_rate_hz / target_rate_hz
    ).astype(np.int64)
    indices = np.clip(indices, 0, len(labels) - 1)
    return labels[indices], mask[indices]


def cache_path(cache_root: Path, relative_path: str) -> Path:
    return cache_root / relative_path
