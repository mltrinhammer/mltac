"""Shared PinSoRo label and manifest conventions."""

from __future__ import annotations

from pathlib import Path

import numpy as np


LABEL_MAPS = {
    "task_engagement": {"goaloriented": 0, "aimless": 1, "adultseeking": 2, "noplay": 3},
    "social_engagement": {"solitary": 0, "onlooker": 1, "parallel": 2, "associative": 3, "cooperative": 4},
}
ROLE_ORDER = ("purple", "yellow")


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


def cache_path(cache_root: Path, relative_path: str) -> Path:
    return cache_root / relative_path
