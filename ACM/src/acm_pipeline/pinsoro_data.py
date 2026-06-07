"""Fixed-window datasets for PinSoRo TCN classification."""

from __future__ import annotations

import csv
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.acm_pipeline.pinsoro import ROLE_ORDER


@dataclass(frozen=True)
class PinSoRoWindow:
    dataset: str
    domain: str
    source_split: str
    model_split: str
    session_id: str
    feature_set: str
    start_frame: int
    end_frame: int
    session_aligned_len: int
    roles: tuple[str, ...]
    supervised: tuple[bool, ...]
    tensor_paths: tuple[Path, ...]
    n_features_per_role: int


def _yes(value: str) -> bool:
    return value.strip().lower() == "yes"


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def read_pinsoro_window_manifest(
    manifest_path: Path,
    project_root: Path,
    split: str | None = None,
) -> list[PinSoRoWindow]:
    """Read either an individual or dyadic PinSoRo window manifest."""

    windows: list[PinSoRoWindow] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row["model_split"] != split:
                continue
            common = {
                "dataset": row["dataset"],
                "domain": row["domain"],
                "source_split": row["source_split"],
                "model_split": row["model_split"],
                "session_id": row["session_id"],
                "feature_set": row["feature_set"],
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "session_aligned_len": int(row["session_aligned_len"]),
            }
            if "role" in row:
                windows.append(
                    PinSoRoWindow(
                        **common,
                        roles=(row["role"],),
                        supervised=(_yes(row["supervised"]),),
                        tensor_paths=(_resolve(project_root, row["tensor_relative_path"]),),
                        n_features_per_role=int(row["n_features"]),
                    )
                )
            else:
                windows.append(
                    PinSoRoWindow(
                        **common,
                        roles=ROLE_ORDER,
                        supervised=(_yes(row["purple_supervised"]), _yes(row["yellow_supervised"])),
                        tensor_paths=(
                            _resolve(project_root, row["purple_tensor_relative_path"]),
                            _resolve(project_root, row["yellow_tensor_relative_path"]),
                        ),
                        n_features_per_role=int(row["n_features_per_role"]),
                    )
                )
    return windows


class PinSoRoWindowDataset(Dataset):
    """Load fixed windows while keeping a bounded tensor cache per worker."""

    def __init__(self, windows: list[PinSoRoWindow], max_cached_tensors: int = 2) -> None:
        if max_cached_tensors < 0:
            raise ValueError("max_cached_tensors must be non-negative.")
        self.windows = windows
        self.max_cached_tensors = max_cached_tensors
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.windows)

    def _load(self, path: Path) -> dict[str, np.ndarray]:
        cached = self._cache.pop(path, None)
        if cached is not None:
            self._cache[path] = cached
            return cached
        with np.load(path) as data:
            loaded = {
                key: np.asarray(data[key])
                for key in ("x", "task_y", "task_mask", "social_y", "social_mask")
            }
        if self.max_cached_tensors > 0:
            self._cache[path] = loaded
            while len(self._cache) > self.max_cached_tensors:
                self._cache.popitem(last=False)
        return loaded

    def load_full_role(self, window: PinSoRoWindow, role_idx: int) -> dict[str, np.ndarray]:
        return self._load(window.tensor_paths[role_idx])

    def __getitem__(self, idx: int) -> dict[str, object]:
        window = self.windows[idx]
        s, e = window.start_frame, window.end_frame
        role_data = [self._load(path) for path in window.tensor_paths]

        x = np.stack([data["x"][s:e].T.copy() for data in role_data], axis=0)
        task_y = np.stack([data["task_y"][s:e] for data in role_data], axis=0)
        social_y = np.stack([data["social_y"][s:e] for data in role_data], axis=0)
        task_mask = np.stack([data["task_mask"][s:e] for data in role_data], axis=0)
        social_mask = np.stack([data["social_mask"][s:e] for data in role_data], axis=0)
        supervised = np.asarray(window.supervised, dtype=np.float32)[:, None]
        task_mask = task_mask * supervised
        social_mask = social_mask * supervised

        return {
            "x": torch.from_numpy(x),
            "task_y": torch.from_numpy(task_y.copy()),
            "social_y": torch.from_numpy(social_y.copy()),
            "task_mask": torch.from_numpy(task_mask.copy()),
            "social_mask": torch.from_numpy(social_mask.copy()),
            "window_index": idx,
        }


class SessionBatchSampler(Sampler[list[int]]):
    """Shuffle session groups while retaining cache-friendly local windows."""

    def __init__(self, windows: list[PinSoRoWindow], batch_size: int, seed: int) -> None:
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        grouped: OrderedDict[tuple[Path, ...], list[int]] = OrderedDict()
        for idx, window in enumerate(windows):
            grouped.setdefault(window.tensor_paths, []).append(idx)
        self.groups = list(grouped.values())

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        groups = list(self.groups)
        rng.shuffle(groups)
        for group in groups:
            local = list(group)
            rng.shuffle(local)
            for start in range(0, len(local), self.batch_size):
                yield local[start : start + self.batch_size]

    def __len__(self) -> int:
        return sum((len(group) + self.batch_size - 1) // self.batch_size for group in self.groups)


def pinsoro_window_collate(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([item[key] for item in batch])
        for key in ("x", "task_y", "social_y", "task_mask", "social_mask")
    } | {
        "window_indices": torch.as_tensor([item["window_index"] for item in batch], dtype=torch.long)
    }
