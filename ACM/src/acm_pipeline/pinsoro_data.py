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


TENSOR_ARRAY_KEYS = ("x", "task_y", "task_mask", "social_y", "social_mask")


def pinsoro_mmap_cache_dir(
    cache_root: Path, source_path: Path, project_root: Path
) -> Path:
    """Map one project-relative NPZ tensor to its mmap-cache directory."""

    try:
        relative = source_path.resolve().relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Tensor path is outside project root: {source_path}") from exc
    return cache_root / relative.parent / relative.stem


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
    tensor_paths: tuple[tuple[Path, ...], ...]
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
                        tensor_paths=((_resolve(project_root, row["tensor_relative_path"]),),),
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
                            (_resolve(project_root, row["purple_tensor_relative_path"]),),
                            (_resolve(project_root, row["yellow_tensor_relative_path"]),),
                        ),
                        n_features_per_role=int(row["n_features_per_role"]),
                    )
                )
    return windows


def _window_identity(window: PinSoRoWindow) -> tuple[object, ...]:
    return (
        window.dataset,
        window.domain,
        window.source_split,
        window.model_split,
        window.session_id,
        window.start_frame,
        window.end_frame,
        window.session_aligned_len,
        window.roles,
        window.supervised,
    )


def read_pinsoro_window_manifests(
    manifest_paths: list[Path],
    project_root: Path,
    split: str | None = None,
) -> list[PinSoRoWindow]:
    """Read and early-fuse one or more aligned PinSoRo window manifests."""

    if not manifest_paths:
        raise ValueError("At least one PinSoRo window manifest is required.")
    by_manifest = [
        read_pinsoro_window_manifest(path, project_root, split)
        for path in manifest_paths
    ]
    if len(by_manifest) == 1:
        return by_manifest[0]
    feature_sets = [windows[0].feature_set for windows in by_manifest if windows]
    if len(feature_sets) != len(by_manifest):
        raise RuntimeError("Every PinSoRo feature manifest must contain windows.")
    if len(set(feature_sets)) != len(feature_sets):
        raise ValueError(f"Duplicate PinSoRo feature sets requested: {feature_sets}")
    indexed = [
        {_window_identity(window): window for window in windows}
        for windows in by_manifest
    ]
    reference_keys = [_window_identity(window) for window in by_manifest[0]]
    reference_set = set(reference_keys)
    for feature_set, windows in zip(feature_sets[1:], indexed[1:]):
        if set(windows) != reference_set:
            missing = len(reference_set - set(windows))
            extra = len(set(windows) - reference_set)
            raise RuntimeError(
                f"PinSoRo feature {feature_set} is not aligned with the reference "
                f"manifest: missing={missing}, extra={extra}."
            )
    fused = []
    for key in reference_keys:
        parts = [windows[key] for windows in indexed]
        reference = parts[0]
        fused.append(
            PinSoRoWindow(
                dataset=reference.dataset,
                domain=reference.domain,
                source_split=reference.source_split,
                model_split=reference.model_split,
                session_id=reference.session_id,
                feature_set="__".join(feature_sets),
                start_frame=reference.start_frame,
                end_frame=reference.end_frame,
                session_aligned_len=reference.session_aligned_len,
                roles=reference.roles,
                supervised=reference.supervised,
                tensor_paths=tuple(
                    tuple(
                        path
                        for part in parts
                        for path in part.tensor_paths[role_idx]
                    )
                    for role_idx in range(len(reference.roles))
                ),
                n_features_per_role=sum(
                    part.n_features_per_role for part in parts
                ),
            )
        )
    return fused


class PinSoRoWindowDataset(Dataset):
    """Load fixed windows while keeping a bounded tensor cache per worker."""

    def __init__(
        self,
        windows: list[PinSoRoWindow],
        max_cached_tensors: int = 2,
        mmap_cache_root: Path | None = None,
        project_root: Path | None = None,
    ) -> None:
        if max_cached_tensors < 0:
            raise ValueError("max_cached_tensors must be non-negative.")
        if mmap_cache_root is not None and project_root is None:
            raise ValueError("project_root is required when mmap_cache_root is set.")
        self.windows = windows
        paths_per_window = max(
            (sum(len(paths) for paths in window.tensor_paths) for window in windows),
            default=0,
        )
        self.max_cached_tensors = max(max_cached_tensors, paths_per_window)
        self.mmap_cache_root = mmap_cache_root
        self.project_root = project_root
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.windows)

    def _load(self, path: Path) -> dict[str, np.ndarray]:
        cached = self._cache.pop(path, None)
        if cached is not None:
            self._cache[path] = cached
            return cached
        if self.mmap_cache_root is None:
            with np.load(path) as data:
                loaded = {key: np.asarray(data[key]) for key in TENSOR_ARRAY_KEYS}
        else:
            cache_dir = pinsoro_mmap_cache_dir(
                self.mmap_cache_root, path, self.project_root
            )
            marker = cache_dir / ".complete"
            if not marker.is_file():
                raise FileNotFoundError(
                    f"Missing PinSoRo mmap cache for {path}: expected {marker}"
                )
            loaded = {
                key: np.load(cache_dir / f"{key}.npy", mmap_mode="r")
                for key in TENSOR_ARRAY_KEYS
            }
        if self.max_cached_tensors > 0:
            self._cache[path] = loaded
            while len(self._cache) > self.max_cached_tensors:
                self._cache.popitem(last=False)
        return loaded

    def load_full_role(self, window: PinSoRoWindow, role_idx: int) -> dict[str, np.ndarray]:
        return self._load(window.tensor_paths[role_idx][0])

    def __getitem__(self, idx: int) -> dict[str, object]:
        window = self.windows[idx]
        s, e = window.start_frame, window.end_frame
        role_data = [
            [self._load(path) for path in role_paths]
            for role_paths in window.tensor_paths
        ]

        x = np.stack(
            [
                np.concatenate([data["x"][s:e] for data in modalities], axis=1)
                .T.copy()
                for modalities in role_data
            ],
            axis=0,
        )
        label_data = [modalities[0] for modalities in role_data]
        task_y = np.stack([data["task_y"][s:e] for data in label_data], axis=0)
        social_y = np.stack([data["social_y"][s:e] for data in label_data], axis=0)
        task_mask = np.stack([data["task_mask"][s:e] for data in label_data], axis=0)
        social_mask = np.stack([data["social_mask"][s:e] for data in label_data], axis=0)
        supervised = np.asarray(window.supervised, dtype=np.float32)[:, None]
        task_mask = task_mask * supervised
        social_mask = social_mask * supervised

        return {
            "x": x,
            "task_y": task_y,
            "social_y": social_y,
            "task_mask": task_mask,
            "social_mask": social_mask,
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
        return sum(
            (len(group) + self.batch_size - 1) // self.batch_size
            for group in self.groups
        )


def pinsoro_window_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    """Batch NumPy windows without invoking Torch's large CPU thread pool."""

    arrays = {
        key: np.stack([item[key] for item in batch])
        for key in ("x", "task_y", "social_y", "task_mask", "social_mask")
    }
    return {key: torch.from_numpy(value) for key, value in arrays.items()} | {
        "window_indices": torch.as_tensor(
            [item["window_index"] for item in batch], dtype=torch.long
        ),
        "has_supervision": any(
            np.any(arrays[f"{head}_mask"]) for head in ("task", "social")
        ),
    }
