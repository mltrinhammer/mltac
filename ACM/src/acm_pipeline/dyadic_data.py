from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.acm_pipeline.data import window_starts


ROLE_ORDER = ("novice", "expert")


@dataclass(frozen=True)
class DyadicManifestExample:
    """One session-level dyadic tensor listed in a dyadic manifest."""

    dataset: str
    session_id: str
    model_split: str
    tensor_path: Path
    n_features: int
    n_features_per_role: int
    aligned_len: int

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.session_id}"


@dataclass
class DyadicSessionTensor:
    """Loaded dyadic arrays for one session."""

    x: np.ndarray
    y: np.ndarray
    target_mask: np.ndarray


def read_dyadic_manifest(manifest_path: Path, project_root: Path, split: str | None = None) -> list[DyadicManifestExample]:
    """Read a dyadic manifest into typed session examples."""

    examples: list[DyadicManifestExample] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row["model_split"] != split:
                continue
            tensor_path = Path(row["tensor_relative_path"])
            if not tensor_path.is_absolute():
                tensor_path = project_root / tensor_path
            examples.append(
                DyadicManifestExample(
                    dataset=row["dataset"],
                    session_id=row["session_id"],
                    model_split=row["model_split"],
                    tensor_path=tensor_path,
                    n_features=int(row["n_features"]),
                    n_features_per_role=int(row["n_features_per_role"]),
                    aligned_len=int(row["aligned_len"]),
                )
            )
    return examples


def load_dyadic_session_tensor(example: DyadicManifestExample) -> DyadicSessionTensor:
    """Load one dyadic NPZ and validate the dyadic tensor contract."""

    with np.load(example.tensor_path, allow_pickle=True) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float32)
        target_mask = np.asarray(data["target_mask"], dtype=np.float32)

    # Dyadic training relies on y/mask having one channel per role. Catch shape
    # problems here so model scripts do not silently train on malformed tensors.
    if x.ndim != 2:
        raise ValueError(f"Expected dyadic x [time, features], got {x.shape} in {example.tensor_path}")
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"Expected dyadic y [time, 2], got {y.shape} in {example.tensor_path}")
    if target_mask.ndim != 2 or target_mask.shape[1] != 2:
        raise ValueError(f"Expected dyadic target_mask [time, 2], got {target_mask.shape} in {example.tensor_path}")
    if x.shape[0] != y.shape[0] or y.shape != target_mask.shape:
        raise ValueError(f"Dyadic x/y/mask lengths do not match in {example.tensor_path}")
    return DyadicSessionTensor(x=x, y=y, target_mask=target_mask)


class WindowedDyadicDataset(Dataset):
    """Lazy dyadic sequence windows for TCN/Transformer models."""

    def __init__(
        self,
        examples: list[DyadicManifestExample],
        window_size: int,
        stride: int,
        include_tail: bool = True,
        max_windows: int | None = None,
        seed: int = 13,
    ) -> None:
        self.examples = examples
        self.window_size = window_size
        self.stride = stride
        self._cache: dict[int, DyadicSessionTensor] = {}

        # Windows are indexed within a single dyadic session tensor. Since each
        # tensor already fuses novice/expert per frame, windows cannot cross
        # person boundaries or session boundaries.
        windows: list[tuple[int, int]] = []
        for example_idx, example in enumerate(examples):
            for start in window_starts(example.aligned_len, window_size, stride, include_tail=include_tail):
                windows.append((example_idx, start))

        if max_windows is not None and len(windows) > max_windows:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(windows), size=max_windows, replace=False))
            windows = [windows[int(idx)] for idx in keep]
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def load_session(self, example_idx: int) -> DyadicSessionTensor:
        cached = self._cache.get(example_idx)
        if cached is not None:
            return cached
        session = load_dyadic_session_tensor(self.examples[example_idx])
        self._cache[example_idx] = session
        return session

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example_idx, start = self.windows[idx]
        session = self.load_session(example_idx)
        end = start + self.window_size
        valid_len = max(0, min(end, session.x.shape[0]) - start)

        x = np.zeros((self.window_size, session.x.shape[1]), dtype=np.float32)
        y = np.zeros((self.window_size, 2), dtype=np.float32)
        target_mask = np.zeros((self.window_size, 2), dtype=np.float32)
        frame_mask = np.zeros(self.window_size, dtype=np.float32)

        if valid_len > 0:
            x[:valid_len] = session.x[start : start + valid_len]
            y[:valid_len] = session.y[start : start + valid_len]
            target_mask[:valid_len] = session.target_mask[start : start + valid_len]
            frame_mask[:valid_len] = 1.0

        # Broadcast frame_mask over target channels. A frame must be real and
        # the role-specific target must be valid to contribute to the loss.
        loss_mask = target_mask * frame_mask[:, None]
        return {
            "x": torch.from_numpy(x.T.copy()),
            "y": torch.from_numpy(y),
            "target_mask": torch.from_numpy(target_mask),
            "frame_mask": torch.from_numpy(frame_mask),
            "loss_mask": torch.from_numpy(loss_mask),
            "example_idx": torch.tensor(example_idx, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
        }


def summarize_dyadic_window(
    x: np.ndarray,
    y: np.ndarray,
    target_mask: np.ndarray,
    start: int,
    window_size: int,
    include_minmax: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Convert one dyadic sequence window into one XGBoost table row."""

    end = min(start + window_size, len(x))
    x_win = x[start:end]
    y_win = y[start:end]
    mask_win = target_mask[start:end] > 0
    if len(x_win) == 0 or not np.any(mask_win):
        return np.empty(0, dtype=np.float32), np.full(2, np.nan, dtype=np.float32), np.zeros(2, dtype=np.float32), 0

    # XGBoost remains a tabular model: it sees summary descriptors of the
    # dyadic feature window and predicts mean engagement for both roles.
    parts = [np.mean(x_win, axis=0), np.std(x_win, axis=0)]
    if include_minmax:
        parts.extend([np.min(x_win, axis=0), np.max(x_win, axis=0)])
    features = np.concatenate(parts).astype(np.float32)

    targets = np.zeros(2, dtype=np.float32)
    weights = np.zeros(2, dtype=np.float32)
    for channel in range(2):
        valid = mask_win[:, channel]
        if np.any(valid):
            targets[channel] = float(np.mean(y_win[valid, channel]))
            weights[channel] = float(np.mean(valid))
        else:
            targets[channel] = 0.0
            weights[channel] = 0.0
    return features, targets, weights, int(mask_win.sum())


def build_dyadic_window_table(
    examples: list[DyadicManifestExample],
    window_size: int,
    stride: int,
    include_minmax: bool = False,
    max_windows: int | None = None,
    seed: int = 13,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Build an in-memory dyadic window-summary table for XGBoost."""

    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []

    for example_idx, example in enumerate(examples):
        session = load_dyadic_session_tensor(example)
        for start in window_starts(example.aligned_len, window_size, stride, include_tail=True):
            features, target, weight, valid_targets = summarize_dyadic_window(
                session.x,
                session.y,
                session.target_mask,
                start=start,
                window_size=window_size,
                include_minmax=include_minmax,
            )
            if features.size == 0:
                continue
            end = min(start + window_size, example.aligned_len)
            rows.append(features)
            targets.append(target)
            weights.append(weight)
            metadata.append({"example_idx": example_idx, "start": start, "valid_len": end - start, "valid_target_count": valid_targets})

    if max_windows is not None and len(rows) > max_windows:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(rows), size=max_windows, replace=False))
        rows = [rows[int(idx)] for idx in keep]
        targets = [targets[int(idx)] for idx in keep]
        weights = [weights[int(idx)] for idx in keep]
        metadata = [metadata[int(idx)] for idx in keep]

    return (
        np.vstack(rows).astype(np.float32),
        np.vstack(targets).astype(np.float32),
        np.vstack(weights).astype(np.float32),
        metadata,
    )

