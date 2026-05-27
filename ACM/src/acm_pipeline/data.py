from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ManifestExample:
    """One session-role tensor listed in a processed/transformed manifest."""

    dataset: str
    session_id: str
    role: str
    model_split: str
    feature_set: str
    transform_method: str
    tensor_path: Path
    n_features: int
    aligned_len: int

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.session_id}/{self.role}"


@dataclass
class SessionTensor:
    """Loaded sequence arrays for one manifest example."""

    x: np.ndarray
    y: np.ndarray
    target_mask: np.ndarray


def read_model_manifest(manifest_path: Path, project_root: Path, split: str | None = None) -> list[ManifestExample]:
    """Read a transformed model manifest into typed examples."""

    examples: list[ManifestExample] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row["model_split"] != split:
                continue
            tensor_path = Path(row["tensor_relative_path"])
            if not tensor_path.is_absolute():
                tensor_path = project_root / tensor_path
            examples.append(
                ManifestExample(
                    dataset=row["dataset"],
                    session_id=row["session_id"],
                    role=row["role"],
                    model_split=row["model_split"],
                    feature_set=row.get("feature_set", ""),
                    transform_method=row.get("transform_method", ""),
                    tensor_path=tensor_path,
                    n_features=int(row["n_features"]),
                    aligned_len=int(row["aligned_len"]),
                )
            )
    return examples


def load_session_tensor(example: ManifestExample) -> SessionTensor:
    """Load one NPZ tensor and return model arrays as float32."""

    # All model scripts consume the same NPZ contract: x is [time, features],
    # y is [time], and target_mask marks frames that should count in losses.
    with np.load(example.tensor_path, allow_pickle=True) as data:
        return SessionTensor(
            x=np.asarray(data["x"], dtype=np.float32),
            y=np.asarray(data["y"], dtype=np.float32),
            target_mask=np.asarray(data["target_mask"], dtype=np.float32),
        )


def window_starts(n_frames: int, window_size: int, stride: int, include_tail: bool = True) -> list[int]:
    """Return deterministic window start indices for one sequence."""

    if n_frames <= 0:
        return []
    if n_frames <= window_size:
        return [0]
    starts = list(range(0, n_frames - window_size + 1, stride))
    if include_tail:
        tail_start = n_frames - window_size
        if starts[-1] != tail_start:
            starts.append(tail_start)
    return starts


class WindowedSequenceDataset(Dataset):
    """Lazy sequence-window dataset for TCN/Transformer-style models."""

    def __init__(
        self,
        examples: list[ManifestExample],
        window_size: int,
        stride: int,
        include_tail: bool = True,
        max_windows: int | None = None,
        seed: int = 13,
    ) -> None:
        self.examples = examples
        self.window_size = window_size
        self.stride = stride
        self._cache: dict[int, SessionTensor] = {}

        # Store only (example_idx, start) pairs. The actual tensor slices are
        # read lazily in __getitem__, avoiding duplicated window files on disk.
        windows: list[tuple[int, int]] = []
        for example_idx, example in enumerate(examples):
            for start in window_starts(example.aligned_len, window_size, stride, include_tail=include_tail):
                windows.append((example_idx, start))

        # A max-window cap is useful for smoke tests and quick local debugging.
        if max_windows is not None and len(windows) > max_windows:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(windows), size=max_windows, replace=False))
            windows = [windows[int(idx)] for idx in keep]
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def load_session(self, example_idx: int) -> SessionTensor:
        cached = self._cache.get(example_idx)
        if cached is not None:
            return cached
        session = load_session_tensor(self.examples[example_idx])
        self._cache[example_idx] = session
        return session

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example_idx, start = self.windows[idx]
        session = self.load_session(example_idx)
        end = start + self.window_size
        valid_len = max(0, min(end, session.x.shape[0]) - start)

        # Pad short/tail windows to a fixed length. frame_mask separates real
        # frames from padding; loss_mask additionally removes invalid targets.
        x = np.zeros((self.window_size, session.x.shape[1]), dtype=np.float32)
        y = np.zeros(self.window_size, dtype=np.float32)
        target_mask = np.zeros(self.window_size, dtype=np.float32)
        frame_mask = np.zeros(self.window_size, dtype=np.float32)

        if valid_len > 0:
            x[:valid_len] = session.x[start : start + valid_len]
            y[:valid_len] = session.y[start : start + valid_len]
            target_mask[:valid_len] = session.target_mask[start : start + valid_len]
            frame_mask[:valid_len] = 1.0

        return {
            "x": torch.from_numpy(x.T.copy()),
            "y": torch.from_numpy(y),
            "target_mask": torch.from_numpy(target_mask),
            "frame_mask": torch.from_numpy(frame_mask),
            "loss_mask": torch.from_numpy(target_mask * frame_mask),
            "example_idx": torch.tensor(example_idx, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
        }


def summarize_window(
    x: np.ndarray,
    y: np.ndarray,
    target_mask: np.ndarray,
    start: int,
    window_size: int,
    include_minmax: bool = False,
) -> tuple[np.ndarray, float, float, int]:
    """Convert one sequence window into one tabular row for XGBoost."""

    end = min(start + window_size, len(x))
    x_win = x[start:end]
    y_win = y[start:end]
    mask_win = target_mask[start:end] > 0
    if len(x_win) == 0 or not np.any(mask_win):
        return np.empty(0, dtype=np.float32), float("nan"), 0.0, 0

    # XGBoost receives compact window descriptors rather than all frames. Mean
    # and std are stable first baselines; min/max can be enabled from the CLI.
    parts = [np.mean(x_win, axis=0), np.std(x_win, axis=0)]
    if include_minmax:
        parts.extend([np.min(x_win, axis=0), np.max(x_win, axis=0)])
    features = np.concatenate(parts).astype(np.float32)
    target = float(np.mean(y_win[mask_win]))
    weight = float(np.mean(mask_win))
    return features, target, weight, int(mask_win.sum())


def build_window_table(
    examples: list[ManifestExample],
    window_size: int,
    stride: int,
    include_minmax: bool = False,
    max_windows: int | None = None,
    seed: int = 13,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Build an in-memory window-summary table for XGBoost."""

    rows: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    metadata: list[dict[str, object]] = []

    # Build metadata alongside the feature matrix. Validation reconstruction
    # later uses example_idx/start/valid_len to average window predictions back
    # over the original session frames.
    for example_idx, example in enumerate(examples):
        session = load_session_tensor(example)
        for start in window_starts(example.aligned_len, window_size, stride, include_tail=True):
            features, target, weight, valid_targets = summarize_window(
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
            metadata.append(
                {
                    "example_idx": example_idx,
                    "start": start,
                    "valid_len": end - start,
                    "valid_target_count": valid_targets,
                }
            )

    if max_windows is not None and len(rows) > max_windows:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(rows), size=max_windows, replace=False))
        rows = [rows[int(idx)] for idx in keep]
        targets = [targets[int(idx)] for idx in keep]
        weights = [weights[int(idx)] for idx in keep]
        metadata = [metadata[int(idx)] for idx in keep]

    return (
        np.vstack(rows).astype(np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        metadata,
    )

