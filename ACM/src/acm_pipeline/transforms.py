from __future__ import annotations

from pathlib import Path

import numpy as np


class FeatureNormalizer:
    """Train-fitted per-dimension z-score normalizer."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        # Store stable float32 arrays so transform outputs are compatible with
        # PyTorch and the processed NPZ tensors stay reasonably small.
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)

    @classmethod
    def fit_npz_paths(cls, paths: list[Path]) -> "FeatureNormalizer":
        """Fit mean/std from aligned train tensors only.

        The implementation streams session tensors one at a time so the full
        dataset does not need to be materialized in memory.
        """

        if not paths:
            raise ValueError("Cannot fit a normalizer without tensor paths.")
        # Establish the expected feature dimension from the first train tensor.
        # Mixed dimensions usually mean a feature-set/cache problem and should
        # fail early rather than silently producing unusable model inputs.
        with np.load(paths[0], allow_pickle=True) as data:
            n_features = int(data["x"].shape[1])
        sums = np.zeros(n_features, dtype=np.float64)
        sums_sq = np.zeros(n_features, dtype=np.float64)
        counts = np.zeros(n_features, dtype=np.float64)
        for path in paths:
            # Accumulate sufficient statistics per feature dimension. This is
            # memory-light compared with concatenating all training sessions.
            with np.load(path, allow_pickle=True) as data:
                x = np.asarray(data["x"], dtype=np.float32)
            if x.shape[1] != n_features:
                raise ValueError(f"Mixed feature dimensions: expected {n_features}, got {x.shape[1]} in {path}")
            finite = np.isfinite(x)
            x_safe = np.where(finite, x, 0.0).astype(np.float64)
            sums += x_safe.sum(axis=0)
            sums_sq += (x_safe * x_safe).sum(axis=0)
            counts += finite.sum(axis=0)

        counts_safe = np.maximum(counts, 1.0)
        mean = sums / counts_safe
        var = np.maximum((sums_sq / counts_safe) - mean**2, 1e-12)
        std = np.sqrt(var)
        mean[counts == 0] = 0.0
        std[counts == 0] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply z-score normalization and replace non-finite values with zero."""

        out = (x.astype(np.float32, copy=False) - self.mean) / self.std
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "FeatureNormalizer":
        with np.load(path) as data:
            return cls(mean=data["mean"], std=data["std"])
