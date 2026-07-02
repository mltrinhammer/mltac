from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
        return f"{self.dataset}/{self.session_id}/{self.role}/{self.feature_set}"


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

    # The active turn-level pipeline consumes role-level tensors with x
    # [time, features], y [time], and target_mask marking valid supervision.
    with np.load(example.tensor_path, allow_pickle=True) as data:
        return SessionTensor(
            x=np.asarray(data["x"], dtype=np.float32),
            y=np.asarray(data["y"], dtype=np.float32),
            target_mask=np.asarray(data["target_mask"], dtype=np.float32),
        )

