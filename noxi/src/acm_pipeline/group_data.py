"""Group-window multimodal datasets for N-participant engagement models."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.acm_pipeline.data import ManifestExample, SessionTensor, load_session_tensor


@dataclass(frozen=True)
class GroupMultimodalWindowSample:
    dataset: str
    session_id: str
    model_split: str
    combo_name: str
    modality_order: tuple[str, ...]
    role_order: tuple[str, ...]
    role_examples: dict[str, dict[str, ManifestExample]]
    start_frame: int
    end_frame: int
    window_idx: int

    @property
    def window_len(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def session_key(self) -> str:
        return f"{self.dataset}/{self.session_id}"


def _resolve_tensor_path(project_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else project_root / path


def read_group_multimodal_window_manifest(
    manifest_path: Path,
    project_root: Path,
    split: str | None = None,
) -> list[GroupMultimodalWindowSample]:
    samples: list[GroupMultimodalWindowSample] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row["model_split"] != split:
                continue

            modality_order = tuple(json.loads(row["modality_order_json"]))
            role_order = tuple(json.loads(row["role_order_json"]))
            specs = json.loads(row["modalities_json"])
            role_examples: dict[str, dict[str, ManifestExample]] = {role: {} for role in role_order}

            for modality_name in modality_order:
                modality_spec = specs[modality_name]
                for role in role_order:
                    role_spec = modality_spec["roles"][role]
                    role_examples[role][modality_name] = ManifestExample(
                        dataset=row["dataset"],
                        session_id=row["session_id"],
                        role=role,
                        model_split=row["model_split"],
                        feature_set=role_spec.get("feature_set", modality_name),
                        transform_method=role_spec.get("transform_method", row.get("transform_method", "")),
                        tensor_path=_resolve_tensor_path(project_root, role_spec["tensor_relative_path"]),
                        n_features=int(role_spec["n_features"]),
                        aligned_len=int(role_spec["aligned_len"]),
                    )

            samples.append(
                GroupMultimodalWindowSample(
                    dataset=row["dataset"],
                    session_id=row["session_id"],
                    model_split=row["model_split"],
                    combo_name=row["combo_name"],
                    modality_order=modality_order,
                    role_order=role_order,
                    role_examples=role_examples,
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    window_idx=int(row["window_idx"]),
                )
            )
    return samples


class GroupMultimodalWindowDataset(Dataset):
    """Windowed group dataset that preserves participant and modality axes."""

    def __init__(
        self,
        samples: list[GroupMultimodalWindowSample],
        min_frames: int = 5,
    ) -> None:
        self.samples = [sample for sample in samples if sample.window_len >= min_frames]
        self._cache: dict[str, SessionTensor] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, example: ManifestExample) -> SessionTensor:
        key = example.key
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        session = load_session_tensor(example)
        self._cache[key] = session
        return session

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]
        s, e = sample.start_frame, sample.end_frame
        role_order = list(sample.role_order)

        x_modalities: dict[str, list[torch.Tensor]] = {name: [] for name in sample.modality_order}
        y_rows: list[np.ndarray] = []
        mask_rows: list[np.ndarray] = []

        for role in role_order:
            reference_session: SessionTensor | None = None
            for modality_name in sample.modality_order:
                session = self._load(sample.role_examples[role][modality_name])
                x_modalities[modality_name].append(torch.from_numpy(session.x[s:e].T.copy()))
                if reference_session is None:
                    reference_session = session
            assert reference_session is not None
            y_rows.append(reference_session.y[s:e])
            mask_rows.append(reference_session.target_mask[s:e])

        return {
            "x_modalities": {name: torch.stack(tensors, dim=0) for name, tensors in x_modalities.items()},
            "modality_order": list(sample.modality_order),
            "role_order": role_order,
            "y": torch.from_numpy(np.stack(y_rows, axis=1).copy()),
            "target_mask": torch.from_numpy(np.stack(mask_rows, axis=1).copy()),
            "window_len": sample.window_len,
            "session_key": sample.session_key,
            "start_frame": sample.start_frame,
            "combo_name": sample.combo_name,
        }


def group_multimodal_window_collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    max_len = max(int(item["window_len"]) for item in batch)
    max_roles = max(len(item["role_order"]) for item in batch)
    batch_size = len(batch)
    modality_order = list(batch[0]["modality_order"])

    x_modalities: dict[str, torch.Tensor] = {}
    for modality_name in modality_order:
        n_features = batch[0]["x_modalities"][modality_name].shape[1]
        x_modalities[modality_name] = torch.zeros(batch_size, max_roles, n_features, max_len, dtype=torch.float32)

    y = torch.zeros(batch_size, max_len, max_roles, dtype=torch.float32)
    target_mask = torch.zeros(batch_size, max_len, max_roles, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, max_len, dtype=torch.float32)
    role_mask = torch.zeros(batch_size, max_roles, dtype=torch.float32)

    session_keys: list[str] = []
    role_orders: list[list[str]] = []
    start_frames = torch.zeros(batch_size, dtype=torch.long)
    window_lens = torch.zeros(batch_size, dtype=torch.long)

    for idx, item in enumerate(batch):
        window_len = int(item["window_len"])
        n_roles = len(item["role_order"])
        for modality_name in modality_order:
            x_modalities[modality_name][idx, :n_roles, :, :window_len] = item["x_modalities"][modality_name]
        y[idx, :window_len, :n_roles] = item["y"]
        target_mask[idx, :window_len, :n_roles] = item["target_mask"]
        frame_mask[idx, :window_len] = 1.0
        role_mask[idx, :n_roles] = 1.0
        session_keys.append(str(item["session_key"]))
        role_orders.append(list(item["role_order"]))
        start_frames[idx] = int(item["start_frame"])
        window_lens[idx] = window_len

    loss_mask = target_mask * frame_mask.unsqueeze(-1) * role_mask.unsqueeze(1)
    return {
        "x_modalities": x_modalities,
        "modality_order": modality_order,
        "role_orders": role_orders,
        "y": y,
        "target_mask": target_mask,
        "frame_mask": frame_mask,
        "role_mask": role_mask,
        "loss_mask": loss_mask,
        "session_keys": session_keys,
        "start_frames": start_frames,
        "window_lens": window_lens,
    }
