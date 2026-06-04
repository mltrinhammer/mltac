"""Turn-based dyadic dataset for TCN models.

Each sample corresponds to one speech turn and provides **both** roles'
features for that interval, kept as separate tensors until collation.  A
custom collate function pads variable-length turns within each batch.
"""

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
class ManifestTurnSample:
    """One precomputed paired-turn row listed in a turn manifest."""

    session_id: str
    dataset: str
    speaker: str
    novice_example: ManifestExample
    expert_example: ManifestExample
    start_frame: int
    end_frame: int
    turn_idx: int

    @property
    def turn_len(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def session_key(self) -> str:
        return f"{self.dataset}/{self.session_id}"


@dataclass(frozen=True)
class MultimodalManifestTurnSample:
    """One precomputed paired-turn row spanning multiple modalities."""

    session_id: str
    dataset: str
    speaker: str
    model_split: str
    combo_name: str
    modality_order: tuple[str, ...]
    novice_examples: dict[str, ManifestExample]
    expert_examples: dict[str, ManifestExample]
    start_frame: int
    end_frame: int
    turn_idx: int

    @property
    def turn_len(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def session_key(self) -> str:
        return f"{self.dataset}/{self.session_id}"


def _resolve_tensor_path(project_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else project_root / path


def read_turn_manifest(manifest_path: Path, project_root: Path, split: str | None = None) -> list[ManifestTurnSample]:
    """Read a paired turn manifest into turn samples backed by source tensors."""

    turns: list[ManifestTurnSample] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row["model_split"] != split:
                continue

            common = {
                "dataset": row["dataset"],
                "session_id": row["session_id"],
                "model_split": row["model_split"],
                "feature_set": row.get("feature_set", ""),
                "transform_method": row.get("transform_method", ""),
                "n_features": int(row["n_features_per_role"]),
            }
            novice_example = ManifestExample(
                role="novice",
                tensor_path=_resolve_tensor_path(project_root, row["novice_tensor_relative_path"]),
                aligned_len=int(row["novice_aligned_len"]),
                **common,
            )
            expert_example = ManifestExample(
                role="expert",
                tensor_path=_resolve_tensor_path(project_root, row["expert_tensor_relative_path"]),
                aligned_len=int(row["expert_aligned_len"]),
                **common,
            )
            turns.append(
                ManifestTurnSample(
                    session_id=row["session_id"],
                    dataset=row["dataset"],
                    speaker=row["speaker"],
                    novice_example=novice_example,
                    expert_example=expert_example,
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    turn_idx=int(row["turn_idx"]),
                )
            )
    return turns


def read_multimodal_turn_manifest(
    manifest_path: Path,
    project_root: Path,
    split: str | None = None,
) -> list[MultimodalManifestTurnSample]:
    """Read a multimodal paired turn manifest into typed samples."""

    turns: list[MultimodalManifestTurnSample] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row["model_split"] != split:
                continue

            modality_order = tuple(json.loads(row["modality_order_json"]))
            modality_specs = json.loads(row["modalities_json"])
            novice_examples: dict[str, ManifestExample] = {}
            expert_examples: dict[str, ManifestExample] = {}

            for modality_name in modality_order:
                spec = modality_specs[modality_name]
                common = {
                    "dataset": row["dataset"],
                    "session_id": row["session_id"],
                    "model_split": row["model_split"],
                    "feature_set": spec.get("feature_set", modality_name),
                    "transform_method": spec.get("transform_method", row.get("transform_method", "")),
                    "n_features": int(spec["n_features_per_role"]),
                }
                novice_examples[modality_name] = ManifestExample(
                    role="novice",
                    tensor_path=_resolve_tensor_path(project_root, spec["novice_tensor_relative_path"]),
                    aligned_len=int(spec["novice_aligned_len"]),
                    **common,
                )
                expert_examples[modality_name] = ManifestExample(
                    role="expert",
                    tensor_path=_resolve_tensor_path(project_root, spec["expert_tensor_relative_path"]),
                    aligned_len=int(spec["expert_aligned_len"]),
                    **common,
                )

            turns.append(
                MultimodalManifestTurnSample(
                    session_id=row["session_id"],
                    dataset=row["dataset"],
                    speaker=row["speaker"],
                    model_split=row["model_split"],
                    combo_name=row["combo_name"],
                    modality_order=modality_order,
                    novice_examples=novice_examples,
                    expert_examples=expert_examples,
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    turn_idx=int(row["turn_idx"]),
                )
            )
    return turns


class TurnDataset(Dataset):
    """Variable-length turn-based dataset for dyadic TCN training.

    Each item returns both roles' features and targets for one turn interval.
    A custom :func:`turn_collate_fn` must be used with the DataLoader so that
    variable-length turns are padded to the batch maximum.
    """

    def __init__(
        self,
        turns: list[ManifestTurnSample],
        min_frames: int = 5,
    ) -> None:
        self.turns = [t for t in turns if t.turn_len >= min_frames]
        self._cache: dict[str, SessionTensor] = {}

    def __len__(self) -> int:
        return len(self.turns)

    def _load(self, example: ManifestExample) -> SessionTensor:
        key = example.key
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        session = load_session_tensor(example)
        self._cache[key] = session
        return session

    def __getitem__(self, idx: int) -> dict[str, object]:
        turn = self.turns[idx]
        s, e = turn.start_frame, turn.end_frame

        novice_session = self._load(turn.novice_example)
        expert_session = self._load(turn.expert_example)

        # Slice both roles' features and targets for this turn interval.
        x_novice = novice_session.x[s:e].T.copy()  # [F, turn_len]
        x_expert = expert_session.x[s:e].T.copy()   # [F, turn_len]

        y_novice = novice_session.y[s:e]             # [turn_len]
        y_expert = expert_session.y[s:e]
        y = np.stack([y_novice, y_expert], axis=1)   # [turn_len, 2]

        tm_novice = novice_session.target_mask[s:e]
        tm_expert = expert_session.target_mask[s:e]
        target_mask = np.stack([tm_novice, tm_expert], axis=1)  # [turn_len, 2]

        return {
            "x_novice": torch.from_numpy(x_novice),
            "x_expert": torch.from_numpy(x_expert),
            "y": torch.from_numpy(y),
            "target_mask": torch.from_numpy(target_mask),
            "turn_len": turn.turn_len,
            "session_key": turn.session_key,
            "start_frame": turn.start_frame,
        }


class MultimodalTurnDataset(Dataset):
    """Variable-length turn dataset that keeps modalities separate until fusion."""

    def __init__(
        self,
        turns: list[MultimodalManifestTurnSample],
        min_frames: int = 5,
    ) -> None:
        self.turns = [t for t in turns if t.turn_len >= min_frames]
        self._cache: dict[str, SessionTensor] = {}

    def __len__(self) -> int:
        return len(self.turns)

    def _load(self, example: ManifestExample) -> SessionTensor:
        key = example.key
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        session = load_session_tensor(example)
        self._cache[key] = session
        return session

    def __getitem__(self, idx: int) -> dict[str, object]:
        turn = self.turns[idx]
        s, e = turn.start_frame, turn.end_frame

        x_modalities: dict[str, dict[str, torch.Tensor]] = {}
        y: torch.Tensor | None = None
        target_mask: torch.Tensor | None = None

        for modality_name in turn.modality_order:
            novice_session = self._load(turn.novice_examples[modality_name])
            expert_session = self._load(turn.expert_examples[modality_name])

            x_novice = torch.from_numpy(novice_session.x[s:e].T.copy())
            x_expert = torch.from_numpy(expert_session.x[s:e].T.copy())
            x_modalities[modality_name] = {
                "x_novice": x_novice,
                "x_expert": x_expert,
            }

            if y is None:
                y_novice = novice_session.y[s:e]
                y_expert = expert_session.y[s:e]
                y = torch.from_numpy(np.stack([y_novice, y_expert], axis=1))

                tm_novice = novice_session.target_mask[s:e]
                tm_expert = expert_session.target_mask[s:e]
                target_mask = torch.from_numpy(np.stack([tm_novice, tm_expert], axis=1))

        assert y is not None
        assert target_mask is not None
        return {
            "x_modalities": x_modalities,
            "modality_order": list(turn.modality_order),
            "y": y,
            "target_mask": target_mask,
            "turn_len": turn.turn_len,
            "session_key": turn.session_key,
            "start_frame": turn.start_frame,
            "combo_name": turn.combo_name,
        }


def turn_collate_fn(batch: list[dict[str, object]]) -> dict[str, torch.Tensor | list]:
    """Pad variable-length turn samples to the batch maximum and stack.

    Features from both roles are concatenated along the channel dimension
    (``[novice_features, expert_features]``) to produce ``x [B, 2F, T_max]``,
    matching the input contract of the separate-encoder TCN models.
    """

    max_len = max(item["turn_len"] for item in batch)
    B = len(batch)
    F = batch[0]["x_novice"].shape[0]

    x = torch.zeros(B, 2 * F, max_len, dtype=torch.float32)
    y = torch.zeros(B, max_len, 2, dtype=torch.float32)
    target_mask = torch.zeros(B, max_len, 2, dtype=torch.float32)
    frame_mask = torch.zeros(B, max_len, dtype=torch.float32)

    session_keys: list[str] = []
    start_frames = torch.zeros(B, dtype=torch.long)
    turn_lens = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        L = item["turn_len"]
        x[i, :F, :L] = item["x_novice"]
        x[i, F:, :L] = item["x_expert"]
        y[i, :L] = item["y"]
        target_mask[i, :L] = item["target_mask"]
        frame_mask[i, :L] = 1.0
        session_keys.append(item["session_key"])
        start_frames[i] = item["start_frame"]
        turn_lens[i] = L

    loss_mask = target_mask * frame_mask.unsqueeze(-1)

    return {
        "x": x,
        "y": y,
        "target_mask": target_mask,
        "frame_mask": frame_mask,
        "loss_mask": loss_mask,
        "session_keys": session_keys,
        "start_frames": start_frames,
        "turn_lens": turn_lens,
    }


def multimodal_turn_collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    """Pad variable-length multimodal turn samples to the batch maximum."""

    max_len = max(item["turn_len"] for item in batch)
    batch_size = len(batch)
    modality_order = list(batch[0]["modality_order"])

    x_modalities: dict[str, torch.Tensor] = {}
    for modality_name in modality_order:
        n_features = batch[0]["x_modalities"][modality_name]["x_novice"].shape[0]
        x_modalities[modality_name] = torch.zeros(batch_size, 2 * n_features, max_len, dtype=torch.float32)

    y = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)
    target_mask = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, max_len, dtype=torch.float32)

    session_keys: list[str] = []
    start_frames = torch.zeros(batch_size, dtype=torch.long)
    turn_lens = torch.zeros(batch_size, dtype=torch.long)

    for idx, item in enumerate(batch):
        turn_len = item["turn_len"]
        for modality_name in modality_order:
            novice_x = item["x_modalities"][modality_name]["x_novice"]
            n_features = novice_x.shape[0]
            x_modalities[modality_name][idx, :n_features, :turn_len] = novice_x
            x_modalities[modality_name][idx, n_features:, :turn_len] = item["x_modalities"][modality_name]["x_expert"]
        y[idx, :turn_len] = item["y"]
        target_mask[idx, :turn_len] = item["target_mask"]
        frame_mask[idx, :turn_len] = 1.0
        session_keys.append(item["session_key"])
        start_frames[idx] = item["start_frame"]
        turn_lens[idx] = turn_len

    loss_mask = target_mask * frame_mask.unsqueeze(-1)

    return {
        "x_modalities": x_modalities,
        "modality_order": modality_order,
        "y": y,
        "target_mask": target_mask,
        "frame_mask": frame_mask,
        "loss_mask": loss_mask,
        "session_keys": session_keys,
        "start_frames": start_frames,
        "turn_lens": turn_lens,
    }
