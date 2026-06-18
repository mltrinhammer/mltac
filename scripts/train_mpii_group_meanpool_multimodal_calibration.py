"""Train MPII group-level mean-pooling multimodal TCN models."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.dyadic_train_utils import write_csv, write_organizer_submission_tree
from src.acm_pipeline.group_data import (
    GroupMultimodalWindowDataset,
    GroupMultimodalWindowSample,
    group_multimodal_window_collate_fn,
    read_group_multimodal_window_manifest,
)
from src.acm_pipeline.group_models import MeanPoolGroupMultimodalTCNRegressor
from src.acm_pipeline.metrics import ccc_loss, masked_mse_loss, regression_metrics


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MPII group mean-pooling multimodal TCN.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--train-split", default="train_internal")
    parser.add_argument("--val-split", default="val_internal")
    parser.add_argument("--test-splits", nargs="*", default=["test_internal", "test"])
    parser.add_argument("--fusion-mode", choices=["gated", "concat"], default="gated")
    parser.add_argument("--fusion-channels", type=int, default=64)
    parser.add_argument("--modality-dropout", type=float, default=0.1)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--encoder-sharing", choices=("shared", "separate"), default="shared")
    parser.add_argument("--max-role-encoders", type=int, default=8)
    parser.add_argument("--prediction-head-sharing", choices=("shared", "role_specific"), default="shared")
    parser.add_argument("--prediction-interaction-scale", type=float, default=0.1)
    parser.add_argument("--metadata", type=Path, help="Optional role-level metadata CSV keyed by dataset/session_id/role.")
    parser.add_argument("--metadata-mode", choices=("age_gender_language", "age_gender", "language_only"), default="age_gender_language")
    parser.add_argument("--metadata-embedding-dim", type=int, default=16)
    parser.add_argument("--metadata-dropout", type=float, default=0.2)
    parser.add_argument("--min-window-frames", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ccc-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.0)
    parser.add_argument(
        "--delta-mse-weight",
        type=float,
        default=0.0,
        help="Weight for MSE between predicted and target temporal deltas.",
    )
    parser.add_argument("--mean-calibration-weight", type=float, default=0.0)
    parser.add_argument("--std-calibration-weight", type=float, default=0.0)
    parser.add_argument("--excess-jitter-weight", type=float, default=0.0)
    for role in ("novice", "expert"):
        parser.add_argument(f"--{role}-ccc-weight", type=float, default=None)
        parser.add_argument(f"--{role}-mse-weight", type=float, default=None)
        parser.add_argument(f"--{role}-delta-mse-weight", type=float, default=None)
        parser.add_argument(f"--{role}-mean-calibration-weight", type=float, default=None)
        parser.add_argument(f"--{role}-std-calibration-weight", type=float, default=None)
        parser.add_argument(f"--{role}-excess-jitter-weight", type=float, default=None)
    parser.add_argument(
        "--excess-jitter-threshold",
        type=float,
        default=0.01,
        help="Absolute frame-delta threshold below which prediction movement is not penalized.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


class _SessionStub:
    def __init__(self, dataset: str, session_id: str, model_split: str, role_names: tuple[str, ...]) -> None:
        self.dataset = dataset
        self.session_id = session_id
        self.model_split = model_split
        self.role_names = role_names


@dataclass(frozen=True)
class MetadataStats:
    age_mean: float
    age_std: float
    languages: tuple[str, ...]


def read_metadata(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    table: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            table[(row["dataset"], row["session_id"], row["role"])] = {
                "age": row.get("age", "").strip(),
                "gender": row.get("gender", "").strip(),
                "language": row.get("language", "").strip(),
            }
    return table


def metadata_stats(
    samples: list[GroupMultimodalWindowSample],
    table: dict[tuple[str, str, str], dict[str, str]],
) -> MetadataStats:
    seen: set[tuple[str, str, str]] = set()
    ages: list[float] = []
    languages: set[str] = set()
    for sample in samples:
        for role in sample.role_order:
            key = (sample.dataset, sample.session_id, role)
            if key in seen:
                continue
            seen.add(key)
            row = table.get(key, {})
            try:
                ages.append(float(row.get("age", "")))
            except ValueError:
                pass
            language = row.get("language", "")
            if language:
                languages.add(language)
    mean = float(np.mean(ages)) if ages else 0.0
    std = float(np.std(ages)) if ages else 1.0
    return MetadataStats(mean, std if std > 1e-6 else 1.0, tuple(sorted(languages)))


def encode_metadata(row: dict[str, str], stats: MetadataStats, mode: str) -> np.ndarray:
    values: list[float] = []
    if mode in {"age_gender_language", "age_gender"}:
        try:
            age_z = (float(row.get("age", "")) - stats.age_mean) / stats.age_std
        except ValueError:
            age_z = 0.0
        gender = row.get("gender", "")
        values.extend(
            [
                float(age_z),
                1.0 if gender == "1" else 0.0,
                1.0 if gender == "2" else 0.0,
                1.0 if gender not in {"1", "2"} else 0.0,
            ]
        )
    if mode in {"age_gender_language", "language_only"}:
        language = row.get("language", "")
        values.extend([1.0 if language == item else 0.0 for item in stats.languages])
        values.append(1.0 if language else 0.0)
    return np.asarray(values, dtype=np.float32)


class MetadataGroupMultimodalWindowDataset(GroupMultimodalWindowDataset):
    def __init__(
        self,
        samples: list[GroupMultimodalWindowSample],
        metadata_table: dict[tuple[str, str, str], dict[str, str]],
        stats: MetadataStats,
        metadata_mode: str,
        min_frames: int = 5,
    ) -> None:
        super().__init__(samples, min_frames=min_frames)
        self.metadata_table = metadata_table
        self.stats = stats
        self.metadata_mode = metadata_mode

    def __getitem__(self, idx: int) -> dict[str, object]:
        item = super().__getitem__(idx)
        sample = self.samples[idx]
        item["metadata"] = torch.from_numpy(
            np.stack(
                [
                    encode_metadata(
                        self.metadata_table.get((sample.dataset, sample.session_id, role), {}),
                        self.stats,
                        self.metadata_mode,
                    )
                    for role in sample.role_order
                ],
                axis=0,
            )
        )
        return item


def metadata_group_collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    collated = group_multimodal_window_collate_fn(batch)
    if "metadata" not in batch[0]:
        return collated
    metadata_dim = int(batch[0]["metadata"].shape[-1])
    metadata = torch.zeros(len(batch), collated["role_mask"].shape[1], metadata_dim, dtype=torch.float32)
    for idx, item in enumerate(batch):
        n_roles = item["metadata"].shape[0]
        metadata[idx, :n_roles] = item["metadata"]
    collated["metadata"] = metadata
    return collated


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(name)


def make_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name.strip() or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_mpii_group_meanpool"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def move_modalities_to_device(x_modalities: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in x_modalities.items()}


def infer_layout(samples: list[GroupMultimodalWindowSample]) -> tuple[str, tuple[str, ...], dict[str, int]]:
    if not samples:
        raise RuntimeError("At least one group-window row is required.")
    combo_name = samples[0].combo_name
    modality_order = samples[0].modality_order
    modality_dims: dict[str, int] = {}
    for sample in samples:
        if sample.combo_name != combo_name:
            raise RuntimeError(f"Expected one combo_name, got {combo_name!r} and {sample.combo_name!r}")
        if sample.modality_order != modality_order:
            raise RuntimeError("All samples must share modality order.")
        for modality_name in modality_order:
            dims = {sample.role_examples[role][modality_name].n_features for role in sample.role_order}
            if len(dims) != 1:
                raise RuntimeError(f"Inconsistent dimensions for {sample.session_key}/{modality_name}: {sorted(dims)}")
            dim = next(iter(dims))
            current = modality_dims.get(modality_name)
            if current is None:
                modality_dims[modality_name] = dim
            elif current != dim:
                raise RuntimeError(f"Inconsistent dimensions for modality {modality_name}: {current} vs {dim}")
    return combo_name, modality_order, modality_dims


def build_model(args: argparse.Namespace, modality_dims: dict[str, int]) -> torch.nn.Module:
    return MeanPoolGroupMultimodalTCNRegressor(
        modality_dims=modality_dims,
        fusion_channels=args.fusion_channels,
        fusion_mode=args.fusion_mode,
        modality_dropout=args.modality_dropout,
        hidden_channels=args.hidden_channels,
        levels=args.levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        encoder_sharing=args.encoder_sharing,
        max_role_encoders=args.max_role_encoders,
        prediction_head_sharing=args.prediction_head_sharing,
        prediction_interaction_scale=args.prediction_interaction_scale,
        metadata_dim=getattr(args, "metadata_dim", 0),
        metadata_embedding_dim=args.metadata_embedding_dim,
        metadata_dropout=args.metadata_dropout,
    )


def masked_delta_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    adjacent_mask = mask[:, 1:] & mask[:, :-1]
    if not torch.any(adjacent_mask):
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    pred_delta = pred[:, 1:] - pred[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    return (pred_delta[adjacent_mask] - target_delta[adjacent_mask]).pow(2).mean()


def masked_mean_std_calibration_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mean_weight: float,
    std_weight: float,
) -> torch.Tensor:
    mask = mask.bool()
    losses = []
    for role_idx in range(pred.shape[2]):
        role_mask = mask[:, :, role_idx]
        if not torch.any(role_mask):
            continue
        p = pred[:, :, role_idx][role_mask]
        y = target[:, :, role_idx][role_mask]
        if mean_weight > 0.0:
            losses.append(mean_weight * (p.mean() - y.mean()).pow(2))
        if std_weight > 0.0 and p.numel() > 1:
            losses.append(std_weight * (p.std(unbiased=False) - y.std(unbiased=False)).pow(2))
    if not losses:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return torch.stack(losses).sum()


def masked_excess_jitter_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    mask = mask.bool()
    adjacent_mask = mask[:, 1:] & mask[:, :-1]
    if not torch.any(adjacent_mask):
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    pred_delta = (pred[:, 1:] - pred[:, :-1]).abs()
    excess = torch.relu(pred_delta - float(threshold))
    return excess[adjacent_mask].pow(2).mean()


def role_mask_from_orders(loss_mask: torch.Tensor, role_orders: list[list[str]], role: str) -> torch.Tensor:
    role_mask = torch.zeros_like(loss_mask)
    for row_idx, order in enumerate(role_orders):
        if role not in order:
            continue
        role_idx = order.index(role)
        role_mask[row_idx, :, role_idx] = loss_mask[row_idx, :, role_idx]
    return role_mask

def role_value(args: argparse.Namespace, role: str, name: str, default: float) -> float:
    value = getattr(args, f"{role}_{name}")
    return float(default if value is None else value)

def has_role_loss_overrides(args: argparse.Namespace) -> bool:
    names = (
        "ccc_weight",
        "mse_weight",
        "delta_mse_weight",
        "mean_calibration_weight",
        "std_calibration_weight",
        "excess_jitter_weight",
    )
    return any(getattr(args, f"{role}_{name}") is not None for role in ("novice", "expert") for name in names)

def weighted_loss_for_mask(
    pred: torch.Tensor,
    y: torch.Tensor,
    loss_mask: torch.Tensor,
    ccc_weight: float,
    mse_weight: float,
    delta_mse_weight: float,
    mean_calibration_weight: float,
    std_calibration_weight: float,
    excess_jitter_weight: float,
    excess_jitter_threshold: float,
) -> torch.Tensor:
    loss = mse_weight * masked_mse_loss(pred, y, loss_mask) + ccc_weight * ccc_loss(pred, y, loss_mask)
    if delta_mse_weight > 0.0:
        loss = loss + delta_mse_weight * masked_delta_mse_loss(pred, y, loss_mask)
    if mean_calibration_weight > 0.0 or std_calibration_weight > 0.0:
        loss = loss + masked_mean_std_calibration_loss(
            pred,
            y,
            loss_mask,
            mean_calibration_weight,
            std_calibration_weight,
        )
    if excess_jitter_weight > 0.0:
        loss = loss + excess_jitter_weight * masked_excess_jitter_loss(
            pred,
            loss_mask,
            excess_jitter_threshold,
        )
    return loss


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        x_modalities = move_modalities_to_device(batch["x_modalities"], device)
        role_mask = batch["role_mask"].to(device)
        y = batch["y"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        metadata = batch.get("metadata")
        if metadata is not None:
            metadata = metadata.to(device)
        pred = model(x_modalities, role_mask=role_mask, metadata=metadata)
        if has_role_loss_overrides(args):
            role_losses = []
            for role in ("novice", "expert"):
                role_loss_mask = role_mask_from_orders(loss_mask, batch["role_orders"], role)
                if not torch.any(role_loss_mask):
                    continue
                role_losses.append(
                    weighted_loss_for_mask(
                        pred,
                        y,
                        role_loss_mask,
                        role_value(args, role, "ccc_weight", args.ccc_weight),
                        role_value(args, role, "mse_weight", args.mse_weight),
                        role_value(args, role, "delta_mse_weight", args.delta_mse_weight),
                        role_value(args, role, "mean_calibration_weight", args.mean_calibration_weight),
                        role_value(args, role, "std_calibration_weight", args.std_calibration_weight),
                        role_value(args, role, "excess_jitter_weight", args.excess_jitter_weight),
                        args.excess_jitter_threshold,
                    )
                )
            if not role_losses:
                continue
            loss = torch.stack(role_losses).mean()
        else:
            loss = weighted_loss_for_mask(
                pred,
                y,
                loss_mask,
                args.ccc_weight,
                args.mse_weight,
                args.delta_mse_weight,
                args.mean_calibration_weight,
                args.std_calibration_weight,
                args.excess_jitter_weight,
                args.excess_jitter_threshold,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def reconstruct(
    model: torch.nn.Module,
    dataset: GroupMultimodalWindowDataset,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    session_info: dict[str, dict[str, object]] = {}
    for sample in dataset.samples:
        key = sample.session_key
        if key in session_info:
            continue
        aligned_len = min(
            sample.role_examples[role][modality].aligned_len
            for role in sample.role_order
            for modality in sample.modality_order
        )
        session_info[key] = {
            "aligned_len": aligned_len,
            "dataset": sample.dataset,
            "session_id": sample.session_id,
            "model_split": sample.model_split,
            "role_names": sample.role_order,
            "sample": sample,
        }

    sums = {
        key: np.zeros((int(info["aligned_len"]), len(info["role_names"])), dtype=np.float64)
        for key, info in session_info.items()
    }
    counts = {
        key: np.zeros((int(info["aligned_len"]), len(info["role_names"])), dtype=np.float64)
        for key, info in session_info.items()
    }

    for batch in loader:
        metadata = batch.get("metadata")
        if metadata is not None:
            metadata = metadata.to(device)
        pred_tensor = model(
            move_modalities_to_device(batch["x_modalities"], device),
            role_mask=batch["role_mask"].to(device),
            metadata=metadata,
        )
        pred = pred_tensor.detach().cpu().numpy()
        frame_mask = batch["frame_mask"].numpy()
        role_mask = batch["role_mask"].numpy()
        start_frames = batch["start_frames"].numpy()
        for row, key in enumerate(batch["session_keys"]):
            start = int(start_frames[row])
            valid_len = int(frame_mask[row].sum())
            n_roles = int(role_mask[row].sum())
            if valid_len <= 0 or n_roles <= 0:
                continue
            end = start + valid_len
            sums[key][start:end, :n_roles] += pred[row, :valid_len, :n_roles]
            counts[key][start:end, :n_roles] += 1.0

    reconstructed: list[dict[str, object]] = []
    for key, info in session_info.items():
        sample = info["sample"]
        aligned_len = int(info["aligned_len"])
        role_names = tuple(info["role_names"])
        y_true_rows: list[np.ndarray] = []
        mask_rows: list[np.ndarray] = []
        reference_modality = sample.modality_order[0]
        for role in role_names:
            session = dataset._load(sample.role_examples[role][reference_modality])
            y_true_rows.append(session.y[:aligned_len])
            mask_rows.append(session.target_mask[:aligned_len])
        y_true = np.stack(y_true_rows, axis=1)
        target_mask = np.stack(mask_rows, axis=1)
        y_pred = np.full_like(y_true, np.nan, dtype=np.float32)
        covered = counts[key] > 0
        y_pred[covered] = (sums[key][covered] / counts[key][covered]).astype(np.float32)
        reconstructed.append(
            {
                "example": _SessionStub(
                    dataset=str(info["dataset"]),
                    session_id=str(info["session_id"]),
                    model_split=str(info["model_split"]),
                    role_names=role_names,
                ),
                "y_true": y_true,
                "target_mask": target_mask,
                "y_pred": y_pred,
                "covered": np.any(covered, axis=1).astype(np.float32),
            }
        )
    return reconstructed


def write_group_metrics(run_dir: Path, reconstructed: list[dict[str, object]]) -> dict[str, float]:
    overall = regression_metrics(
        np.concatenate([item["y_true"].reshape(-1) for item in reconstructed]),
        np.concatenate([item["y_pred"].reshape(-1) for item in reconstructed]),
        np.concatenate([item["target_mask"].reshape(-1) for item in reconstructed]),
    )
    write_csv(
        run_dir / "metrics_overall.csv",
        ["group", "n_frames", "ccc", "mae", "rmse", "pearson"],
        [{"group": "overall", "n_frames": overall.n_frames, "ccc": overall.ccc, "mae": overall.mae, "rmse": overall.rmse, "pearson": overall.pearson}],
    )

    role_rows = []
    for role in sorted({role for item in reconstructed for role in item["example"].role_names}):
        y_true_parts = []
        y_pred_parts = []
        mask_parts = []
        for item in reconstructed:
            role_names = item["example"].role_names
            if role not in role_names:
                continue
            idx = role_names.index(role)
            y_true_parts.append(item["y_true"][:, idx])
            y_pred_parts.append(item["y_pred"][:, idx])
            mask_parts.append(item["target_mask"][:, idx])
        metrics = regression_metrics(np.concatenate(y_true_parts), np.concatenate(y_pred_parts), np.concatenate(mask_parts))
        role_rows.append({"role": role, "n_frames": metrics.n_frames, "ccc": metrics.ccc, "mae": metrics.mae, "rmse": metrics.rmse, "pearson": metrics.pearson})
    if role_rows:
        write_csv(run_dir / "metrics_by_role.csv", list(role_rows[0].keys()), role_rows)

    session_rows = []
    for item in reconstructed:
        metrics = regression_metrics(item["y_true"].reshape(-1), item["y_pred"].reshape(-1), item["target_mask"].reshape(-1))
        session_rows.append(
            {
                "dataset": item["example"].dataset,
                "session_id": item["example"].session_id,
                "n_frames": metrics.n_frames,
                "ccc": metrics.ccc,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "pearson": metrics.pearson,
            }
        )
    if session_rows:
        write_csv(run_dir / "metrics_by_session.csv", list(session_rows[0].keys()), session_rows)
    return {"ccc": overall.ccc, "mae": overall.mae, "rmse": overall.rmse, "pearson": overall.pearson}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = make_run_dir(args)

    train_samples = read_group_multimodal_window_manifest(args.manifest, PROJECT_ROOT, split=args.train_split)
    val_samples = read_group_multimodal_window_manifest(args.manifest, PROJECT_ROOT, split=args.val_split)
    if not train_samples or not val_samples:
        raise RuntimeError("Both train and validation group-window rows are required.")

    combo_name, modality_order, modality_dims = infer_layout(train_samples + val_samples)
    metadata_stats_payload = None
    if args.metadata is not None:
        metadata_table = read_metadata(args.metadata)
        stats = metadata_stats(train_samples, metadata_table)
        args.metadata_dim = int(encode_metadata({}, stats, args.metadata_mode).shape[0])
        train_dataset = MetadataGroupMultimodalWindowDataset(train_samples, metadata_table, stats, args.metadata_mode, args.min_window_frames)
        val_dataset = MetadataGroupMultimodalWindowDataset(val_samples, metadata_table, stats, args.metadata_mode, args.min_window_frames)
        collate_fn = metadata_group_collate_fn
        metadata_stats_payload = {"age_mean": stats.age_mean, "age_std": stats.age_std, "languages": list(stats.languages)}
    else:
        args.metadata_dim = 0
        train_dataset = GroupMultimodalWindowDataset(train_samples, min_frames=args.min_window_frames)
        val_dataset = GroupMultimodalWindowDataset(val_samples, min_frames=args.min_window_frames)
        collate_fn = group_multimodal_window_collate_fn
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    config = serializable_args(args)
    config.update(
        {
            "run_dir": str(run_dir),
            "combo_name": combo_name,
            "modality_order": list(modality_order),
            "modality_dims": modality_dims,
            "n_train_windows": len(train_dataset),
            "n_val_windows": len(val_dataset),
            "metadata_dim": args.metadata_dim,
            "metadata_stats": metadata_stats_payload,
        }
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    model = build_model(args, modality_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_ccc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    log_rows: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args,
        )
        reconstructed = reconstruct(model, val_dataset, val_loader, device)
        val_metrics = write_group_metrics(run_dir, reconstructed)
        val_ccc = val_metrics["ccc"]
        improved = np.isfinite(val_ccc) and val_ccc > best_ccc + args.min_delta
        if improved:
            best_ccc = val_ccc
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_ccc": val_ccc,
                    "args": serializable_args(args),
                    "combo_name": combo_name,
                    "modality_dims": modality_dims,
                },
                run_dir / "model_best.pt",
            )
        else:
            stale_epochs += 1
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_ccc": val_metrics["ccc"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_pearson": val_metrics["pearson"],
                "best_epoch": best_epoch,
                "best_val_ccc": best_ccc,
                "stale_epochs": stale_epochs,
            }
        )
        write_csv(run_dir / "training_log.csv", list(log_rows[0].keys()), log_rows)
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_ccc={val_ccc:.5f} best_epoch={best_epoch}", flush=True)
        if args.patience > 0 and epoch >= args.min_epochs and stale_epochs >= args.patience:
            print(f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d} best_val_ccc={best_ccc:.5f}", flush=True)
            break

    best_checkpoint = run_dir / "model_best.pt"
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        reconstructed = reconstruct(model, val_dataset, val_loader, device)
        write_group_metrics(run_dir, reconstructed)
        write_organizer_submission_tree(run_dir / "val_submission_format", reconstructed)

        for test_split in args.test_splits or []:
            test_samples = read_group_multimodal_window_manifest(args.manifest, PROJECT_ROOT, split=test_split)
            if not test_samples:
                continue
            if args.metadata is not None:
                test_dataset = MetadataGroupMultimodalWindowDataset(test_samples, metadata_table, stats, args.metadata_mode, args.min_window_frames)
            else:
                test_dataset = GroupMultimodalWindowDataset(test_samples, min_frames=args.min_window_frames)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
            test_reconstructed = reconstruct(model, test_dataset, test_loader, device)
            write_organizer_submission_tree(run_dir / f"{test_split}_submission_format", test_reconstructed)
            print(f"test_split={test_split} sessions={len(test_reconstructed)}", flush=True)

    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
