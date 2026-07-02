"""Train multimodal turn-level TCN baselines from paired multimodal turn manifests.

This trainer mirrors the unimodal turn-level training contract but consumes a
joined multimodal turn manifest and applies within-role multimodal fusion
before one of the retained TCN backbones predicts both novice and expert
engagement on every interval.
"""

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

from src.acm_pipeline.dyadic_train_utils import (
    grouped_dyadic_metric_outputs,
    write_csv,
    write_dyadic_prediction_csv,
    write_organizer_submission_tree,
)
from src.acm_pipeline.metrics import ccc_loss, masked_mse_loss
from src.acm_pipeline.models_tcn import MultimodalTurnTCNRegressor
from src.acm_pipeline.turn_data import (
    MultimodalManifestTurnSample,
    MultimodalTurnDataset,
    multimodal_turn_collate_fn,
    read_multimodal_turn_manifest,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal turn-segmented TCN models from paired multimodal supervision.")
    parser.add_argument("--manifest", type=Path, required=True, help="Multimodal paired turn manifest CSV.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--train-split", default="train_internal")
    parser.add_argument("--val-split", default="val_internal")
    parser.add_argument("--test-splits", nargs="*", default=["test_internal", "test_additional"],
                        help="Splits to run test inference on after training.")

    parser.add_argument("--backbone", choices=["simple", "dyadic_shared", "dyadic_role_heads", "gated_partner", "shared_attention", "attention"], default="dyadic_shared")
    parser.add_argument("--fusion-mode", choices=["gated", "concat"], default="gated")
    parser.add_argument("--fusion-channels", type=int, default=64)
    parser.add_argument("--modality-dropout", type=float, default=0.1)
    parser.add_argument("--metadata", nargs="*", type=Path, default=[], help="Role metadata CSV files keyed by dataset/session_id/role.")
    parser.add_argument("--metadata-set", choices=["none", "domain_role", "full"], default="none")
    parser.add_argument("--metadata-injection", choices=["none", "output_calibration", "before_head"], default="none")
    parser.add_argument("--metadata-embedding-dim", type=int, default=16)
    parser.add_argument("--metadata-dropout", type=float, default=0.1)
    parser.add_argument("--preload-tensors", action="store_true", help="Eagerly load all train/validation tensors into each dataset cache before training.")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--causal-tcn", action="store_true", help="Use left-padded causal temporal convolutions in supported TCN backbones.")
    parser.add_argument("--attention-context", choices=["self", "partner", "joint"], default="joint")
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-past-frames", type=int, default=1500)
    parser.add_argument("--exclude-current-frame", action="store_true")

    parser.add_argument("--min-turn-frames", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ccc-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.0,
                        help="Weight for MSE loss component. Set to 0.0 for CCC-only loss.")
    parser.add_argument("--role-mean-calibration-weight", type=float, default=0.0,
                        help="Per-role batch mean calibration penalty weight.")
    parser.add_argument("--role-std-calibration-weight", type=float, default=0.0,
                        help="Per-role batch standard deviation calibration penalty weight.")
    parser.add_argument("--extreme-mse-weight", type=float, default=0.0,
                        help="Auxiliary MSE weight with target extremes upweighted.")
    parser.add_argument("--extreme-weight", type=float, default=1.0,
                        help="Multiplier for targets below/above the extreme thresholds.")
    parser.add_argument("--extreme-low-threshold", type=float, default=0.3)
    parser.add_argument("--extreme-high-threshold", type=float, default=0.8)
    parser.add_argument("--bin-mean-loss-weight", type=float, default=0.0,
                        help="Coarse target-bin mean matching penalty weight.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=0, help="Print train batch progress every N batches; 0 disables batch progress logging.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--save-final-checkpoint",
        action="store_true",
        help="Also save model_final.pt after the last trained epoch for fixed-epoch final training.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load output-root/run-name/model_best.pt and export validation/test predictions without training.",
    )
    return parser.parse_args()


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
    run_name = args.run_name.strip() or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_tcn_multimodal_{args.backbone}_{args.fusion_mode}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    def convert(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return {k: convert(v) for k, v in vars(args).items()}


def move_modalities_to_device(x_modalities: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in x_modalities.items()}


ROLES = ("novice", "expert")
DOMAINS = ("noxi", "noxij")


@dataclass(frozen=True)
class MetadataStats:
    age_mean: float
    age_std: float
    languages: tuple[str, ...]


def read_role_metadata(paths: list[Path]) -> dict[tuple[str, str, str], dict[str, str]]:
    table: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                key = (row["dataset"].strip(), row["session_id"].strip(), row["role"].strip())
                table[key] = {
                    "age": row.get("age", "").strip(),
                    "gender": row.get("gender", "").strip(),
                    "language": row.get("language", "").strip(),
                }
    return table


def metadata_stats(
    turns: list[MultimodalManifestTurnSample],
    table: dict[tuple[str, str, str], dict[str, str]],
) -> MetadataStats:
    seen: set[tuple[str, str, str]] = set()
    ages: list[float] = []
    languages: set[str] = set()
    for turn in turns:
        for role in ROLES:
            key = (turn.dataset, turn.session_id, role)
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


def encode_role_metadata(
    dataset: str,
    role: str,
    row: dict[str, str],
    stats: MetadataStats,
    metadata_set: str,
) -> np.ndarray:
    values: list[float] = []
    values.extend(1.0 if dataset == domain else 0.0 for domain in DOMAINS)
    values.extend(1.0 if role == item else 0.0 for item in ROLES)
    if metadata_set == "full":
        try:
            age = float(row.get("age", ""))
            age_missing = 0.0
        except ValueError:
            age = stats.age_mean
            age_missing = 1.0
        gender = row.get("gender", "")
        language = row.get("language", "")
        values.extend(
            [
                (age - stats.age_mean) / stats.age_std,
                age_missing,
                1.0 if gender == "1" else 0.0,
                1.0 if gender == "2" else 0.0,
                1.0 if gender not in {"1", "2"} else 0.0,
            ]
        )
        values.extend(1.0 if language == item else 0.0 for item in stats.languages)
        values.append(1.0 if not language else 0.0)
    return np.asarray(values, dtype=np.float32)


class MetadataMultimodalTurnDataset(MultimodalTurnDataset):
    def __init__(
        self,
        turns: list[MultimodalManifestTurnSample],
        metadata_table: dict[tuple[str, str, str], dict[str, str]],
        stats: MetadataStats,
        metadata_set: str,
        min_frames: int = 5,
    ) -> None:
        super().__init__(turns, min_frames=min_frames)
        self.metadata_vectors = [
            torch.from_numpy(
                np.stack(
                    [
                        encode_role_metadata(
                            turn.dataset,
                            role,
                            metadata_table.get((turn.dataset, turn.session_id, role), {}),
                            stats,
                            metadata_set,
                        )
                        for role in ROLES
                    ],
                    axis=0,
                )
            )
            for turn in self.turns
        ]

    def __getitem__(self, idx: int) -> dict[str, object]:
        item = super().__getitem__(idx)
        item["metadata"] = self.metadata_vectors[idx]
        return item


def metadata_to_device(batch: dict[str, object], device: torch.device) -> torch.Tensor | None:
    metadata = batch.get("metadata")
    if metadata is None:
        return None
    return metadata.to(device)


def infer_layout(turns: list[MultimodalManifestTurnSample]) -> tuple[str, tuple[str, ...], dict[str, int]]:
    if not turns:
        raise RuntimeError("At least one multimodal turn row is required.")

    combo_name = turns[0].combo_name
    modality_order = turns[0].modality_order
    modality_dims: dict[str, int] = {}
    for turn in turns:
        if turn.combo_name != combo_name:
            raise RuntimeError(f"Expected one combo_name, got {combo_name!r} and {turn.combo_name!r}")
        if turn.modality_order != modality_order:
            raise RuntimeError("All multimodal turns must share the same modality order.")
        for modality_name in modality_order:
            novice_dim = turn.novice_examples[modality_name].n_features
            expert_dim = turn.expert_examples[modality_name].n_features
            if novice_dim != expert_dim:
                raise RuntimeError(
                    f"Modality {modality_name!r} has mismatched role dimensions: novice={novice_dim}, expert={expert_dim}"
                )
            current = modality_dims.get(modality_name)
            if current is None:
                modality_dims[modality_name] = novice_dim
            elif current != novice_dim:
                raise RuntimeError(
                    f"Modality {modality_name!r} has inconsistent dimensions across turns: {current} vs {novice_dim}"
                )
    return combo_name, modality_order, modality_dims


def build_model(args: argparse.Namespace, modality_dims: dict[str, int]) -> torch.nn.Module:
    return MultimodalTurnTCNRegressor(
        modality_dims=modality_dims,
        backbone_model=args.backbone,
        fusion_channels=args.fusion_channels,
        fusion_mode=args.fusion_mode,
        modality_dropout=args.modality_dropout,
        hidden_channels=args.hidden_channels,
        levels=args.levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        attention_context=args.attention_context,
        attention_heads=args.attention_heads,
        attention_past_frames=args.attention_past_frames,
        exclude_current_frame=args.exclude_current_frame,
        metadata_dim=getattr(args, "metadata_dim", 0),
        metadata_injection=args.metadata_injection,
        metadata_embedding_dim=args.metadata_embedding_dim,
        metadata_dropout=args.metadata_dropout,
        causal_tcn=args.causal_tcn,
    )


def masked_role_mean_std_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mean_weight: float,
    std_weight: float,
) -> torch.Tensor:
    loss = pred.new_tensor(0.0)
    if mean_weight <= 0.0 and std_weight <= 0.0:
        return loss
    for channel in range(pred.shape[-1]):
        valid = mask[..., channel] > 0
        if int(valid.sum().item()) < 2:
            continue
        pred_values = pred[..., channel][valid]
        target_values = target[..., channel][valid]
        if mean_weight > 0.0:
            loss = loss + mean_weight * (pred_values.mean() - target_values.mean()).pow(2)
        if std_weight > 0.0:
            pred_std = pred_values.std(unbiased=False)
            target_std = target_values.std(unbiased=False)
            loss = loss + std_weight * (pred_std - target_std).pow(2)
    return loss


def extreme_weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    low_threshold: float,
    high_threshold: float,
    extreme_weight: float,
) -> torch.Tensor:
    valid = mask > 0
    if int(valid.sum().item()) == 0:
        return pred.new_tensor(0.0)
    weights = torch.ones_like(target)
    extreme = (target < low_threshold) | (target > high_threshold)
    weights = torch.where(extreme, weights * extreme_weight, weights)
    squared_error = (pred - target).pow(2)
    numerator = (squared_error * weights * valid.to(weights.dtype)).sum()
    denominator = (weights * valid.to(weights.dtype)).sum().clamp_min(1.0)
    return numerator / denominator


def target_bin_mean_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bins = ((0.0, 0.3), (0.3, 0.8), (0.8, 1.000001))
    loss = pred.new_tensor(0.0)
    count = 0
    for channel in range(pred.shape[-1]):
        channel_target = target[..., channel]
        channel_mask = mask[..., channel] > 0
        for low, high in bins:
            valid = channel_mask & (channel_target >= low) & (channel_target < high)
            if int(valid.sum().item()) < 2:
                continue
            pred_mean = pred[..., channel][valid].mean()
            target_mean = channel_target[valid].mean()
            loss = loss + (pred_mean - target_mean).pow(2)
            count += 1
    if count == 0:
        return pred.new_tensor(0.0)
    return loss / count


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ccc_weight: float,
    mse_weight: float = 0.0,
    role_mean_calibration_weight: float = 0.0,
    role_std_calibration_weight: float = 0.0,
    extreme_mse_weight: float = 0.0,
    extreme_low_threshold: float = 0.3,
    extreme_high_threshold: float = 0.8,
    extreme_weight: float = 1.0,
    bin_mean_loss_weight: float = 0.0,
    progress_every: int = 0,
) -> float:
    model.train()
    losses: list[float] = []
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        if progress_every > 0 and (batch_idx == 1 or batch_idx % progress_every == 0):
            cache_size = len(getattr(loader.dataset, "_cache", {}))
            print(
                f"train_batch_start batch={batch_idx}/{total_batches} max_turn_len={int(batch['frame_mask'].shape[1])} "
                f"dataset_cache_sessions={cache_size}",
                flush=True,
            )
        x_modalities = move_modalities_to_device(batch["x_modalities"], device)
        y = batch["y"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_modalities, metadata=metadata_to_device(batch, device))
        loss = mse_weight * masked_mse_loss(pred, y, loss_mask) + ccc_weight * ccc_loss(pred, y, loss_mask)
        if role_mean_calibration_weight > 0.0 or role_std_calibration_weight > 0.0:
            loss = loss + masked_role_mean_std_loss(
                pred,
                y,
                loss_mask,
                role_mean_calibration_weight,
                role_std_calibration_weight,
            )
        if extreme_mse_weight > 0.0:
            loss = loss + extreme_mse_weight * extreme_weighted_mse_loss(
                pred,
                y,
                loss_mask,
                extreme_low_threshold,
                extreme_high_threshold,
                extreme_weight,
            )
        if bin_mean_loss_weight > 0.0:
            loss = loss + bin_mean_loss_weight * target_bin_mean_loss(pred, y, loss_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if progress_every > 0 and (batch_idx == 1 or batch_idx % progress_every == 0):
            print(f"train_batch_done batch={batch_idx}/{total_batches} loss={losses[-1]:.5f}", flush=True)
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def reconstruct_validation(
    model: torch.nn.Module,
    dataset: MultimodalTurnDataset,
    loader: DataLoader,
    device: torch.device,
    collect_gate_weights: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model.eval()

    session_info: dict[str, dict[str, object]] = {}
    for turn in dataset.turns:
        key = turn.session_key
        if key in session_info:
            continue
        reference_modality = turn.modality_order[0]
        novice_example = turn.novice_examples[reference_modality]
        expert_example = turn.expert_examples[reference_modality]
        session_len = min(novice_example.aligned_len, expert_example.aligned_len)
        session_info[key] = {
            "aligned_len": session_len,
            "novice_example": novice_example,
            "expert_example": expert_example,
            "dataset": turn.dataset,
            "session_id": turn.session_id,
            "model_split": turn.model_split,
        }

    sums: dict[str, np.ndarray] = {key: np.zeros((info["aligned_len"], 2), dtype=np.float64) for key, info in session_info.items()}
    counts: dict[str, np.ndarray] = {key: np.zeros(info["aligned_len"], dtype=np.float64) for key, info in session_info.items()}

    gate_sums: dict[str, np.ndarray] = {}
    gate_counts: dict[str, float] = {}
    modality_order: list[str] = []

    for batch in loader:
        x_modalities = move_modalities_to_device(batch["x_modalities"], device)
        if collect_gate_weights:
            pred_tensor, gate_info = model(x_modalities, metadata=metadata_to_device(batch, device), return_gate_weights=True)
        else:
            pred_tensor = model(x_modalities, metadata=metadata_to_device(batch, device))
            gate_info = None

        pred = pred_tensor.detach().cpu().numpy()
        frame_mask = batch["frame_mask"].numpy()
        session_keys = batch["session_keys"]
        start_frames = batch["start_frames"].numpy()

        if collect_gate_weights and gate_info is not None and gate_info.get("novice_weights") is not None:
            modality_order = list(gate_info["modality_order"])
            novice_weights = gate_info["novice_weights"].detach().cpu().numpy()
            expert_weights = gate_info["expert_weights"].detach().cpu().numpy()
            if not gate_sums:
                gate_sums = {
                    "novice": np.zeros(len(modality_order), dtype=np.float64),
                    "expert": np.zeros(len(modality_order), dtype=np.float64),
                }
                gate_counts = {"novice": 0.0, "expert": 0.0}

            for row in range(pred.shape[0]):
                valid_len = int(frame_mask[row].sum())
                if valid_len <= 0:
                    continue
                gate_sums["novice"] += novice_weights[row, :valid_len].sum(axis=0)
                gate_sums["expert"] += expert_weights[row, :valid_len].sum(axis=0)
                gate_counts["novice"] += valid_len
                gate_counts["expert"] += valid_len

        for row in range(pred.shape[0]):
            key = session_keys[row]
            start = int(start_frames[row])
            valid_len = int(frame_mask[row].sum())
            if valid_len <= 0:
                continue
            end = start + valid_len
            sums[key][start:end] += pred[row, :valid_len]
            counts[key][start:end] += 1.0

    reconstructed: list[dict[str, object]] = []
    for key, info in session_info.items():
        aligned_len = int(info["aligned_len"])
        novice_session = dataset._load(info["novice_example"])
        expert_session = dataset._load(info["expert_example"])

        y_true = np.stack([novice_session.y[:aligned_len], expert_session.y[:aligned_len]], axis=1)
        target_mask = np.stack(
            [novice_session.target_mask[:aligned_len], expert_session.target_mask[:aligned_len]],
            axis=1,
        )

        y_pred = np.full((aligned_len, 2), np.nan, dtype=np.float32)
        covered = counts[key] > 0
        if np.any(covered):
            y_pred[covered] = (sums[key][covered] / counts[key][covered, None]).astype(np.float32)

        reconstructed.append(
            {
                "example": _SessionStub(
                    dataset=str(info["dataset"]),
                    session_id=str(info["session_id"]),
                    model_split=str(info["model_split"]),
                ),
                "y_true": y_true,
                "target_mask": target_mask,
                "y_pred": y_pred,
                "covered": covered.astype(np.float32),
            }
        )

    gate_rows: list[dict[str, object]] = []
    if gate_sums and modality_order:
        for role in ("novice", "expert"):
            denominator = gate_counts[role] if gate_counts[role] > 0 else 1.0
            for idx, modality_name in enumerate(modality_order):
                gate_rows.append(
                    {
                        "role": role,
                        "modality": modality_name,
                        "mean_gate_weight": gate_sums[role][idx] / denominator,
                    }
                )

    return reconstructed, gate_rows


class _SessionStub:
    def __init__(self, dataset: str, session_id: str, model_split: str, role_names: tuple[str, ...] = ("novice", "expert")) -> None:
        self.dataset = dataset
        self.session_id = session_id
        self.model_split = model_split
        self.role_names = role_names


def main() -> None:
    args = parse_args()
    if args.min_epochs < 0:
        raise ValueError("--min-epochs must be non-negative.")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be non-negative.")
    nonnegative_loss_args = {
        "--role-mean-calibration-weight": args.role_mean_calibration_weight,
        "--role-std-calibration-weight": args.role_std_calibration_weight,
        "--extreme-mse-weight": args.extreme_mse_weight,
        "--bin-mean-loss-weight": args.bin_mean_loss_weight,
    }
    for name, value in nonnegative_loss_args.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
    if args.extreme_weight < 1.0:
        raise ValueError("--extreme-weight must be at least 1.0.")
    if args.extreme_low_threshold >= args.extreme_high_threshold:
        raise ValueError("--extreme-low-threshold must be less than --extreme-high-threshold.")
    if args.metadata_injection == "none" and args.metadata_set != "none":
        raise ValueError("--metadata-set must be none when --metadata-injection is none.")
    if args.metadata_injection != "none" and args.metadata_set == "none":
        raise ValueError("--metadata-set is required when metadata injection is enabled.")
    if args.metadata_injection != "none" and not args.metadata:
        raise ValueError("--metadata CSV path(s) are required when metadata injection is enabled.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = make_run_dir(args)

    print(f"startup run_dir={run_dir}", flush=True)
    print(f"startup read_manifest train split={args.train_split}", flush=True)
    train_turns = read_multimodal_turn_manifest(args.manifest, PROJECT_ROOT, split=args.train_split)
    print(f"startup read_manifest val split={args.val_split}", flush=True)
    val_turns = read_multimodal_turn_manifest(args.manifest, PROJECT_ROOT, split=args.val_split)
    if not train_turns or not val_turns:
        raise RuntimeError("Both train and validation multimodal turn rows are required.")
    print(f"startup manifest_counts train_turns={len(train_turns)} val_turns={len(val_turns)}", flush=True)

    combo_name, modality_order, modality_dims = infer_layout(train_turns + val_turns)
    print(f"startup layout combo={combo_name} modalities={list(modality_order)} dims={modality_dims}", flush=True)
    metadata_stats_payload = None
    if args.metadata_injection != "none":
        metadata_table = read_role_metadata(args.metadata)
        stats = metadata_stats(train_turns, metadata_table)
        args.metadata_dim = int(
            encode_role_metadata("noxi", "novice", {}, stats, args.metadata_set).shape[0]
        )
        metadata_stats_payload = {"age_mean": stats.age_mean, "age_std": stats.age_std, "languages": list(stats.languages)}
        train_dataset = MetadataMultimodalTurnDataset(train_turns, metadata_table, stats, args.metadata_set, min_frames=args.min_turn_frames)
        val_dataset = MetadataMultimodalTurnDataset(val_turns, metadata_table, stats, args.metadata_set, min_frames=args.min_turn_frames)
        print(f"startup metadata set={args.metadata_set} dim={args.metadata_dim} stats={metadata_stats_payload}", flush=True)
    else:
        args.metadata_dim = 0
        train_dataset = MultimodalTurnDataset(train_turns, min_frames=args.min_turn_frames)
        val_dataset = MultimodalTurnDataset(val_turns, min_frames=args.min_turn_frames)
    if args.preload_tensors:
        print("startup preload_tensors train", flush=True)
        train_dataset.preload()
        print(f"startup preload_tensors train_done cache_entries={len(train_dataset._cache)}", flush=True)
        print("startup preload_tensors val", flush=True)
        val_dataset.preload()
        print(f"startup preload_tensors val_done cache_entries={len(val_dataset._cache)}", flush=True)
    print(f"startup dataset_counts train={len(train_dataset)} val={len(val_dataset)}", flush=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=multimodal_turn_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=multimodal_turn_collate_fn,
    )

    config = serializable_args(args)
    config.update(
        {
            "run_dir": str(run_dir),
            "combo_name": combo_name,
            "modality_order": list(modality_order),
            "modality_dims": modality_dims,
            "n_modalities": len(modality_order),
            "output_dim": 2,
            "metadata_dim": args.metadata_dim,
            "metadata_stats": metadata_stats_payload,
            "n_train_turns": len(train_dataset),
            "n_val_turns": len(val_dataset),
        }
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"startup build_model device={device}", flush=True)
    model = build_model(args, modality_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("startup enter_training_loop", flush=True)

    best_ccc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    log_rows: list[dict[str, object]] = []

    if args.eval_only and not (run_dir / "model_best.pt").exists():
        raise FileNotFoundError(f"--eval-only requires an existing checkpoint: {run_dir / 'model_best.pt'}")

    epoch = 0
    val_ccc = float("nan")
    for epoch in range(1, args.epochs + 1):
        if args.eval_only:
            break
        print(f"epoch_start epoch={epoch:03d}", flush=True)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            ccc_weight=args.ccc_weight,
            mse_weight=args.mse_weight,
            role_mean_calibration_weight=args.role_mean_calibration_weight,
            role_std_calibration_weight=args.role_std_calibration_weight,
            extreme_mse_weight=args.extreme_mse_weight,
            extreme_low_threshold=args.extreme_low_threshold,
            extreme_high_threshold=args.extreme_high_threshold,
            extreme_weight=args.extreme_weight,
            bin_mean_loss_weight=args.bin_mean_loss_weight,
            progress_every=args.progress_every,
        )
        print(f"validation_start epoch={epoch:03d}", flush=True)
        reconstructed, _gate_rows = reconstruct_validation(model, val_dataset, val_loader, device, collect_gate_weights=False)
        val_metrics = grouped_dyadic_metric_outputs(run_dir, reconstructed)
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
        print(
            f"epoch={epoch:03d}  train_loss={train_loss:.5f}  val_ccc={val_ccc:.5f}  best_epoch={best_epoch}",
            flush=True,
        )
        if args.patience > 0 and epoch >= args.min_epochs and stale_epochs >= args.patience:
            print(
                f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d} "
                f"best_val_ccc={best_ccc:.5f} stale_epochs={stale_epochs} "
                f"patience={args.patience} min_delta={args.min_delta:.5f}",
                flush=True,
            )
            break

    if args.save_final_checkpoint:
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
            run_dir / "model_final.pt",
        )

    best_checkpoint_path = run_dir / "model_best.pt"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        reconstructed, gate_rows = reconstruct_validation(model, val_dataset, val_loader, device, collect_gate_weights=True)
        grouped_dyadic_metric_outputs(run_dir, reconstructed)
        write_dyadic_prediction_csv(run_dir / "val_predictions.csv", reconstructed)
        write_organizer_submission_tree(run_dir / "val_submission_format", reconstructed)
        if gate_rows:
            write_csv(run_dir / "val_gate_weights.csv", list(gate_rows[0].keys()), gate_rows)

        # Test-split inference: generate submission-format predictions for
        # held-out test sessions (labels withheld by organizers).
        for test_split in (args.test_splits or []):
            test_turns = read_multimodal_turn_manifest(args.manifest, PROJECT_ROOT, split=test_split)
            if not test_turns:
                continue
            if args.metadata_injection != "none":
                test_dataset = MetadataMultimodalTurnDataset(test_turns, metadata_table, stats, args.metadata_set, min_frames=args.min_turn_frames)
            else:
                test_dataset = MultimodalTurnDataset(test_turns, min_frames=args.min_turn_frames)
            if len(test_dataset) == 0:
                continue
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=multimodal_turn_collate_fn,
            )
            test_reconstructed, _ = reconstruct_validation(model, test_dataset, test_loader, device)
            write_organizer_submission_tree(run_dir / "test_submission_format", test_reconstructed)
            print(f"test_split={test_split}  sessions={len(test_reconstructed)}", flush=True)

    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()