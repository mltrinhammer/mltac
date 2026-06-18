"""Train NOXI dyadic TCN regression experts with role metadata before heads."""

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
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_tcn_turns import _SessionStub, serializable_args
from src.acm_pipeline.dyadic_train_utils import (
    grouped_dyadic_metric_outputs,
    write_csv,
    write_dyadic_prediction_csv,
    write_organizer_submission_tree,
)
from src.acm_pipeline.metrics import ccc_loss, masked_mse_loss
from src.acm_pipeline.models_tcn import TemporalBlock
from src.acm_pipeline.turn_data import ManifestTurnSample, TurnDataset, read_turn_manifest


ROLES = ("novice", "expert")
METADATA_FIELDS = ("age_z", "gender_1", "gender_2", "gender_unknown", "language_known")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NOXI dyadic TCN regression with metadata heads.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-mode", choices=("age_gender_language", "age_gender", "language_only"), default="age_gender_language")
    parser.add_argument("--metadata-dropout", type=float, default=0.2)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "MoE" / "experiments")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--train-split", default="train_internal")
    parser.add_argument("--val-split", default="val_internal")
    parser.add_argument("--test-splits", nargs="*", default=["test_internal", "test_additional"])
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--causal-tcn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-turn-frames", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--min-epochs", type=int, default=24)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ccc-weight", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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


def metadata_stats(turns: list[ManifestTurnSample], table: dict[tuple[str, str, str], dict[str, str]]) -> MetadataStats:
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
    if not ages:
        raise RuntimeError("No age metadata found in train split.")
    mean = float(np.mean(ages))
    std = float(np.std(ages))
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


class MetadataTurnDataset(TurnDataset):
    def __init__(
        self,
        turns: list[ManifestTurnSample],
        metadata_table: dict[tuple[str, str, str], dict[str, str]],
        stats: MetadataStats,
        metadata_mode: str,
        min_frames: int = 5,
    ) -> None:
        super().__init__(turns, min_frames=min_frames)
        self.metadata_table = metadata_table
        self.stats = stats
        self.metadata_mode = metadata_mode

    def __getitem__(self, idx: int) -> dict[str, object]:
        item = super().__getitem__(idx)
        turn = self.turns[idx]
        item["metadata"] = torch.from_numpy(
            np.stack(
                [
                    encode_metadata(self.metadata_table.get((turn.dataset, turn.session_id, role), {}), self.stats, self.metadata_mode)
                    for role in ROLES
                ],
                axis=0,
            )
        )
        return item


def metadata_collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    max_len = max(int(item["turn_len"]) for item in batch)
    batch_size = len(batch)
    n_features = batch[0]["x_novice"].shape[0]
    metadata_dim = batch[0]["metadata"].shape[-1]

    x = torch.zeros(batch_size, 2 * n_features, max_len, dtype=torch.float32)
    y = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)
    target_mask = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, max_len, dtype=torch.float32)
    metadata = torch.zeros(batch_size, 2, metadata_dim, dtype=torch.float32)
    session_keys: list[str] = []
    start_frames = torch.zeros(batch_size, dtype=torch.long)
    turn_lens = torch.zeros(batch_size, dtype=torch.long)

    for idx, item in enumerate(batch):
        length = int(item["turn_len"])
        x[idx, :n_features, :length] = item["x_novice"]
        x[idx, n_features:, :length] = item["x_expert"]
        y[idx, :length] = item["y"]
        target_mask[idx, :length] = item["target_mask"]
        frame_mask[idx, :length] = 1.0
        metadata[idx] = item["metadata"]
        session_keys.append(str(item["session_key"]))
        start_frames[idx] = int(item["start_frame"])
        turn_lens[idx] = length

    return {
        "x": x,
        "y": y,
        "target_mask": target_mask,
        "frame_mask": frame_mask,
        "loss_mask": target_mask * frame_mask.unsqueeze(-1),
        "metadata": metadata,
        "session_keys": session_keys,
        "start_frames": start_frames,
        "turn_lens": turn_lens,
    }


class NoxiDyadicMetadataHeadTCN(nn.Module):
    def __init__(
        self,
        n_features_per_role: int,
        metadata_dim: int,
        hidden_channels: int,
        levels: int,
        kernel_size: int,
        dropout: float,
        metadata_dropout: float,
        causal_tcn: bool,
    ) -> None:
        super().__init__()
        channels = [2 * n_features_per_role] + [hidden_channels] * levels
        self.tcn = nn.Sequential(
            *[
                TemporalBlock(channels[idx], channels[idx + 1], kernel_size, 2**idx, dropout, causal=causal_tcn)
                for idx in range(levels)
            ]
        )
        self.metadata_dropout = nn.Dropout(metadata_dropout)
        self.head = nn.Conv1d(hidden_channels + 2 * metadata_dim, 2, kernel_size=1)

    def forward(self, x: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        hidden = self.tcn(x)
        batch, _channels, time = hidden.shape
        meta = self.metadata_dropout(metadata.reshape(batch, -1)).unsqueeze(-1).expand(-1, -1, time)
        return self.head(torch.cat([hidden, meta], dim=1)).transpose(1, 2)


def build_model(args: argparse.Namespace, n_features_per_role: int, metadata_dim: int) -> nn.Module:
    return NoxiDyadicMetadataHeadTCN(
        n_features_per_role=n_features_per_role,
        metadata_dim=metadata_dim,
        hidden_channels=args.hidden_channels,
        levels=args.levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        metadata_dropout=args.metadata_dropout,
        causal_tcn=bool(args.causal_tcn),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ccc_weight: float,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        metadata = batch["metadata"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, metadata)
        loss = masked_mse_loss(pred, y, loss_mask) + ccc_weight * ccc_loss(pred, y, loss_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def reconstruct_validation(
    model: nn.Module,
    dataset: MetadataTurnDataset,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    session_info: dict[str, dict[str, object]] = {}
    for turn in dataset.turns:
        key = turn.session_key
        if key not in session_info:
            session_len = min(turn.novice_example.aligned_len, turn.expert_example.aligned_len)
            session_info[key] = {
                "aligned_len": session_len,
                "novice_example": turn.novice_example,
                "expert_example": turn.expert_example,
                "dataset": turn.dataset,
                "session_id": turn.session_id,
                "model_split": turn.novice_example.model_split,
            }
    sums = {key: np.zeros((int(value["aligned_len"]), 2), dtype=np.float64) for key, value in session_info.items()}
    counts = {key: np.zeros(int(value["aligned_len"]), dtype=np.float64) for key, value in session_info.items()}

    for batch in loader:
        pred = model(batch["x"].to(device), batch["metadata"].to(device)).detach().cpu().numpy()
        frame_mask = batch["frame_mask"].numpy()
        for row, key in enumerate(batch["session_keys"]):
            start = int(batch["start_frames"][row])
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
            [novice_session.target_mask[:aligned_len], expert_session.target_mask[:aligned_len]], axis=1
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
    return reconstructed


def run_dir_for(args: argparse.Namespace) -> Path:
    name = args.run_name.strip() or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_noxi_metadata_head"
    path = args.output_root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = run_dir_for(args)

    metadata_table = read_metadata(args.metadata)
    train_turns = read_turn_manifest(args.manifest, PROJECT_ROOT, split=args.train_split)
    val_turns = read_turn_manifest(args.manifest, PROJECT_ROOT, split=args.val_split)
    if not train_turns or not val_turns:
        raise RuntimeError("Both train and validation turn rows are required.")
    feature_dims = sorted(
        {turn.novice_example.n_features for turn in train_turns + val_turns}
        | {turn.expert_example.n_features for turn in train_turns + val_turns}
    )
    if len(feature_dims) != 1:
        raise RuntimeError(f"Expected one fixed feature dimension, got {feature_dims}")
    n_features_per_role = feature_dims[0]
    stats = metadata_stats(train_turns, metadata_table)
    metadata_dim = int(encode_metadata({}, stats, args.metadata_mode).shape[0])

    train_dataset = MetadataTurnDataset(train_turns, metadata_table, stats, args.metadata_mode, args.min_turn_frames)
    val_dataset = MetadataTurnDataset(val_turns, metadata_table, stats, args.metadata_mode, args.min_turn_frames)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=metadata_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=metadata_collate_fn,
    )

    config = serializable_args(args)
    config.update(
        {
            "model": "dyadic_metadata_head",
            "run_dir": str(run_dir),
            "n_features_per_role": n_features_per_role,
            "output_dim": 2,
            "metadata_dim": metadata_dim,
            "metadata_stats": {"age_mean": stats.age_mean, "age_std": stats.age_std, "languages": list(stats.languages)},
            "n_train_turns": len(train_dataset),
            "n_val_turns": len(val_dataset),
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model = build_model(args, n_features_per_role, metadata_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_ccc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    log_rows: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.ccc_weight)
        reconstructed = reconstruct_validation(model, val_dataset, val_loader, device)
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

    best_checkpoint_path = run_dir / "model_best.pt"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        reconstructed = reconstruct_validation(model, val_dataset, val_loader, device)
        grouped_dyadic_metric_outputs(run_dir, reconstructed)
        write_dyadic_prediction_csv(run_dir / "val_predictions.csv", reconstructed)
        write_organizer_submission_tree(run_dir / "val_submission_format", reconstructed)
        for test_split in args.test_splits or []:
            test_turns = read_turn_manifest(args.manifest, PROJECT_ROOT, split=test_split)
            if not test_turns:
                continue
            test_dataset = MetadataTurnDataset(test_turns, metadata_table, stats, args.metadata_mode, args.min_turn_frames)
            if len(test_dataset) == 0:
                continue
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=metadata_collate_fn,
            )
            test_reconstructed = reconstruct_validation(model, test_dataset, test_loader, device)
            write_organizer_submission_tree(run_dir / "test_submission_format", test_reconstructed)
            print(f"test_split={test_split} sessions={len(test_reconstructed)}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
