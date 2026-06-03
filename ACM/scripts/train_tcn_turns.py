"""Train turn-level TCN baselines from paired dyadic turn manifests.

Instead of fixed-size sliding windows, this script consumes precomputed
speech-turn rows and trains one of three active TCN variants on those
variable-length segments:

* ``simple``: one shared person-level TCN applied independently to novice and expert
* ``dyadic_shared``: one shared dyadic encoder over concatenated role inputs
* ``attention``: separate role encoders plus role-specific temporal attention

All variants predict engagement for both novice and expert at every frame.

Usage example
-------------
    python scripts/train_tcn_turns.py \
    --manifest outputs/manifests/model_processed_manifest_audio_egemaps_raw_turns.csv \
    --model attention \
        --epochs 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.dyadic_train_utils import grouped_dyadic_metric_outputs, write_csv, write_dyadic_prediction_csv
from src.acm_pipeline.metrics import ccc_loss, masked_mse_loss
from src.acm_pipeline.models_tcn import (
    DyadicTCNRegressor,
    IndependentDyadicTCNRegressor,
    RoleAttentionTCNRegressor,
)
from src.acm_pipeline.turn_data import TurnDataset, read_turn_manifest, turn_collate_fn

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train turn-segmented TCN models from paired turn supervision.")
    parser.add_argument("--manifest", type=Path, required=True, help="Paired turn manifest CSV.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--train-split", default="train_internal")
    parser.add_argument("--val-split", default="val_internal")

    # Model
    parser.add_argument("--model", choices=["simple", "dyadic_shared", "attention"], default="attention")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    # attention specific
    parser.add_argument("--attention-context", choices=["self", "partner", "joint"], default="joint")
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-past-frames", type=int, default=1500)
    parser.add_argument("--exclude-current-frame", action="store_true")

    # Training
    parser.add_argument("--min-turn-frames", type=int, default=5, help="Drop turns shorter than this many frames.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ccc-weight", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    run_name = args.run_name.strip() or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_tcn_turns_{args.model}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(args: argparse.Namespace, n_features_per_role: int) -> torch.nn.Module:
    common = dict(
        n_features_per_role=n_features_per_role,
        hidden_channels=args.hidden_channels,
        levels=args.levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    )
    if args.model == "simple":
        return IndependentDyadicTCNRegressor(**common)
    if args.model == "dyadic_shared":
        return DyadicTCNRegressor(
            input_dim=2 * n_features_per_role,
            hidden_channels=args.hidden_channels,
            levels=args.levels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        )
    if args.model == "attention":
        return RoleAttentionTCNRegressor(
            **common,
            attention_context=args.attention_context,
            attention_heads=args.attention_heads,
            attention_past_frames=args.attention_past_frames,
            exclude_current_frame=args.exclude_current_frame,
        )
    raise ValueError(f"Unknown model type: {args.model}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
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
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = masked_mse_loss(pred, y, loss_mask) + ccc_weight * ccc_loss(pred, y, loss_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


# ---------------------------------------------------------------------------
# Validation reconstruction
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_validation(
    model: torch.nn.Module,
    dataset: TurnDataset,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, object]]:
    """Reconstruct session-level predictions from non-overlapping turn predictions."""

    model.eval()

    # Collect (session_key → aligned_len) so we can allocate accumulators.
    session_info: dict[str, dict] = {}
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
            }

    # Pre-allocate per-session prediction and count buffers.
    sums: dict[str, np.ndarray] = {k: np.zeros((v["aligned_len"], 2), dtype=np.float64) for k, v in session_info.items()}
    counts: dict[str, np.ndarray] = {k: np.zeros(v["aligned_len"], dtype=np.float64) for k, v in session_info.items()}

    for batch in loader:
        pred = model(batch["x"].to(device)).detach().cpu().numpy()  # [B, T, 2]
        frame_mask = batch["frame_mask"].numpy()
        session_keys = batch["session_keys"]
        start_frames = batch["start_frames"].numpy()

        for row in range(pred.shape[0]):
            key = session_keys[row]
            start = int(start_frames[row])
            valid_len = int(frame_mask[row].sum())
            if valid_len <= 0:
                continue
            end = start + valid_len
            sums[key][start:end] += pred[row, :valid_len]
            counts[key][start:end] += 1.0

    # Build the reconstructed list in the same format expected by
    # grouped_dyadic_metric_outputs.
    reconstructed: list[dict[str, object]] = []
    for key, info in session_info.items():
        aligned_len = info["aligned_len"]
        novice_session = dataset._load(info["novice_example"])
        expert_session = dataset._load(info["expert_example"])

        y_true = np.stack(
            [novice_session.y[:aligned_len], expert_session.y[:aligned_len]], axis=1
        )  # [time, 2]
        target_mask = np.stack(
            [novice_session.target_mask[:aligned_len], expert_session.target_mask[:aligned_len]], axis=1
        )  # [time, 2]

        y_pred = np.full((aligned_len, 2), np.nan, dtype=np.float32)
        covered = counts[key] > 0
        if np.any(covered):
            y_pred[covered] = (sums[key][covered] / counts[key][covered, None]).astype(np.float32)

        # grouped_dyadic_metric_outputs expects example.dataset and
        # example.session_id — we create a lightweight stand-in.
        example_stub = _SessionStub(dataset=info["dataset"], session_id=info["session_id"])
        reconstructed.append({
            "example": example_stub,
            "y_true": y_true,
            "target_mask": target_mask,
            "y_pred": y_pred,
            "covered": covered.astype(np.float32),
        })

    return reconstructed


class _SessionStub:
    """Lightweight stand-in for DyadicManifestExample used by metric utilities."""

    def __init__(self, dataset: str, session_id: str) -> None:
        self.dataset = dataset
        self.session_id = session_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.min_epochs < 0:
        raise ValueError("--min-epochs must be non-negative.")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be non-negative.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = make_run_dir(args)

    train_turns = read_turn_manifest(args.manifest, PROJECT_ROOT, split=args.train_split)
    val_turns = read_turn_manifest(args.manifest, PROJECT_ROOT, split=args.val_split)
    feature_dims = sorted(
        {
            turn.novice_example.n_features
            for turn in train_turns + val_turns
        }
        | {
            turn.expert_example.n_features
            for turn in train_turns + val_turns
        }
    )

    if not train_turns or not val_turns:
        raise RuntimeError("Both train and validation turn rows are required.")
    if len(feature_dims) != 1:
        raise RuntimeError(f"Expected one fixed feature dimension, got {feature_dims}")
    n_features_per_role = feature_dims[0]

    train_dataset = TurnDataset(train_turns, min_frames=args.min_turn_frames)
    val_dataset = TurnDataset(val_turns, min_frames=args.min_turn_frames)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=turn_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=turn_collate_fn,
    )

    # Save config.
    config = serializable_args(args)
    config.update({
        "run_dir": str(run_dir),
        "n_features_per_role": n_features_per_role,
        "output_dim": 2,
        "n_train_turns": len(train_dataset),
        "n_val_turns": len(val_dataset),
    })
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    model = build_model(args, n_features_per_role).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ccc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    log_rows: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, ccc_weight=args.ccc_weight)
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

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_ccc": val_metrics["ccc"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_pearson": val_metrics["pearson"],
            "best_epoch": best_epoch,
            "best_val_ccc": best_ccc,
            "stale_epochs": stale_epochs,
        })
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

    best_checkpoint_path = run_dir / "model_best.pt"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        reconstructed = reconstruct_validation(model, val_dataset, val_loader, device)
        grouped_dyadic_metric_outputs(run_dir, reconstructed)
        write_dyadic_prediction_csv(run_dir / "val_predictions.csv", reconstructed)

    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
