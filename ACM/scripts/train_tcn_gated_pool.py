from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Gated pooled-context TCN trainer.
#
# This experiment uses two role-specific TCN encoders. For each prediction
# frame, the partner hidden sequence is mean-pooled over a past window, a
# learned gate decides how much of that partner context to use, and a
# role-specific head predicts engagement.
from src.acm_pipeline.dyadic_data import ROLE_ORDER, WindowedDyadicDataset, read_dyadic_manifest
from src.acm_pipeline.dyadic_train_utils import grouped_dyadic_metric_outputs, write_csv, write_dyadic_prediction_csv
from src.acm_pipeline.metrics import ccc_loss, masked_mse_loss
from src.acm_pipeline.models_tcn import GatedPooledTCNRegressor


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a gated pooled-partner-context TCN from a dyadic manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--train-split", default="train_internal")
    parser.add_argument("--val-split", default="val_internal")
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--stride", type=int, default=125)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ccc-weight", type=float, default=0.5)
    parser.add_argument("--partner-pool-frames", type=int, default=750, help="Partner past-context window in frames; 750=30s and 1500=60s at 25 Hz.")
    parser.add_argument("--include-current-frame", action="store_true", help="Include partner frame t in the pooled context. Default excludes it for interaction interpretation.")
    parser.add_argument("--gate-type", default="scalar", choices=["scalar", "channel"], help="scalar is easier to interpret; channel is more flexible.")
    parser.add_argument("--save-gates", action="store_true", help="Export gate diagnostics from the best checkpoint.")
    parser.add_argument("--gate-export-query-stride", type=int, default=125, help="Write sampled gate values every N frames.")
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
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
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def make_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name.strip() or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_tcn_gated_pool"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def save_config(args: argparse.Namespace, run_dir: Path, input_dim: int, n_features_per_role: int, n_train: int, n_val: int) -> None:
    payload = serializable_args(args)
    payload.update(
        {
            "run_dir": str(run_dir),
            "input_dim": input_dim,
            "n_features_per_role": n_features_per_role,
            "output_dim": 2,
            "n_train_examples": n_train,
            "n_val_examples": n_val,
        }
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def train_one_epoch(model: GatedPooledTCNRegressor, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, ccc_weight: float) -> float:
    model.train()
    losses = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)

        # The model predicts both role channels. Masks remove padded frames and
        # missing target values from the loss.
        loss = masked_mse_loss(pred, y, loss_mask) + ccc_weight * ccc_loss(pred, y, loss_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def update_gate_diagnostics(
    diagnostics: dict[str, object],
    gate_values: np.ndarray,
    role: str,
    dataset: str,
    session_id: str,
    session_len: int,
    window_start: int,
    valid_len: int,
    query_stride: int,
) -> None:
    """Aggregate gate values and keep a sampled time-series export."""

    # gate_values is [time] after averaging over gate channels if needed.
    role_summary = diagnostics["role_summary"]
    session_summary = diagnostics["session_summary"]
    phase_summary = diagnostics["phase_summary"]
    timeseries_rows = diagnostics["timeseries_rows"]
    query_stride = max(1, int(query_stride))

    for local_idx in range(valid_len):
        frame_idx = window_start + local_idx
        gate = float(gate_values[local_idx])
        phase_decile = min(9, int((frame_idx / max(1, session_len)) * 10))

        role_summary[role][0] += gate
        role_summary[role][1] += gate * gate
        role_summary[role][2] += 1

        session_key = (dataset, session_id, role)
        session_summary[session_key][0] += gate
        session_summary[session_key][1] += gate * gate
        session_summary[session_key][2] += 1

        phase_key = (role, phase_decile)
        phase_summary[phase_key][0] += gate
        phase_summary[phase_key][1] += gate * gate
        phase_summary[phase_key][2] += 1

        if local_idx % query_stride == 0:
            timeseries_rows.append(
                {
                    "dataset": dataset,
                    "session_id": session_id,
                    "role": role,
                    "frame_idx": frame_idx,
                    "time_seconds": frame_idx / 25.0,
                    "session_phase_decile": phase_decile,
                    "gate": gate,
                }
            )


def summary_row(prefix: dict[str, object], values: list[float]) -> dict[str, object]:
    """Convert sum/sumsq/count gate accumulators to mean/std rows."""

    total, total_sq, count = values
    mean = total / max(1, count)
    variance = max(0.0, total_sq / max(1, count) - mean * mean)
    return {**prefix, "n_frames": int(count), "mean_gate": mean, "std_gate": float(np.sqrt(variance))}


def write_gate_diagnostics(run_dir: Path, diagnostics: dict[str, object]) -> None:
    """Write gate summaries to CSV files."""

    role_rows = [summary_row({"role": role}, values) for role, values in sorted(diagnostics["role_summary"].items())]
    write_csv(run_dir / "gate_by_role.csv", ["role", "n_frames", "mean_gate", "std_gate"], role_rows)

    session_rows = [
        summary_row({"dataset": dataset, "session_id": session_id, "role": role}, values)
        for (dataset, session_id, role), values in sorted(diagnostics["session_summary"].items())
    ]
    write_csv(run_dir / "gate_by_session.csv", ["dataset", "session_id", "role", "n_frames", "mean_gate", "std_gate"], session_rows)

    phase_rows = [
        summary_row({"role": role, "session_phase_decile": phase}, values)
        for (role, phase), values in sorted(diagnostics["phase_summary"].items())
    ]
    write_csv(run_dir / "gate_by_session_phase.csv", ["role", "session_phase_decile", "n_frames", "mean_gate", "std_gate"], phase_rows)

    timeseries_rows = diagnostics["timeseries_rows"]
    fieldnames = ["dataset", "session_id", "role", "frame_idx", "time_seconds", "session_phase_decile", "gate"]
    if timeseries_rows:
        write_csv(run_dir / "gate_timeseries_sample.csv", fieldnames, timeseries_rows)
    else:
        with (run_dir / "gate_timeseries_sample.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(fieldnames)


@torch.no_grad()
def reconstruct_validation(
    model: GatedPooledTCNRegressor,
    dataset: WindowedDyadicDataset,
    loader: DataLoader,
    device: torch.device,
    collect_gates: bool = False,
    gate_query_stride: int = 125,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    model.eval()
    sums = [np.zeros((example.aligned_len, 2), dtype=np.float64) for example in dataset.examples]
    counts = [np.zeros(example.aligned_len, dtype=np.float64) for example in dataset.examples]
    diagnostics: dict[str, object] | None = None
    if collect_gates:
        diagnostics = {
            "role_summary": defaultdict(lambda: [0.0, 0.0, 0]),
            "session_summary": defaultdict(lambda: [0.0, 0.0, 0]),
            "phase_summary": defaultdict(lambda: [0.0, 0.0, 0]),
            "timeseries_rows": [],
        }

    for batch in loader:
        if collect_gates:
            pred_tensor, gate_payload = model(batch["x"].to(device), return_gates=True)
        else:
            pred_tensor = model(batch["x"].to(device))
            gate_payload = None
        pred = pred_tensor.detach().cpu().numpy()
        frame_mask = batch["frame_mask"].numpy()
        example_idx = batch["example_idx"].numpy()
        starts = batch["start"].numpy()

        if collect_gates and gate_payload is not None and diagnostics is not None:
            novice_gate = gate_payload["novice_gate"].detach().cpu().numpy().mean(axis=1)
            expert_gate = gate_payload["expert_gate"].detach().cpu().numpy().mean(axis=1)
            for row_idx in range(pred.shape[0]):
                ex_idx = int(example_idx[row_idx])
                example = dataset.examples[ex_idx]
                valid_len = int(frame_mask[row_idx].sum())
                if valid_len <= 0:
                    continue
                update_gate_diagnostics(
                    diagnostics,
                    novice_gate[row_idx],
                    ROLE_ORDER[0],
                    example.dataset,
                    example.session_id,
                    example.aligned_len,
                    int(starts[row_idx]),
                    valid_len,
                    gate_query_stride,
                )
                update_gate_diagnostics(
                    diagnostics,
                    expert_gate[row_idx],
                    ROLE_ORDER[1],
                    example.dataset,
                    example.session_id,
                    example.aligned_len,
                    int(starts[row_idx]),
                    valid_len,
                    gate_query_stride,
                )

        for row_idx in range(pred.shape[0]):
            ex_idx = int(example_idx[row_idx])
            start = int(starts[row_idx])
            valid_len = int(frame_mask[row_idx].sum())
            if valid_len <= 0:
                continue
            end = start + valid_len
            sums[ex_idx][start:end] += pred[row_idx, :valid_len]
            counts[ex_idx][start:end] += 1.0

    reconstructed: list[dict[str, object]] = []
    for ex_idx, example in enumerate(dataset.examples):
        session = dataset.load_session(ex_idx)
        pred = np.full((example.aligned_len, 2), np.nan, dtype=np.float32)
        covered = counts[ex_idx] > 0
        pred[covered] = (sums[ex_idx][covered] / counts[ex_idx][covered, None]).astype(np.float32)
        reconstructed.append(
            {
                "example": example,
                "y_true": session.y[: example.aligned_len],
                "target_mask": session.target_mask[: example.aligned_len],
                "y_pred": pred,
                "covered": covered.astype(np.float32),
            }
        )
    return reconstructed, diagnostics


def main() -> None:
    args = parse_args()
    if args.min_epochs < 0:
        raise ValueError("--min-epochs must be non-negative.")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be non-negative.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = make_run_dir(args)

    train_examples = read_dyadic_manifest(args.manifest, PROJECT_ROOT, split=args.train_split)
    val_examples = read_dyadic_manifest(args.manifest, PROJECT_ROOT, split=args.val_split)
    if not train_examples or not val_examples:
        raise RuntimeError("Both train and validation examples are required.")

    input_dims = sorted({example.n_features for example in train_examples + val_examples})
    per_role_dims = sorted({example.n_features_per_role for example in train_examples + val_examples})
    if len(input_dims) != 1 or len(per_role_dims) != 1:
        raise RuntimeError(f"Expected fixed input dimensions, got input={input_dims}, per_role={per_role_dims}")
    if input_dims[0] != 2 * per_role_dims[0]:
        raise RuntimeError(f"Expected dyadic input_dim == 2 * per_role_dim, got {input_dims[0]} and {per_role_dims[0]}")

    input_dim = input_dims[0]
    n_features_per_role = per_role_dims[0]
    save_config(args, run_dir, input_dim=input_dim, n_features_per_role=n_features_per_role, n_train=len(train_examples), n_val=len(val_examples))

    train_dataset = WindowedDyadicDataset(train_examples, args.window_size, args.stride, max_windows=args.max_train_windows, seed=args.seed)
    val_dataset = WindowedDyadicDataset(val_examples, args.window_size, args.stride)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = GatedPooledTCNRegressor(
        n_features_per_role=n_features_per_role,
        hidden_channels=args.hidden_channels,
        levels=args.levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        partner_pool_frames=args.partner_pool_frames,
        exclude_current_frame=not args.include_current_frame,
        gate_type=args.gate_type,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ccc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    log_rows: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, ccc_weight=args.ccc_weight)
        reconstructed, _ = reconstruct_validation(model, val_dataset, val_loader, device, collect_gates=False)
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
            write_dyadic_prediction_csv(run_dir / "val_predictions.csv", reconstructed)
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
        reconstructed, diagnostics = reconstruct_validation(
            model,
            val_dataset,
            val_loader,
            device,
            collect_gates=args.save_gates,
            gate_query_stride=args.gate_export_query_stride,
        )
        grouped_dyadic_metric_outputs(run_dir, reconstructed)
        write_dyadic_prediction_csv(run_dir / "val_predictions.csv", reconstructed)
        if diagnostics is not None:
            write_gate_diagnostics(run_dir, diagnostics)

    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
