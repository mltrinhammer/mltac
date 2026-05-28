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

# Attention TCN trainer.
#
# This experiment keeps separate role TCN encoders, then uses role-specific
# attention heads. The attention context can be self-only, partner-only, or a
# joint self+partner history window.
from src.acm_pipeline.dyadic_data import ROLE_ORDER, WindowedDyadicDataset, read_dyadic_manifest
from src.acm_pipeline.dyadic_train_utils import grouped_dyadic_metric_outputs, write_csv, write_dyadic_prediction_csv
from src.acm_pipeline.metrics import ccc_loss, masked_mse_loss
from src.acm_pipeline.models_tcn import RoleAttentionTCNRegressor


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a role-specific TCN with self/partner/joint attention heads.")
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ccc-weight", type=float, default=0.5)
    parser.add_argument("--attention-context", default="joint", choices=["self", "partner", "joint"])
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-past-frames", type=int, default=1500, help="Past context length in frames; use -1 for all available past frames in the window.")
    parser.add_argument("--exclude-current-frame", action="store_true", help="Prevent attention from using source frame t for query frame t when possible.")
    parser.add_argument("--save-attention", action="store_true", help="Export attention diagnostics for the best validation epoch.")
    parser.add_argument("--attention-export-topk", type=int, default=10)
    parser.add_argument("--attention-export-query-stride", type=int, default=125, help="Export top-k attention for every Nth query frame.")
    parser.add_argument("--attention-summary-bins", type=int, nargs="+", default=[0, 25, 75, 250, 750, 1500])
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
    run_name = args.run_name.strip() or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_tcn_attention"
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


def attention_past_arg(value: int) -> int | None:
    return None if value < 0 else value


def train_one_epoch(model: RoleAttentionTCNRegressor, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, ccc_weight: float) -> float:
    model.train()
    losses = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)

        # Output and targets are [batch, time, 2]. The target mask keeps
        # missing annotations and padded tail frames out of the objective.
        loss = masked_mse_loss(pred, y, loss_mask) + ccc_weight * ccc_loss(pred, y, loss_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def lag_bin(abs_lag: int, bins: list[int]) -> str:
    """Place an absolute lag in a readable frame-bin label."""

    ordered = sorted(set(int(value) for value in bins if value >= 0))
    if not ordered:
        return "all"
    previous = 0
    for boundary in ordered:
        if abs_lag <= boundary:
            return f"{previous}-{boundary}"
        previous = boundary + 1
    return f">{ordered[-1]}"


def update_attention_diagnostics(
    diagnostics: dict[str, object],
    weights: np.ndarray,
    source_blocks: list[str],
    role: str,
    dataset: str,
    session_id: str,
    session_len: int,
    window_start: int,
    valid_len: int,
    topk: int,
    query_stride: int,
    bins: list[int],
) -> None:
    """Aggregate attention weights and keep a sampled top-k export."""

    # weights is [query_time, source_blocks * window_time]. Source block labels
    # are relative to the predicted role: "self" or "partner".
    lag_summary = diagnostics["lag_summary"]
    source_summary = diagnostics["source_summary"]
    bin_summary = diagnostics["bin_summary"]
    phase_summary = diagnostics["phase_summary"]
    topk_rows = diagnostics["topk_rows"]
    time_len = weights.shape[0]
    source_count = len(source_blocks)
    query_stride = max(1, int(query_stride))
    topk = max(0, int(topk))

    for query_local in range(0, valid_len, query_stride):
        query_frame = window_start + query_local
        row = weights[query_local]

        # Top-k rows are sampled for interpretability. Aggregates below use all
        # available source positions for the sampled query frames.
        if topk > 0:
            k = min(topk, row.size)
            top_indices = np.argpartition(row, -k)[-k:]
            top_indices = top_indices[np.argsort(row[top_indices])[::-1]]
        else:
            top_indices = np.empty(0, dtype=np.int64)

        for source_index, weight in enumerate(row):
            source_block = int(source_index // time_len)
            source_local = int(source_index % time_len)
            if source_block >= source_count or source_local >= valid_len:
                continue
            source = source_blocks[source_block]
            source_frame = window_start + source_local
            relative_lag = source_frame - query_frame
            abs_lag = abs(relative_lag)
            weight_value = float(weight)
            phase_decile = min(9, int((source_frame / max(1, session_len)) * 10))

            lag_key = (role, source, relative_lag)
            lag_summary[lag_key][0] += weight_value
            lag_summary[lag_key][1] += 1

            source_key = (role, source)
            source_summary[source_key][0] += weight_value
            source_summary[source_key][1] += 1

            bin_key = (role, source, lag_bin(abs_lag, bins))
            bin_summary[bin_key][0] += weight_value
            bin_summary[bin_key][1] += 1

            phase_key = (role, source, phase_decile)
            phase_summary[phase_key][0] += weight_value
            phase_summary[phase_key][1] += 1

        for source_index in top_indices:
            source_block = int(source_index // time_len)
            source_local = int(source_index % time_len)
            if source_block >= source_count or source_local >= valid_len:
                continue
            source = source_blocks[source_block]
            source_frame = window_start + source_local
            relative_lag = source_frame - query_frame
            phase_decile = min(9, int((source_frame / max(1, session_len)) * 10))
            topk_rows.append(
                {
                    "dataset": dataset,
                    "session_id": session_id,
                    "predicted_role": role,
                    "source": source,
                    "window_start": window_start,
                    "query_frame_idx": query_frame,
                    "source_frame_idx": source_frame,
                    "relative_lag_frames": relative_lag,
                    "relative_lag_seconds": relative_lag / 25.0,
                    "source_session_phase_decile": phase_decile,
                    "attention_weight": float(row[source_index]),
                }
            )


def write_attention_diagnostics(run_dir: Path, diagnostics: dict[str, object]) -> None:
    """Write attention summaries to CSV files."""

    lag_rows = [
        {
            "predicted_role": role,
            "source": source,
            "relative_lag_frames": lag,
            "relative_lag_seconds": lag / 25.0,
            "attention_sum": values[0],
            "n_observations": values[1],
            "mean_attention": values[0] / max(1, values[1]),
        }
        for (role, source, lag), values in sorted(diagnostics["lag_summary"].items())
    ]
    write_csv(run_dir / "attention_by_lag.csv", ["predicted_role", "source", "relative_lag_frames", "relative_lag_seconds", "attention_sum", "n_observations", "mean_attention"], lag_rows)

    source_rows = [
        {"predicted_role": role, "source": source, "attention_sum": values[0], "n_observations": values[1], "mean_attention": values[0] / max(1, values[1])}
        for (role, source), values in sorted(diagnostics["source_summary"].items())
    ]
    write_csv(run_dir / "attention_by_source.csv", ["predicted_role", "source", "attention_sum", "n_observations", "mean_attention"], source_rows)

    bin_rows = [
        {"predicted_role": role, "source": source, "lag_bin_frames": bin_label, "attention_sum": values[0], "n_observations": values[1], "mean_attention": values[0] / max(1, values[1])}
        for (role, source, bin_label), values in sorted(diagnostics["bin_summary"].items())
    ]
    write_csv(run_dir / "attention_by_lag_bin.csv", ["predicted_role", "source", "lag_bin_frames", "attention_sum", "n_observations", "mean_attention"], bin_rows)

    phase_rows = [
        {
            "predicted_role": role,
            "source": source,
            "source_session_phase_decile": phase,
            "attention_sum": values[0],
            "n_observations": values[1],
            "mean_attention": values[0] / max(1, values[1]),
        }
        for (role, source, phase), values in sorted(diagnostics["phase_summary"].items())
    ]
    write_csv(run_dir / "attention_by_session_phase.csv", ["predicted_role", "source", "source_session_phase_decile", "attention_sum", "n_observations", "mean_attention"], phase_rows)

    topk_rows = diagnostics["topk_rows"]
    if topk_rows:
        write_csv(
            run_dir / "attention_topk.csv",
            [
                "dataset",
                "session_id",
                "predicted_role",
                "source",
                "window_start",
                "query_frame_idx",
                "source_frame_idx",
                "relative_lag_frames",
                "relative_lag_seconds",
                "source_session_phase_decile",
                "attention_weight",
            ],
            topk_rows,
        )
    else:
        with (run_dir / "attention_topk.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dataset", "session_id", "predicted_role", "source", "window_start", "query_frame_idx", "source_frame_idx", "relative_lag_frames", "relative_lag_seconds", "source_session_phase_decile", "attention_weight"])


@torch.no_grad()
def reconstruct_validation(
    model: RoleAttentionTCNRegressor,
    dataset: WindowedDyadicDataset,
    loader: DataLoader,
    device: torch.device,
    collect_attention: bool,
    attention_topk: int,
    attention_query_stride: int,
    attention_bins: list[int],
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    model.eval()
    sums = [np.zeros((example.aligned_len, 2), dtype=np.float64) for example in dataset.examples]
    counts = [np.zeros(example.aligned_len, dtype=np.float64) for example in dataset.examples]
    diagnostics: dict[str, object] | None = None
    if collect_attention:
        diagnostics = {
            "lag_summary": defaultdict(lambda: [0.0, 0]),
            "source_summary": defaultdict(lambda: [0.0, 0]),
            "bin_summary": defaultdict(lambda: [0.0, 0]),
            "phase_summary": defaultdict(lambda: [0.0, 0]),
            "topk_rows": [],
        }

    for batch in loader:
        if collect_attention:
            output = model(batch["x"].to(device), return_attention=True)
            pred_tensor, attention_payload = output
        else:
            pred_tensor = model(batch["x"].to(device))
            attention_payload = None
        pred = pred_tensor.detach().cpu().numpy()
        frame_mask = batch["frame_mask"].numpy()
        example_idx = batch["example_idx"].numpy()
        starts = batch["start"].numpy()

        if collect_attention and attention_payload is not None and diagnostics is not None:
            novice_weights = attention_payload["novice_weights"].detach().cpu().numpy()
            expert_weights = attention_payload["expert_weights"].detach().cpu().numpy()
            for row_idx in range(pred.shape[0]):
                ex_idx = int(example_idx[row_idx])
                example = dataset.examples[ex_idx]
                valid_len = int(frame_mask[row_idx].sum())
                if valid_len <= 0:
                    continue
                update_attention_diagnostics(
                    diagnostics,
                    novice_weights[row_idx],
                    attention_payload["novice_sources"],
                    "novice",
                    example.dataset,
                    example.session_id,
                    example.aligned_len,
                    int(starts[row_idx]),
                    valid_len,
                    attention_topk,
                    attention_query_stride,
                    attention_bins,
                )
                update_attention_diagnostics(
                    diagnostics,
                    expert_weights[row_idx],
                    attention_payload["expert_sources"],
                    "expert",
                    example.dataset,
                    example.session_id,
                    example.aligned_len,
                    int(starts[row_idx]),
                    valid_len,
                    attention_topk,
                    attention_query_stride,
                    attention_bins,
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

    model = RoleAttentionTCNRegressor(
        n_features_per_role=n_features_per_role,
        hidden_channels=args.hidden_channels,
        levels=args.levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        attention_context=args.attention_context,
        attention_heads=args.attention_heads,
        attention_past_frames=attention_past_arg(args.attention_past_frames),
        exclude_current_frame=args.exclude_current_frame,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ccc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    log_rows: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, ccc_weight=args.ccc_weight)
        # Attention diagnostics are intentionally not collected inside the
        # training loop. Full attention matrices are large, so diagnostics are
        # exported once from the best checkpoint after training.
        reconstructed, _ = reconstruct_validation(
            model,
            val_dataset,
            val_loader,
            device,
            collect_attention=False,
            attention_topk=args.attention_export_topk,
            attention_query_stride=args.attention_export_query_stride,
            attention_bins=args.attention_summary_bins,
        )
        val_metrics = grouped_dyadic_metric_outputs(run_dir, reconstructed)
        val_ccc = val_metrics["ccc"]
        improved = np.isfinite(val_ccc) and val_ccc > best_ccc
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
            }
        )
        write_csv(run_dir / "training_log.csv", list(log_rows[0].keys()), log_rows)
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_ccc={val_ccc:.5f} best_epoch={best_epoch}", flush=True)
        if args.patience > 0 and stale_epochs >= args.patience:
            break

    if args.save_attention and (run_dir / "model_best.pt").exists():
        # Reload the best checkpoint and export attention diagnostics once.
        # This keeps multi-epoch training practical while still making the
        # selected model interpretable.
        checkpoint = torch.load(run_dir / "model_best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        reconstructed, diagnostics = reconstruct_validation(
            model,
            val_dataset,
            val_loader,
            device,
            collect_attention=True,
            attention_topk=args.attention_export_topk,
            attention_query_stride=args.attention_export_query_stride,
            attention_bins=args.attention_summary_bins,
        )
        write_dyadic_prediction_csv(run_dir / "val_predictions.csv", reconstructed)
        if diagnostics is not None:
            write_attention_diagnostics(run_dir, diagnostics)

    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
