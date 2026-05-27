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

# TCN training consumes any transformed manifest with x/y/target_mask tensors.
# Windowing happens lazily in the dataset, so no overlapping window files are
# written to disk and the same tensor set can support many window sizes.
from src.acm_pipeline.data import WindowedSequenceDataset, read_model_manifest
from src.acm_pipeline.metrics import ccc_loss, masked_mse_loss
from src.acm_pipeline.models_tcn import TCNRegressor
from src.acm_pipeline.train_utils import grouped_metric_outputs, write_csv, write_prediction_csv


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TCN baseline from any transformed feature manifest.")
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
    if args.run_name.strip():
        run_name = args.run_name.strip()
    else:
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_tcn"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def save_config(args: argparse.Namespace, run_dir: Path, input_dim: int, n_train: int, n_val: int) -> None:
    payload = serializable_args(args)
    payload.update({"run_dir": str(run_dir), "input_dim": input_dim, "n_train_examples": n_train, "n_val_examples": n_val})
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def train_one_epoch(model: TCNRegressor, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, ccc_weight: float) -> float:
    model.train()
    losses = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)

        # MSE keeps frame-wise errors grounded; CCC encourages correlation,
        # scale, and location agreement with the continuous target.
        loss = masked_mse_loss(pred, y, loss_mask) + ccc_weight * ccc_loss(pred, y, loss_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def reconstruct_validation(model: TCNRegressor, dataset: WindowedSequenceDataset, loader: DataLoader, device: torch.device) -> list[dict[str, object]]:
    model.eval()
    sums = [np.zeros(example.aligned_len, dtype=np.float64) for example in dataset.examples]
    counts = [np.zeros(example.aligned_len, dtype=np.float64) for example in dataset.examples]

    # Each frame can be predicted by multiple overlapping windows. Validation
    # reconstruction averages all available predictions back to session length.
    for batch in loader:
        pred = model(batch["x"].to(device)).detach().cpu().numpy()
        frame_mask = batch["frame_mask"].numpy()
        example_idx = batch["example_idx"].numpy()
        starts = batch["start"].numpy()
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
        pred = np.full(example.aligned_len, np.nan, dtype=np.float32)
        covered = counts[ex_idx] > 0
        pred[covered] = (sums[ex_idx][covered] / counts[ex_idx][covered]).astype(np.float32)
        reconstructed.append(
            {
                "example": example,
                "y_true": session.y[: example.aligned_len],
                "target_mask": session.target_mask[: example.aligned_len],
                "y_pred": pred,
                "covered": covered.astype(np.float32),
            }
        )
    return reconstructed


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = make_run_dir(args)

    train_examples = read_model_manifest(args.manifest, PROJECT_ROOT, split=args.train_split)
    val_examples = read_model_manifest(args.manifest, PROJECT_ROOT, split=args.val_split)
    if not train_examples or not val_examples:
        raise RuntimeError("Both train and validation examples are required.")
    input_dims = sorted({example.n_features for example in train_examples + val_examples})
    if len(input_dims) != 1:
        raise RuntimeError(f"Expected one fixed input dimension, got {input_dims}")
    input_dim = input_dims[0]
    save_config(args, run_dir, input_dim=input_dim, n_train=len(train_examples), n_val=len(val_examples))

    train_dataset = WindowedSequenceDataset(train_examples, args.window_size, args.stride, max_windows=args.max_train_windows, seed=args.seed)
    val_dataset = WindowedSequenceDataset(val_examples, args.window_size, args.stride)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TCNRegressor(input_dim, args.hidden_channels, args.levels, args.kernel_size, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ccc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    log_rows: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, ccc_weight=args.ccc_weight)
        reconstructed = reconstruct_validation(model, val_dataset, val_loader, device)
        val_metrics = grouped_metric_outputs(run_dir, reconstructed)
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
            write_prediction_csv(run_dir / "val_predictions.csv", reconstructed)
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

    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()

