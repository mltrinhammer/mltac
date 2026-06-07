"""Train one PinSoRo feature set with one of the three TCN variants."""

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

from src.acm_pipeline.pinsoro_data import (
    PinSoRoWindowDataset,
    SessionBatchSampler,
    pinsoro_window_collate,
    read_pinsoro_window_manifest,
)
from src.acm_pipeline.pinsoro_models_tcn import build_pinsoro_tcn
from src.acm_pipeline.pinsoro_train_utils import (
    HEADS,
    masked_multitask_cross_entropy,
    write_csv,
    write_metric_outputs,
    write_predictions,
    write_test_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PinSoRo fixed-window TCN classifiers.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=["simple", "dyadic_shared", "attention"], required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "pinsoro" / "experiments")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--train-split", default="train_internal")
    parser.add_argument("--val-split", default="val_internal")
    parser.add_argument("--test-split", default="test_internal")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-cached-tensors", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def make_loader(dataset: PinSoRoWindowDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    if shuffle:
        return DataLoader(
            dataset,
            batch_sampler=SessionBatchSampler(dataset.windows, args.batch_size, args.seed),
            num_workers=args.num_workers,
            collate_fn=pinsoro_window_collate,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pinsoro_window_collate,
    )


def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        if not any(torch.any(batch[f"{head}_mask"]) for head in HEADS):
            continue
        optimizer.zero_grad(set_to_none=True)
        loss = masked_multitask_cross_entropy(model(batch["x"]), batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def reconstruct(model: torch.nn.Module, dataset: PinSoRoWindowDataset, loader: DataLoader, device: torch.device) -> list[dict[str, object]]:
    model.eval()
    accumulators: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for batch in loader:
        indices = batch["window_indices"].numpy()
        logits = model(batch["x"].to(device))
        logits_np = {head: value.detach().cpu().numpy() for head, value in logits.items()}
        for batch_idx, window_idx in enumerate(indices):
            window = dataset.windows[int(window_idx)]
            s, e = window.start_frame, window.end_frame
            for role_idx, role in enumerate(window.roles):
                key = (window.domain, window.source_split, window.session_id, role)
                if key not in accumulators:
                    full = dataset.load_full_role(window, role_idx)
                    accumulators[key] = {
                        "domain": window.domain,
                        "source_split": window.source_split,
                        "session_id": window.session_id,
                        "role": role,
                        "task_y": full["task_y"][: window.session_aligned_len],
                        "task_mask": full["task_mask"][: window.session_aligned_len].astype(bool),
                        "social_y": full["social_y"][: window.session_aligned_len],
                        "social_mask": full["social_mask"][: window.session_aligned_len].astype(bool),
                        "count": np.zeros(window.session_aligned_len, dtype=np.float64),
                        "task_sum": np.zeros((window.session_aligned_len, 4), dtype=np.float64),
                        "social_sum": np.zeros((window.session_aligned_len, 5), dtype=np.float64),
                    }
                acc = accumulators[key]
                acc["count"][s:e] += 1.0
                for head in HEADS:
                    acc[f"{head}_sum"][s:e] += logits_np[head][batch_idx, role_idx]

    reconstructed = []
    for acc in accumulators.values():
        covered = acc.pop("count") > 0
        acc["covered"] = covered
        for head in HEADS:
            sums = acc.pop(f"{head}_sum")
            pred = np.full(len(covered), -1, dtype=np.int64)
            pred[covered] = np.argmax(sums[covered], axis=1)
            acc[f"{head}_pred"] = pred
            acc[f"{head}_mask"] = np.asarray(acc[f"{head}_mask"]) & covered
        reconstructed.append(acc)
    return reconstructed


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_windows = read_pinsoro_window_manifest(args.manifest, PROJECT_ROOT, args.train_split)
    val_windows = read_pinsoro_window_manifest(args.manifest, PROJECT_ROOT, args.val_split)
    if not train_windows or not val_windows:
        raise RuntimeError("Both train and validation PinSoRo windows are required.")
    role_counts = {len(window.roles) for window in train_windows + val_windows}
    expected_roles = 1 if args.model == "simple" else 2
    if role_counts != {expected_roles}:
        raise RuntimeError(f"Model {args.model} requires {expected_roles}-role manifest rows, got {sorted(role_counts)}")
    feature_dims = {window.n_features_per_role for window in train_windows + val_windows}
    if len(feature_dims) != 1:
        raise RuntimeError(f"Expected one feature dimension, got {sorted(feature_dims)}")
    n_features = feature_dims.pop()

    train_dataset = PinSoRoWindowDataset(train_windows, args.max_cached_tensors)
    val_dataset = PinSoRoWindowDataset(val_windows, args.max_cached_tensors)
    train_loader = make_loader(train_dataset, args, shuffle=True)
    val_loader = make_loader(val_dataset, args, shuffle=False)
    model = build_pinsoro_tcn(
        args.model,
        n_features,
        args.hidden_channels,
        args.levels,
        args.kernel_size,
        args.dropout,
        args.attention_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pinsoro_{train_windows[0].feature_set}_{args.model}_seed{args.seed}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = serializable_args(args) | {
        "feature_set": train_windows[0].feature_set,
        "n_features_per_role": n_features,
        "n_train_windows": len(train_dataset),
        "n_val_windows": len(val_dataset),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_kappa = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        reconstructed = reconstruct(model, val_dataset, val_loader, device)
        val_metrics = write_metric_outputs(run_dir, reconstructed)
        mean_kappa = val_metrics["mean_kappa"]
        improved = np.isfinite(mean_kappa) and mean_kappa > best_kappa + args.min_delta
        if improved:
            best_kappa = mean_kappa
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_mean_kappa": mean_kappa}, run_dir / "model_best.pt")
        else:
            stale_epochs += 1
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mean_kappa": mean_kappa,
                "best_epoch": best_epoch,
                "best_val_mean_kappa": best_kappa,
                "stale_epochs": stale_epochs,
            }
        )
        write_csv(run_dir / "training_log.csv", log_rows)
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_mean_kappa={mean_kappa:.5f} best_epoch={best_epoch}", flush=True)
        if args.patience > 0 and epoch >= args.min_epochs and stale_epochs >= args.patience:
            break

    checkpoint_path = run_dir / "model_best.pt"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
        reconstructed = reconstruct(model, val_dataset, val_loader, device)
        write_metric_outputs(run_dir, reconstructed)
        write_predictions(run_dir / "val_predictions.csv", reconstructed)
        test_windows = read_pinsoro_window_manifest(args.manifest, PROJECT_ROOT, args.test_split)
        if test_windows:
            test_dataset = PinSoRoWindowDataset(test_windows, args.max_cached_tensors)
            test_loader = make_loader(test_dataset, args, shuffle=False)
            test_reconstructed = reconstruct(model, test_dataset, test_loader, device)
            write_test_predictions(run_dir / "test_predictions.csv", test_reconstructed)
    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
