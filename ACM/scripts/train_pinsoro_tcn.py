"""Train one PinSoRo feature set with one of the three TCN variants."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
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
    read_pinsoro_window_manifests,
)
from src.acm_pipeline.pinsoro_models_tcn import build_pinsoro_tcn
from src.acm_pipeline.pinsoro_train_utils import (
    CLASS_COUNTS,
    HEADS,
    fill_and_validate_prediction_coverage,
    masked_multitask_cross_entropy,
    prediction_coverage_rows,
    write_csv,
    write_metric_outputs,
    write_pinsoro_submission_tree,
    write_predictions,
    write_test_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PinSoRo fixed-window TCN classifiers."
    )
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--model", choices=["simple", "dyadic_shared", "attention"], required=True
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pinsoro" / "experiments",
    )
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
    parser.add_argument("--mmap-cache-root", type=Path)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from model_last.pt in the run directory when available.",
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


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: (
            str(value)
            if isinstance(value, Path)
            else [str(item) if isinstance(item, Path) else item for item in value]
            if isinstance(value, list)
            else value
        )
        for key, value in vars(args).items()
    }


def make_loader(
    dataset: PinSoRoWindowDataset,
    args: argparse.Namespace,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    common = {
        "num_workers": args.num_workers,
        "collate_fn": pinsoro_window_collate,
        "pin_memory": pin_memory,
        "persistent_workers": args.num_workers > 0,
    }
    if shuffle:
        return DataLoader(
            dataset,
            batch_sampler=SessionBatchSampler(
                dataset.windows, args.batch_size, args.seed
            ),
            **common,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **common,
    )


def compute_class_weights(
    windows: list, dataset: PinSoRoWindowDataset
) -> dict[str, torch.Tensor]:
    counts = {head: np.zeros(CLASS_COUNTS[head], dtype=np.int64) for head in HEADS}
    seen: set[Path] = set()
    for window in windows:
        for role_idx, paths in enumerate(window.tensor_paths):
            label_path = paths[0]
            if not window.supervised[role_idx] or label_path in seen:
                continue
            seen.add(label_path)
            data = dataset.load_full_role(window, role_idx)
            for head in HEADS:
                labels = np.asarray(data[f"{head}_y"], dtype=np.int64)
                mask = np.asarray(data[f"{head}_mask"]).astype(bool)
                counts[head] += np.bincount(labels[mask], minlength=CLASS_COUNTS[head])
    weights = {}
    for head in HEADS:
        total = counts[head].sum()
        values = np.zeros(CLASS_COUNTS[head], dtype=np.float32)
        present = counts[head] > 0
        values[present] = total / (present.sum() * counts[head][present])
        weights[head] = torch.from_numpy(values)
    return weights


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: dict[str, torch.Tensor],
) -> float:
    model.train()
    loss_sum = torch.zeros((), device=device)
    n_batches = 0
    non_blocking = device.type == "cuda"
    for batch in loader:
        if not batch["has_supervision"]:
            continue
        batch = {
            key: value.to(device, non_blocking=non_blocking)
            for key, value in batch.items()
            if key not in ("window_indices", "has_supervision")
        }
        optimizer.zero_grad(set_to_none=True)
        loss = masked_multitask_cross_entropy(model(batch["x"]), batch, class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        loss_sum += loss.detach()
        n_batches += 1
    return float((loss_sum / n_batches).item()) if n_batches else float("nan")


@torch.inference_mode()
def reconstruct(
    model: torch.nn.Module,
    dataset: PinSoRoWindowDataset,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    accumulators: dict[tuple[str, str, str, str], dict[str, object]] = {}
    non_blocking = device.type == "cuda"
    for batch in loader:
        indices = batch["window_indices"].numpy()
        logits = model(batch["x"].to(device, non_blocking=non_blocking))
        logits_np = {
            head: value.detach().cpu().numpy() for head, value in logits.items()
        }
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
                        "task_mask": full["task_mask"][
                            : window.session_aligned_len
                        ].astype(bool),
                        "social_y": full["social_y"][: window.session_aligned_len],
                        "social_mask": full["social_mask"][
                            : window.session_aligned_len
                        ].astype(bool),
                        "covered": np.zeros(window.session_aligned_len, dtype=bool),
                        "task_sum": np.zeros(
                            (window.session_aligned_len, 4), dtype=np.float64
                        ),
                        "social_sum": np.zeros(
                            (window.session_aligned_len, 5), dtype=np.float64
                        ),
                    }
                acc = accumulators[key]
                acc["covered"][s:e] = True
                for head in HEADS:
                    acc[f"{head}_sum"][s:e] += logits_np[head][batch_idx, role_idx]

    reconstructed = []
    for acc in accumulators.values():
        covered = np.asarray(acc["covered"], dtype=bool)
        for head in HEADS:
            sums = acc.pop(f"{head}_sum")
            pred = np.full(len(covered), -1, dtype=np.int64)
            pred[covered] = np.argmax(sums[covered], axis=1)
            acc[f"{head}_pred"] = pred
            acc[f"{head}_mask"] = np.asarray(acc[f"{head}_mask"]).astype(bool)
        reconstructed.append(acc)
    return fill_and_validate_prediction_coverage(reconstructed)


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    return torch.load(path, map_location=device, weights_only=False)


def main() -> None:
    args = parse_args()
    if args.mmap_cache_root is not None and not args.mmap_cache_root.is_absolute():
        args.mmap_cache_root = PROJECT_ROOT / args.mmap_cache_root
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_windows = read_pinsoro_window_manifests(
        args.manifest, PROJECT_ROOT, args.train_split
    )
    val_windows = read_pinsoro_window_manifests(
        args.manifest, PROJECT_ROOT, args.val_split
    )
    if not train_windows or not val_windows:
        raise RuntimeError("Both train and validation PinSoRo windows are required.")
    role_counts = {len(window.roles) for window in train_windows + val_windows}
    expected_roles = 1 if args.model == "simple" else 2
    if role_counts != {expected_roles}:
        raise RuntimeError(
            f"Model {args.model} requires {expected_roles}-role manifest rows, got {sorted(role_counts)}"
        )
    feature_dims = {
        window.n_features_per_role for window in train_windows + val_windows
    }
    if len(feature_dims) != 1:
        raise RuntimeError(
            f"Expected one feature dimension, got {sorted(feature_dims)}"
        )
    n_features = feature_dims.pop()

    train_dataset = PinSoRoWindowDataset(
        train_windows, args.max_cached_tensors, args.mmap_cache_root, PROJECT_ROOT
    )
    val_dataset = PinSoRoWindowDataset(
        val_windows, args.max_cached_tensors, args.mmap_cache_root, PROJECT_ROOT
    )
    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_dataset, args, shuffle=True, pin_memory=pin_memory)
    class_weights = {
        head: value.to(device)
        for head, value in compute_class_weights(train_windows, train_dataset).items()
    }
    val_loader = make_loader(val_dataset, args, shuffle=False, pin_memory=pin_memory)
    model = build_pinsoro_tcn(
        args.model,
        n_features,
        args.hidden_channels,
        args.levels,
        args.kernel_size,
        args.dropout,
        args.attention_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    run_name = (
        args.run_name
        or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pinsoro_{train_windows[0].feature_set}_{args.model}_seed{args.seed}"
    )
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = serializable_args(args) | {
        "feature_set": train_windows[0].feature_set,
        "n_features_per_role": n_features,
        "n_train_windows": len(train_dataset),
        "n_val_windows": len(val_dataset),
        "class_weights": {
            head: value.cpu().tolist() for head, value in class_weights.items()
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_organizer_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    log_rows: list[dict[str, object]] = []
    start_epoch = 1
    last_checkpoint_path = run_dir / "model_last.pt"
    if args.resume and last_checkpoint_path.exists():
        checkpoint = load_checkpoint(last_checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        completed_epoch = int(checkpoint["epoch"])
        start_epoch = completed_epoch + 1
        best_organizer_score = float(checkpoint["best_val_organizer_score"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        log_rows = list(checkpoint["log_rows"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )
        if isinstance(train_loader.batch_sampler, SessionBatchSampler):
            train_loader.batch_sampler.epoch = completed_epoch
        print(
            f"Resuming {run_name} after epoch {completed_epoch:03d} "
            f"(best_epoch={best_epoch}, stale_epochs={stale_epochs})",
            flush=True,
        )
    elif args.resume:
        print(
            f"No last checkpoint found for {run_name}; starting from epoch 1.",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, device, class_weights)
        train_seconds = time.perf_counter() - train_started
        val_started = time.perf_counter()
        reconstructed = reconstruct(model, val_dataset, val_loader, device)
        val_metrics = write_metric_outputs(run_dir, reconstructed)
        val_seconds = time.perf_counter() - val_started
        organizer_score = val_metrics["organizer_score"]
        improved = (
            np.isfinite(organizer_score)
            and organizer_score > best_organizer_score + args.min_delta
        )
        if improved:
            best_organizer_score = organizer_score
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                run_dir / "model_best.pt",
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_organizer_score": organizer_score,
                },
            )
        else:
            stale_epochs += 1
        epoch_seconds = time.perf_counter() - epoch_started
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_organizer_score": organizer_score,
                "best_epoch": best_epoch,
                "best_val_organizer_score": best_organizer_score,
                "stale_epochs": stale_epochs,
                "train_seconds": train_seconds,
                "val_seconds": val_seconds,
                "epoch_seconds": epoch_seconds,
                "train_windows_per_second": len(train_dataset) / train_seconds,
                "val_windows_per_second": len(val_dataset) / val_seconds,
            }
        )
        write_csv(run_dir / "training_log.csv", log_rows)
        last_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_val_organizer_score": best_organizer_score,
            "stale_epochs": stale_epochs,
            "log_rows": log_rows,
            "torch_rng_state": torch.get_rng_state(),
        }
        if device.type == "cuda":
            last_checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        save_checkpoint(last_checkpoint_path, last_checkpoint)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"val_organizer_score={organizer_score:.5f} best_epoch={best_epoch} "
            f"train_seconds={train_seconds:.1f} val_seconds={val_seconds:.1f}",
            flush=True,
        )
        if (
            args.patience > 0
            and epoch >= args.min_epochs
            and stale_epochs >= args.patience
        ):
            break

    checkpoint_path = run_dir / "model_best.pt"
    if checkpoint_path.exists():
        model.load_state_dict(
            load_checkpoint(checkpoint_path, device)["model_state_dict"]
        )
        reconstructed = reconstruct(model, val_dataset, val_loader, device)
        write_metric_outputs(run_dir, reconstructed)
        coverage_rows = prediction_coverage_rows(reconstructed, "validation")
        write_predictions(run_dir / "val_predictions.csv", reconstructed)
        test_windows = read_pinsoro_window_manifests(
            args.manifest, PROJECT_ROOT, args.test_split
        )
        if test_windows:
            test_dataset = PinSoRoWindowDataset(
                test_windows, args.max_cached_tensors, args.mmap_cache_root, PROJECT_ROOT
            )
            test_loader = make_loader(
                test_dataset, args, shuffle=False, pin_memory=pin_memory
            )
            test_reconstructed = reconstruct(model, test_dataset, test_loader, device)
            coverage_rows.extend(prediction_coverage_rows(test_reconstructed, "test"))
            write_test_predictions(run_dir / "test_predictions.csv", test_reconstructed)
            write_pinsoro_submission_tree(
                run_dir / "test_submission_format", test_reconstructed
            )
        write_csv(run_dir / "prediction_coverage.csv", coverage_rows)
    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
