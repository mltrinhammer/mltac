"""Train one PinSoRo MoE expert with age/gender injected at prediction heads."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_pinsoro_tcn import (  # noqa: E402
    compute_class_weights,
    compute_cr_social_weights,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    serializable_args,
    set_seed,
)
from src.acm_pipeline.models_tcn import TemporalBlock  # noqa: E402
from src.acm_pipeline.pinsoro_data import (  # noqa: E402
    PinSoRoWindow,
    PinSoRoWindowDataset,
    SessionBatchSampler,
    read_pinsoro_window_manifests,
)
from src.acm_pipeline.pinsoro_train_utils import (  # noqa: E402
    HEADS,
    fill_and_validate_prediction_coverage,
    masked_multitask_cross_entropy,
    prediction_coverage_rows,
    write_csv,
    write_metric_outputs,
    write_pinsoro_submission_tree,
    write_prediction_scores,
    write_predictions,
    write_test_predictions,
)


TASK_CLASSES = 4
SOCIAL_CLASSES = 5
METADATA_FIELDS = ("age_z", "gender_1", "gender_2", "gender_unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PinSoRo dyadic TCN with metadata heads.")
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-mode", choices=("age_gender", "age_only", "gender_only"), default="age_gender")
    parser.add_argument("--metadata-dropout", type=float, default=0.2)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--train-split", default="train_internal")
    parser.add_argument("--val-split", default="val_internal")
    parser.add_argument("--test-split", default="test_internal")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--causal-tcn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cr-social-weighting",
        choices=("shared_inverse", "unweighted", "sqrt_inverse", "capped_inverse", "targeted"),
        default="shared_inverse",
    )
    parser.add_argument("--cr-social-weight-cap", type=float, default=5.0)
    parser.add_argument("--cr-social-target-class0-weight", type=float, default=1.0)
    parser.add_argument("--cr-social-target-class2-weight", type=float, default=2.0)
    parser.add_argument("--cr-social-target-class3-weight", type=float, default=0.5)
    parser.add_argument("--cr-social-focal-gamma", type=float, default=0.0)
    parser.add_argument("--cr-social-class3-oversample", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--min-epochs", type=int, default=24)
    parser.add_argument("--min-delta", type=float, default=5e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-cached-tensors", type=int, default=2)
    parser.add_argument("--mmap-cache-root", type=Path)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--soft-label-mode",
        choices=("none", "soft_uniform", "soft_confidence"),
        default="none",
        help="Use optional disagreement soft-label tensors for training.",
    )
    return parser.parse_args()


def make_encoder(input_dim: int, hidden_channels: int, levels: int, kernel_size: int, dropout: float, causal: bool) -> nn.Sequential:
    channels = [input_dim] + [hidden_channels] * levels
    return nn.Sequential(
        *[
            TemporalBlock(channels[idx], channels[idx + 1], kernel_size, 2**idx, dropout, causal=causal)
            for idx in range(levels)
        ]
    )


class PinSoRoDyadicMetadataHeadTCN(nn.Module):
    """Dyadic TCN with role metadata concatenated immediately before heads."""

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
        self.encoder = make_encoder(2 * n_features_per_role, hidden_channels, levels, kernel_size, dropout, causal_tcn)
        self.metadata_dropout = nn.Dropout(metadata_dropout)
        head_dim = hidden_channels + 2 * metadata_dim
        self.task_head = nn.Conv1d(head_dim, 2 * TASK_CLASSES, kernel_size=1)
        self.social_head = nn.Conv1d(head_dim, 2 * SOCIAL_CLASSES, kernel_size=1)

    @staticmethod
    def reshape(logits: torch.Tensor, roles: int, classes: int) -> torch.Tensor:
        batch, _, time = logits.shape
        return logits.reshape(batch, roles, classes, time).permute(0, 1, 3, 2)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor | None = None, metadata: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x.shape[1] != 2:
            raise ValueError(f"Metadata-head dyadic model requires two roles, got shape {tuple(x.shape)}")
        if metadata is None:
            raise ValueError("metadata is required for metadata-head model")
        batch, roles, features, time = x.shape
        hidden = self.encoder(x.reshape(batch, roles * features, time))
        meta = self.metadata_dropout(metadata.reshape(batch, roles * metadata.shape[-1]))
        meta = meta.unsqueeze(-1).expand(-1, -1, time)
        head_input = torch.cat([hidden, meta], dim=1)
        return {
            "task": self.reshape(self.task_head(head_input), roles, TASK_CLASSES),
            "social": self.reshape(self.social_head(head_input), roles, SOCIAL_CLASSES),
        }


def read_metadata(path: Path) -> dict[tuple[str, str, str], tuple[float | None, str]]:
    table: dict[tuple[str, str, str], tuple[float | None, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            age = None if row.get("age", "") == "" else float(row["age"])
            gender = str(row.get("gender", "")).strip()
            table[(row["source_split"], row["session_id"], row["role"])] = (age, gender)
    return table


def metadata_stats(windows: list[PinSoRoWindow], table: dict[tuple[str, str, str], tuple[float | None, str]]) -> dict[str, float]:
    seen: set[tuple[str, str, str]] = set()
    ages = []
    for window in windows:
        for role in window.roles:
            key = (window.source_split, window.session_id, role)
            if key in seen:
                continue
            seen.add(key)
            age, _gender = table.get(key, (None, ""))
            if age is not None:
                ages.append(age)
    if not ages:
        raise RuntimeError("No age metadata found for train windows.")
    mean = float(np.mean(ages))
    std = float(np.std(ages))
    return {"age_mean": mean, "age_std": std if std > 1.0e-6 else 1.0}


def encode_one(age: float | None, gender: str, stats: dict[str, float], mode: str) -> np.ndarray:
    values = np.zeros(len(METADATA_FIELDS), dtype=np.float32)
    if mode in {"age_gender", "age_only"}:
        values[0] = 0.0 if age is None else (age - stats["age_mean"]) / stats["age_std"]
    if mode in {"age_gender", "gender_only"}:
        if gender == "1":
            values[1] = 1.0
        elif gender == "2":
            values[2] = 1.0
        else:
            values[3] = 1.0
    return values


class MetadataWindowDataset(PinSoRoWindowDataset):
    def __init__(self, *args, metadata_table: dict[tuple[str, str, str], tuple[float | None, str]], metadata_stats: dict[str, float], metadata_mode: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metadata_table = metadata_table
        self.metadata_stats = metadata_stats
        self.metadata_mode = metadata_mode

    def __getitem__(self, idx: int) -> dict[str, object]:
        item = super().__getitem__(idx)
        window = self.windows[idx]
        item["metadata"] = np.stack(
            [
                encode_one(*self.metadata_table.get((window.source_split, window.session_id, role), (None, "")), self.metadata_stats, self.metadata_mode)
                for role in window.roles
            ],
            axis=0,
        )
        return item


def metadata_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    required = ("x", "task_y", "social_y", "task_mask", "social_mask", "metadata")
    optional = (
        "task_soft_y",
        "task_soft_mask",
        "task_weight",
        "social_soft_y",
        "social_soft_mask",
        "social_weight",
    )
    arrays = {
        key: np.stack([item[key] for item in batch])
        for key in required + tuple(key for key in optional if all(key in item for item in batch))
    }
    supervision_keys = ("task_soft_mask", "social_soft_mask") if "task_soft_mask" in arrays else ("task_mask", "social_mask")
    return {key: torch.from_numpy(value) for key, value in arrays.items()} | {
        "domain_id": torch.as_tensor([item["domain_id"] for item in batch], dtype=torch.long),
        "window_indices": torch.as_tensor([item["window_index"] for item in batch], dtype=torch.long),
        "has_supervision": any(np.any(arrays[key]) for key in supervision_keys),
    }


def make_loader(dataset: MetadataWindowDataset, args: argparse.Namespace, shuffle: bool, pin_memory: bool) -> DataLoader:
    common = {
        "num_workers": args.num_workers,
        "collate_fn": metadata_collate,
        "pin_memory": pin_memory,
        "persistent_workers": args.num_workers > 0,
    }
    if shuffle:
        return DataLoader(dataset, batch_sampler=SessionBatchSampler(dataset.windows, args.batch_size, args.seed), **common)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **common)



def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: dict[str, torch.Tensor],
    cr_social_focal_gamma: float,
    soft_label_mode: str = "none",
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
        loss = masked_multitask_cross_entropy(
            model(batch["x"], batch["domain_id"], batch["metadata"]),
            batch,
            class_weights,
            cr_social_focal_gamma=cr_social_focal_gamma,
            soft_label_mode=soft_label_mode,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        loss_sum += loss.detach()
        n_batches += 1
    return float((loss_sum / n_batches).item()) if n_batches else float("nan")


@torch.inference_mode()
def reconstruct(model: nn.Module, dataset: MetadataWindowDataset, loader: DataLoader, device: torch.device) -> list[dict[str, object]]:
    model.eval()
    accumulators: dict[tuple[str, str, str, str], dict[str, object]] = {}
    non_blocking = device.type == "cuda"
    for batch in loader:
        indices = batch["window_indices"].numpy()
        logits = model(
            batch["x"].to(device, non_blocking=non_blocking),
            batch["domain_id"].to(device, non_blocking=non_blocking),
            batch["metadata"].to(device, non_blocking=non_blocking),
        )
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
                        "covered": np.zeros(window.session_aligned_len, dtype=bool),
                        "coverage_count": np.zeros(window.session_aligned_len, dtype=np.int32),
                        "task_sum": np.zeros((window.session_aligned_len, TASK_CLASSES), dtype=np.float64),
                        "social_sum": np.zeros((window.session_aligned_len, SOCIAL_CLASSES), dtype=np.float64),
                    }
                acc = accumulators[key]
                acc["covered"][s:e] = True
                acc["coverage_count"][s:e] += 1
                for head in HEADS:
                    acc[f"{head}_sum"][s:e] += logits_np[head][batch_idx, role_idx]
    reconstructed = []
    for acc in accumulators.values():
        covered = np.asarray(acc["covered"], dtype=bool)
        coverage_count = np.asarray(acc.pop("coverage_count"), dtype=np.int32)
        for head in HEADS:
            sums = acc.pop(f"{head}_sum")
            pred = np.full(len(covered), -1, dtype=np.int64)
            averaged_logits = np.zeros_like(sums)
            averaged_logits[covered] = sums[covered] / coverage_count[covered, None]
            shifted = averaged_logits[covered] - averaged_logits[covered].max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            probabilities = np.zeros_like(averaged_logits)
            probabilities[covered] = exp / exp.sum(axis=1, keepdims=True)
            pred[covered] = np.argmax(averaged_logits[covered], axis=1)
            acc[f"{head}_pred"] = pred
            acc[f"{head}_logits"] = averaged_logits
            acc[f"{head}_probabilities"] = probabilities
            acc[f"{head}_mask"] = np.asarray(acc[f"{head}_mask"]).astype(bool)
        reconstructed.append(acc)
    return fill_and_validate_prediction_coverage(reconstructed)


def make_dataset(windows: list[PinSoRoWindow], args: argparse.Namespace, table: dict[tuple[str, str, str], tuple[float | None, str]], stats: dict[str, float]) -> MetadataWindowDataset:
    return MetadataWindowDataset(
        windows,
        args.max_cached_tensors,
        args.mmap_cache_root,
        PROJECT_ROOT,
        metadata_table=table,
        metadata_stats=stats,
        metadata_mode=args.metadata_mode,
    )



def window_has_cr_social_class3(window: PinSoRoWindow, dataset: MetadataWindowDataset) -> bool:
    if window.domain != "CR":
        return False
    for role_idx, supervised in enumerate(window.supervised):
        if not supervised:
            continue
        data = dataset.load_full_role(window, role_idx)
        labels = np.asarray(data["social_y"], dtype=np.int64)[window.start_frame : window.end_frame]
        mask = np.asarray(data["social_mask"])[window.start_frame : window.end_frame].astype(bool)
        if np.any(mask & (labels == 3)):
            return True
    return False


def oversample_cr_social_class3_windows(
    windows: list[PinSoRoWindow],
    args: argparse.Namespace,
    table: dict[tuple[str, str, str], tuple[float | None, str]],
    stats: dict[str, float],
) -> list[PinSoRoWindow]:
    multiplier = max(1, int(args.cr_social_class3_oversample))
    if multiplier <= 1:
        return windows
    probe_dataset = make_dataset(windows, args, table, stats)
    class3_windows = [window for window in windows if window_has_cr_social_class3(window, probe_dataset)]
    if not class3_windows:
        print("CR social class-3 oversampling requested, but no matching windows found.", flush=True)
        return windows
    expanded = list(windows) + class3_windows * (multiplier - 1)
    print(
        f"CR social class-3 oversampling: base_windows={len(windows)} "
        f"class3_windows={len(class3_windows)} multiplier={multiplier} expanded_windows={len(expanded)}",
        flush=True,
    )
    return expanded

def main() -> None:
    args = parse_args()
    if args.mmap_cache_root is not None and not args.mmap_cache_root.is_absolute():
        args.mmap_cache_root = PROJECT_ROOT / args.mmap_cache_root
    if args.metadata is not None and not args.metadata.is_absolute():
        args.metadata = PROJECT_ROOT / args.metadata
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_windows = read_pinsoro_window_manifests(args.manifest, PROJECT_ROOT, args.train_split)
    val_windows = read_pinsoro_window_manifests(args.manifest, PROJECT_ROOT, args.val_split)
    if not train_windows or not val_windows:
        raise RuntimeError("Both train and validation PinSoRo windows are required.")
    role_counts = {len(window.roles) for window in train_windows + val_windows}
    if role_counts != {2}:
        raise RuntimeError(f"Metadata-head model requires dyadic rows, got {sorted(role_counts)}")
    feature_dims = {window.n_features_per_role for window in train_windows + val_windows}
    if len(feature_dims) != 1:
        raise RuntimeError(f"Expected one feature dimension, got {sorted(feature_dims)}")
    n_features = feature_dims.pop()

    table = read_metadata(args.metadata)
    stats = metadata_stats(train_windows, table)
    train_windows = oversample_cr_social_class3_windows(train_windows, args, table, stats)
    train_dataset = make_dataset(train_windows, args, table, stats)
    val_dataset = make_dataset(val_windows, args, table, stats)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_dataset, args, shuffle=True, pin_memory=pin_memory)
    val_loader = make_loader(val_dataset, args, shuffle=False, pin_memory=pin_memory)
    class_weights = {head: value.to(device) for head, value in compute_class_weights(train_windows, train_dataset).items()}
    cr_social_weights = compute_cr_social_weights(
        train_windows,
        train_dataset,
        args.cr_social_weighting,
        args.cr_social_weight_cap,
        args.cr_social_target_class2_weight,
        args.cr_social_target_class3_weight,
        args.cr_social_target_class0_weight,
    )
    if cr_social_weights is not None:
        class_weights["cr_social"] = cr_social_weights.to(device)
    model = PinSoRoDyadicMetadataHeadTCN(
        n_features,
        len(METADATA_FIELDS),
        args.hidden_channels,
        args.levels,
        args.kernel_size,
        args.dropout,
        args.metadata_dropout,
        args.causal_tcn,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pinsoro_{train_windows[0].feature_set}_metadata_head_seed{args.seed}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = serializable_args(args) | {
        "model": "dyadic_metadata_head",
        "feature_set": train_windows[0].feature_set,
        "n_features_per_role": n_features,
        "metadata_fields": list(METADATA_FIELDS),
        "metadata_stats": stats,
        "n_train_windows": len(train_dataset),
        "n_val_windows": len(val_dataset),
        "class_weights": {head: value.cpu().tolist() for head, value in class_weights.items()},
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
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state_all"]])
        if isinstance(train_loader.batch_sampler, SessionBatchSampler):
            train_loader.batch_sampler.epoch = completed_epoch
        print(f"Resuming {run_name} after epoch {completed_epoch:03d}", flush=True)
    elif args.resume:
        print(f"No last checkpoint found for {run_name}; starting from epoch 1.", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_started = time.perf_counter()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights,
            args.cr_social_focal_gamma,
            args.soft_label_mode,
        )
        train_seconds = time.perf_counter() - train_started
        val_started = time.perf_counter()
        reconstructed = reconstruct(model, val_dataset, val_loader, device)
        val_metrics = write_metric_outputs(run_dir, reconstructed)
        val_seconds = time.perf_counter() - val_started
        organizer_score = val_metrics["organizer_score"]
        improved = np.isfinite(organizer_score) and organizer_score > best_organizer_score + args.min_delta
        if improved:
            best_organizer_score = organizer_score
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(run_dir / "model_best.pt", {"epoch": epoch, "model_state_dict": model.state_dict(), "val_organizer_score": organizer_score})
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
            f"epoch={epoch:03d} train_loss={train_loss:.5f} val_organizer_score={organizer_score:.5f} "
            f"best_epoch={best_epoch} train_seconds={train_seconds:.1f} val_seconds={val_seconds:.1f}",
            flush=True,
        )
        if args.patience > 0 and epoch >= args.min_epochs and stale_epochs >= args.patience:
            break

    checkpoint_path = run_dir / "model_best.pt"
    if checkpoint_path.exists():
        model.load_state_dict(load_checkpoint(checkpoint_path, device)["model_state_dict"])
        reconstructed = reconstruct(model, val_dataset, val_loader, device)
        write_metric_outputs(run_dir, reconstructed)
        coverage_rows = prediction_coverage_rows(reconstructed, "validation")
        write_predictions(run_dir / "val_predictions.csv", reconstructed)
        write_prediction_scores(run_dir / "val_prediction_scores.csv.gz", reconstructed)
        test_windows = read_pinsoro_window_manifests(args.manifest, PROJECT_ROOT, args.test_split)
        if test_windows:
            test_dataset = make_dataset(test_windows, args, table, stats)
            test_loader = make_loader(test_dataset, args, shuffle=False, pin_memory=pin_memory)
            test_reconstructed = reconstruct(model, test_dataset, test_loader, device)
            coverage_rows.extend(prediction_coverage_rows(test_reconstructed, "test"))
            write_test_predictions(run_dir / "test_predictions.csv", test_reconstructed)
            write_prediction_scores(run_dir / "test_prediction_scores.csv.gz", test_reconstructed, supervised_only=False)
            write_pinsoro_submission_tree(run_dir / "test_submission_format", test_reconstructed)
        write_csv(run_dir / "prediction_coverage.csv", coverage_rows)
    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
