"""Train DAPA-inspired group-level engagement model (TCN + BiLSTM + CrossAttention).

Supports joint NoXi + MPIIGI training with domain prompts, EMA, and cosine
annealing with warmup.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
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

from src.acm_pipeline.dyadic_train_utils import write_csv, write_organizer_submission_tree
from src.acm_pipeline.group_data import (
    GroupMultimodalWindowDataset,
    GroupMultimodalWindowSample,
    group_multimodal_window_collate_fn,
    read_group_multimodal_window_manifest,
)
from src.acm_pipeline.group_models import DAPAGroupMultimodalRegressor
from src.acm_pipeline.metrics import ccc_loss, masked_mse_loss, regression_metrics


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments"


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        """Swap model params with EMA shadow params for evaluation."""
        self._backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore model params from backup after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.shadow)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine annealing to zero."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DAPA-inspired group engagement model.")
    # Data
    parser.add_argument("--manifest", type=Path, required=True, help="Primary MPIIGI group-window manifest.")
    parser.add_argument("--noxi-manifest", type=Path, default=None, help="Optional NoXi group-format manifest for joint training.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--train-split", default="train_internal")
    parser.add_argument("--val-split", default="val_internal")
    parser.add_argument("--test-splits", nargs="*", default=["test_internal", "test"])
    parser.add_argument("--min-window-frames", type=int, default=5)

    # Fusion
    parser.add_argument("--fusion-mode", choices=["gated", "concat"], default="gated")
    parser.add_argument("--fusion-channels", type=int, default=64)
    parser.add_argument("--modality-dropout", type=float, default=0.1)

    # TCN
    parser.add_argument("--hidden-channels", type=int, default=64, help="TCN hidden channels.")
    parser.add_argument("--levels", type=int, default=4, help="TCN dilated levels.")
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2, help="TCN dropout.")

    # BiLSTM
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--lstm-dropout", type=float, default=0.1)

    # Cross-attention
    parser.add_argument("--cross-attn-heads", type=int, default=4)
    parser.add_argument("--cross-attn-dropout", type=float, default=0.1)

    # Domain prompting
    parser.add_argument("--n-domains", type=int, default=2)
    parser.add_argument("--n-prompt-tokens", type=int, default=4)
    parser.add_argument("--use-domain-prompts", action="store_true", default=True)
    parser.add_argument("--no-domain-prompts", dest="use_domain_prompts", action="store_false")

    # Encoder sharing
    parser.add_argument("--encoder-sharing", choices=("shared", "separate"), default="shared")
    parser.add_argument("--max-role-encoders", type=int, default=8)

    # Prediction
    parser.add_argument("--prediction-dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--ccc-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    # EMA
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument("--ema-decay", type=float, default=0.999)

    # Early stopping (0 disables)
    parser.add_argument("--patience", type=int, default=0, help="0 = train all epochs, no early stopping.")
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-3)

    # Misc
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers (reused from train_mpii_group_meanpool_multimodal.py)
# ---------------------------------------------------------------------------

class _SessionStub:
    def __init__(self, dataset: str, session_id: str, model_split: str, role_names: tuple[str, ...]) -> None:
        self.dataset = dataset
        self.session_id = session_id
        self.model_split = model_split
        self.role_names = role_names


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
    run_name = args.run_name.strip() or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_dapa_group"
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


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(args: argparse.Namespace, modality_dims: dict[str, int]) -> DAPAGroupMultimodalRegressor:
    return DAPAGroupMultimodalRegressor(
        modality_dims=modality_dims,
        fusion_channels=args.fusion_channels,
        fusion_mode=args.fusion_mode,
        modality_dropout=args.modality_dropout,
        tcn_hidden_channels=args.hidden_channels,
        tcn_levels=args.levels,
        tcn_kernel_size=args.kernel_size,
        tcn_dropout=args.dropout,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        cross_attn_heads=args.cross_attn_heads,
        cross_attn_dropout=args.cross_attn_dropout,
        n_domains=args.n_domains,
        n_prompt_tokens=args.n_prompt_tokens,
        use_domain_prompts=args.use_domain_prompts,
        encoder_sharing=args.encoder_sharing,
        max_role_encoders=args.max_role_encoders,
        prediction_dropout=args.prediction_dropout,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    ema: EMA | None,
    device: torch.device,
    ccc_weight: float,
    mse_weight: float,
    grad_clip: float,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        x_modalities = move_modalities_to_device(batch["x_modalities"], device)
        role_mask = batch["role_mask"].to(device)
        y = batch["y"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        domain_ids = batch["domain_ids"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x_modalities, role_mask=role_mask, domain_ids=domain_ids)
        loss = mse_weight * masked_mse_loss(pred, y, loss_mask) + ccc_weight * ccc_loss(pred, y, loss_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


# ---------------------------------------------------------------------------
# Reconstruction & metrics (adapted from train_mpii_group_meanpool_multimodal)
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct(
    model: nn.Module,
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
        pred_tensor = model(
            move_modalities_to_device(batch["x_modalities"], device),
            role_mask=batch["role_mask"].to(device),
            domain_ids=batch["domain_ids"].to(device),
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = make_run_dir(args)

    # Load primary manifest (MPIIGI)
    train_samples = read_group_multimodal_window_manifest(args.manifest, PROJECT_ROOT, split=args.train_split)
    val_samples = read_group_multimodal_window_manifest(args.manifest, PROJECT_ROOT, split=args.val_split)

    # Optionally add NoXi samples for joint training
    if args.noxi_manifest is not None:
        noxi_train = read_group_multimodal_window_manifest(args.noxi_manifest, PROJECT_ROOT, split=args.train_split)
        noxi_val = read_group_multimodal_window_manifest(args.noxi_manifest, PROJECT_ROOT, split=args.val_split)
        print(f"Joint training: MPIIGI train={len(train_samples)}, NoXi train={len(noxi_train)}")
        train_samples = train_samples + noxi_train
        val_samples = val_samples + noxi_val

    if not train_samples or not val_samples:
        raise RuntimeError("Both train and validation group-window rows are required.")

    combo_name, modality_order, modality_dims = infer_layout(train_samples + val_samples)
    train_dataset = GroupMultimodalWindowDataset(train_samples, min_frames=args.min_window_frames)
    val_dataset = GroupMultimodalWindowDataset(val_samples, min_frames=args.min_window_frames)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=group_multimodal_window_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=group_multimodal_window_collate_fn,
    )

    # Save config
    config = serializable_args(args)
    config.update({
        "run_dir": str(run_dir),
        "combo_name": combo_name,
        "modality_order": list(modality_order),
        "modality_dims": modality_dims,
        "n_train_windows": len(train_dataset),
        "n_val_windows": len(val_dataset),
    })
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    # Build model
    model = build_model(args, modality_dims).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_steps)
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None

    best_ccc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    log_rows: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, ema, device,
            args.ccc_weight, args.mse_weight, args.grad_clip,
        )

        # Evaluate with EMA params if available
        if ema is not None:
            ema.apply_shadow(model)

        reconstructed = reconstruct(model, val_dataset, val_loader, device)
        val_metrics = write_group_metrics(run_dir, reconstructed)
        val_ccc = val_metrics["ccc"]

        if ema is not None:
            ema.restore(model)

        improved = np.isfinite(val_ccc) and val_ccc > best_ccc + args.min_delta
        if improved:
            best_ccc = val_ccc
            best_epoch = epoch
            stale_epochs = 0
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_ccc": val_ccc,
                "args": serializable_args(args),
                "combo_name": combo_name,
                "modality_dims": modality_dims,
            }
            if ema is not None:
                save_dict["ema_state_dict"] = ema.state_dict()
            torch.save(save_dict, run_dir / "model_best.pt")
        else:
            stale_epochs += 1

        current_lr = optimizer.param_groups[0]["lr"]
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
            "lr": current_lr,
        })
        write_csv(run_dir / "training_log.csv", list(log_rows[0].keys()), log_rows)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} val_ccc={val_ccc:.5f} "
            f"best_epoch={best_epoch} lr={current_lr:.2e}",
            flush=True,
        )

        if args.patience > 0 and epoch >= args.min_epochs and stale_epochs >= args.patience:
            print(f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d} best_val_ccc={best_ccc:.5f}", flush=True)
            break

    # Final evaluation with best checkpoint
    best_checkpoint = run_dir / "model_best.pt"
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        # Load EMA params for final inference if available
        if "ema_state_dict" in checkpoint:
            ema_state = checkpoint["ema_state_dict"]
            model_state = model.state_dict()
            for name in ema_state:
                if name in model_state:
                    model_state[name] = ema_state[name]
            model.load_state_dict(model_state)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        reconstructed = reconstruct(model, val_dataset, val_loader, device)
        write_group_metrics(run_dir, reconstructed)
        write_organizer_submission_tree(run_dir / "val_submission_format", reconstructed)

        for test_split in args.test_splits or []:
            test_samples = read_group_multimodal_window_manifest(args.manifest, PROJECT_ROOT, split=test_split)
            if not test_samples:
                continue
            test_dataset = GroupMultimodalWindowDataset(test_samples, min_frames=args.min_window_frames)
            test_loader = DataLoader(
                test_dataset, batch_size=args.val_batch_size, shuffle=False,
                num_workers=args.num_workers, collate_fn=group_multimodal_window_collate_fn,
            )
            test_reconstructed = reconstruct(model, test_dataset, test_loader, device)
            write_organizer_submission_tree(run_dir / f"{test_split}_submission_format", test_reconstructed)
            print(f"test_split={test_split} sessions={len(test_reconstructed)}", flush=True)

    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
