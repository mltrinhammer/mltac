"""Train PinSoRo projected multimodal dyadic-shared classifiers.

This experiment mirrors the NOXI settings file at the architecture level while
keeping PinSoRo's classification loss and organizer-score evaluation.
"""

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
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from src.acm_pipeline.models_tcn import TemporalBlock  # noqa: E402
from src.acm_pipeline.pinsoro_data import (  # noqa: E402
    PinSoRoWindowDataset,
    read_pinsoro_window_manifest,
    read_pinsoro_window_manifests,
)
from src.acm_pipeline.pinsoro_train_utils import (  # noqa: E402
    CLASS_COUNTS,
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
from train_pinsoro_tcn import (  # noqa: E402
    compute_class_weights,
    compute_cr_social_weights,
    load_checkpoint,
    make_loader,
    resolve_device,
    save_checkpoint,
    serializable_args,
    set_seed,
)


DEFAULT_FEATURES = ("audio_w2vbert2", "text_xlm_roberta", "visual_videomae")


class RoleProjectedFusion(nn.Module):
    """Project modalities separately per role, then concat or softmax-gate them."""

    def __init__(
        self,
        modality_dims: dict[str, int],
        fusion_channels: int,
        modality_dropout: float,
        fusion_mode: str,
    ) -> None:
        super().__init__()
        if not modality_dims:
            raise ValueError("At least one modality is required.")
        if not 0.0 <= modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1).")
        if fusion_mode not in {"gated", "concat"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.modality_order = tuple(modality_dims)
        self.modality_dims = dict(modality_dims)
        self.fusion_channels = fusion_channels
        self.modality_dropout = modality_dropout
        self.fusion_mode = fusion_mode
        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(dim, fusion_channels)
                for name, dim in self.modality_dims.items()
            }
        )
        gate_hidden = max(1, fusion_channels // 2)
        self.gates = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(fusion_channels, gate_hidden),
                    nn.ReLU(),
                    nn.Linear(gate_hidden, 1),
                )
                for name in self.modality_order
            }
        )

    @property
    def fused_channels_per_role(self) -> int:
        if self.fusion_mode == "gated":
            return self.fusion_channels
        return self.fusion_channels * len(self.modality_order)

    def _keep_mask(self, device: torch.device) -> torch.Tensor | None:
        if not self.training or self.modality_dropout <= 0.0 or len(self.modality_order) <= 1:
            return None
        keep = torch.rand(len(self.modality_order), device=device) >= self.modality_dropout
        if not torch.any(keep):
            keep[torch.randint(len(self.modality_order), size=(1,), device=device)] = True
        return keep

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
        # x: [B, 2, sum(F_m), T]
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError(f"Expected dyadic x [B, 2, F, T], got {tuple(x.shape)}")
        pieces = torch.split(x, [self.modality_dims[name] for name in self.modality_order], dim=2)
        keep = self._keep_mask(x.device)
        fused_roles = []
        weight_roles = []
        for role_idx in range(2):
            projected = []
            for modality_name, piece in zip(self.modality_order, pieces):
                # [B, F, T] -> [B, T, F] -> [B, T, C]
                projected.append(self.projections[modality_name](piece[:, role_idx].transpose(1, 2)))
            if keep is not None:
                projected = [
                    value if bool(keep[idx].item()) else torch.zeros_like(value)
                    for idx, value in enumerate(projected)
                ]
            if self.fusion_mode == "concat":
                fused_roles.append(torch.cat(projected, dim=-1).transpose(1, 2))
                weight_roles.append(None)
                continue
            logits = torch.cat(
                [self.gates[name](value) for name, value in zip(self.modality_order, projected)],
                dim=-1,
            )
            if keep is not None:
                logits = logits.masked_fill(~keep.view(1, 1, -1), -1.0e9)
            weights = torch.softmax(logits, dim=-1)
            fused = sum(weights[..., idx : idx + 1] * value for idx, value in enumerate(projected))
            fused_roles.append(fused.transpose(1, 2))
            weight_roles.append(weights)
        return torch.stack(fused_roles, dim=1), {
            "modality_order": list(self.modality_order),
            "fusion_mode": self.fusion_mode,
            "purple_weights": weight_roles[0],
            "yellow_weights": weight_roles[1],
        }


class ProjectedFusionDyadicSharedTCN(nn.Module):
    """Projected multimodal fusion followed by a dyadic shared TCN classifier."""

    def __init__(
        self,
        modality_dims: dict[str, int],
        fusion_mode: str = "gated",
        fusion_channels: int = 64,
        hidden_channels: int = 64,
        levels: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.2,
        modality_dropout: float = 0.1,
        causal_tcn: bool = False,
    ) -> None:
        super().__init__()
        self.fusion = RoleProjectedFusion(modality_dims, fusion_channels, modality_dropout, fusion_mode)
        channels = [2 * self.fusion.fused_channels_per_role] + [hidden_channels] * levels
        self.encoder = nn.Sequential(
            *[
                TemporalBlock(channels[idx], channels[idx + 1], kernel_size, 2**idx, dropout, causal=causal_tcn)
                for idx in range(levels)
            ]
        )
        self.task_head = nn.Conv1d(hidden_channels, 2 * CLASS_COUNTS["task"], kernel_size=1)
        self.social_head = nn.Conv1d(hidden_channels, 2 * CLASS_COUNTS["social"], kernel_size=1)

    @staticmethod
    def _reshape(logits: torch.Tensor, classes: int) -> torch.Tensor:
        batch, _channels, time = logits.shape
        return logits.reshape(batch, 2, classes, time).permute(0, 1, 3, 2)

    def forward(
        self,
        x: torch.Tensor,
        domain_ids: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _ = metadata
        fused, _info = self.fusion(x)
        batch, roles, features, time = fused.shape
        hidden = self.encoder(fused.reshape(batch, roles * features, time))
        return {
            "task": self._reshape(self.task_head(hidden), CLASS_COUNTS["task"]),
            "social": self._reshape(self.social_head(hidden), CLASS_COUNTS["social"]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PinSoRo projected early-fusion model.")
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "MoE" / "experiments" / "pinsoro_noxi_settings_gated_fusion")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--domain-scope", choices=("both", "CC", "CR"), default="both")
    parser.add_argument("--train-split", default="train_internal")
    parser.add_argument("--val-split", default="val_internal")
    parser.add_argument("--test-split", default="test_internal")
    parser.add_argument("--fusion-mode", choices=("gated", "concat"), default="gated")
    parser.add_argument("--fusion-channels", type=int, default=64)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--modality-dropout", type=float, default=0.1)
    parser.add_argument("--causal-tcn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cr-social-weighting", choices=("shared_inverse", "unweighted", "sqrt_inverse", "capped_inverse", "targeted"), default="shared_inverse")
    parser.add_argument("--cr-social-weight-cap", type=float, default=5.0)
    parser.add_argument("--cr-social-target-class2-weight", type=float, default=2.0)
    parser.add_argument("--cr-social-target-class3-weight", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-cached-tensors", type=int, default=6)
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


def filter_domain(windows: list, domain_scope: str) -> list:
    if domain_scope == "both":
        return windows
    return [window for window in windows if window.domain == domain_scope]


def modality_dims(manifests: list[Path], split: str) -> dict[str, int]:
    dims: dict[str, int] = {}
    for manifest in manifests:
        windows = read_pinsoro_window_manifest(manifest, PROJECT_ROOT, split)
        if not windows:
            windows = read_pinsoro_window_manifest(manifest, PROJECT_ROOT, None)
        if not windows:
            raise RuntimeError(f"No windows found in {manifest}")
        feature_sets = {window.feature_set for window in windows}
        feature_dims = {window.n_features_per_role for window in windows}
        if len(feature_sets) != 1 or len(feature_dims) != 1:
            raise RuntimeError(f"Expected one feature/dim in {manifest}, got {feature_sets} / {feature_dims}")
        dims[next(iter(feature_sets))] = next(iter(feature_dims))
    return dims


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: dict[str, torch.Tensor],
    soft_label_mode: str = "none",
) -> float:
    model.train()
    total = torch.zeros((), device=device)
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
            model(batch["x"], batch["domain_id"]),
            batch,
            class_weights,
            soft_label_mode=soft_label_mode,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += loss.detach()
        n_batches += 1
    return float((total / n_batches).item()) if n_batches else float("nan")


@torch.inference_mode()
def reconstruct(model: nn.Module, dataset: PinSoRoWindowDataset, loader, device: torch.device) -> list[dict[str, object]]:
    model.eval()
    accumulators: dict[tuple[str, str, str, str], dict[str, object]] = {}
    non_blocking = device.type == "cuda"
    for batch in loader:
        indices = batch["window_indices"].numpy()
        metadata = batch.get("metadata")
        if metadata is not None:
            metadata = metadata.to(device, non_blocking=non_blocking)
        logits = model(
            batch["x"].to(device, non_blocking=non_blocking),
            batch["domain_id"].to(device, non_blocking=non_blocking),
            metadata=metadata,
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
                        "task_sum": np.zeros((window.session_aligned_len, CLASS_COUNTS["task"]), dtype=np.float64),
                        "social_sum": np.zeros((window.session_aligned_len, CLASS_COUNTS["social"]), dtype=np.float64),
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


def main() -> None:
    args = parse_args()
    if args.mmap_cache_root is not None and not args.mmap_cache_root.is_absolute():
        args.mmap_cache_root = PROJECT_ROOT / args.mmap_cache_root
    set_seed(args.seed)
    device = resolve_device(args.device)
    dims = modality_dims(args.manifest, args.train_split)

    train_windows = filter_domain(read_pinsoro_window_manifests(args.manifest, PROJECT_ROOT, args.train_split), args.domain_scope)
    val_windows = filter_domain(read_pinsoro_window_manifests(args.manifest, PROJECT_ROOT, args.val_split), args.domain_scope)
    if not train_windows or not val_windows:
        raise RuntimeError(f"Missing train/val windows for domain_scope={args.domain_scope}")
    if {len(window.roles) for window in train_windows + val_windows} != {2}:
        raise RuntimeError("Projected dyadic fusion requires dyadic two-role manifests.")

    train_dataset = PinSoRoWindowDataset(train_windows, args.max_cached_tensors, args.mmap_cache_root, PROJECT_ROOT)
    val_dataset = PinSoRoWindowDataset(val_windows, args.max_cached_tensors, args.mmap_cache_root, PROJECT_ROOT)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_dataset, args, shuffle=True, pin_memory=pin_memory)
    val_loader = make_loader(val_dataset, args, shuffle=False, pin_memory=pin_memory)

    class_weights = {
        head: value.to(device)
        for head, value in compute_class_weights(train_windows, train_dataset).items()
    }
    cr_social_weights = compute_cr_social_weights(
        train_windows,
        train_dataset,
        args.cr_social_weighting,
        args.cr_social_weight_cap,
        args.cr_social_target_class2_weight,
        args.cr_social_target_class3_weight,
    )
    if cr_social_weights is not None:
        class_weights["cr_social"] = cr_social_weights.to(device)

    model = ProjectedFusionDyadicSharedTCN(
        modality_dims=dims,
        fusion_mode=args.fusion_mode,
        fusion_channels=args.fusion_channels,
        hidden_channels=args.hidden_channels,
        levels=args.levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        modality_dropout=args.modality_dropout,
        causal_tcn=args.causal_tcn,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    feature_name = "__".join(dims)
    run_name = args.run_name or (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pinsoro_{args.domain_scope.lower()}_"
        f"{feature_name}_{args.fusion_mode}_dyadic_shared_seed{args.seed}"
    )
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = serializable_args(args) | {
        "architecture": "projected_multimodal_dyadic_shared_tcn",
        "fusion_mode": args.fusion_mode,
        "classification_loss": "weighted multitask cross entropy",
        "soft_label_mode": args.soft_label_mode,
        "organizer_score": "mean validation Cohen kappa across available domain x task/social rows",
        "modality_dims": dims,
        "n_train_windows": len(train_dataset),
        "n_val_windows": len(val_dataset),
        "class_weights": {head: value.detach().cpu().tolist() for head, value in class_weights.items()},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    log_rows: list[dict[str, object]] = []
    start_epoch = 1
    last_checkpoint_path = run_dir / "model_last.pt"
    if args.resume and last_checkpoint_path.exists():
        checkpoint = load_checkpoint(last_checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_val_organizer_score"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        log_rows = list(checkpoint["log_rows"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state_all"]])

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_started = time.perf_counter()
        train_loss = train_epoch(
            model, train_loader, optimizer, device, class_weights, args.soft_label_mode
        )
        train_seconds = time.perf_counter() - train_started
        val_started = time.perf_counter()
        reconstructed = reconstruct(model, val_dataset, val_loader, device)
        val_metrics = write_metric_outputs(run_dir, reconstructed)
        val_seconds = time.perf_counter() - val_started
        score = val_metrics["organizer_score"]
        improved = np.isfinite(score) and score > best_score + args.min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                run_dir / "model_best.pt",
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_organizer_score": score},
            )
        else:
            stale_epochs += 1
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_organizer_score": score,
                "best_epoch": best_epoch,
                "best_val_organizer_score": best_score,
                "stale_epochs": stale_epochs,
                "train_seconds": train_seconds,
                "val_seconds": val_seconds,
                "epoch_seconds": time.perf_counter() - started,
            }
        )
        write_csv(run_dir / "training_log.csv", log_rows)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_val_organizer_score": best_score,
            "stale_epochs": stale_epochs,
            "log_rows": log_rows,
            "torch_rng_state": torch.get_rng_state(),
        }
        if device.type == "cuda":
            checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        save_checkpoint(last_checkpoint_path, checkpoint)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"val_organizer_score={score:.5f} best_epoch={best_epoch}",
            flush=True,
        )
        if args.patience > 0 and epoch >= args.min_epochs and stale_epochs >= args.patience:
            break

    if (run_dir / "model_best.pt").exists():
        model.load_state_dict(load_checkpoint(run_dir / "model_best.pt", device)["model_state_dict"])
        reconstructed = reconstruct(model, val_dataset, val_loader, device)
        write_metric_outputs(run_dir, reconstructed)
        coverage_rows = prediction_coverage_rows(reconstructed, "validation")
        write_predictions(run_dir / "val_predictions.csv", reconstructed)
        write_prediction_scores(run_dir / "val_prediction_scores.csv.gz", reconstructed)
        test_windows = filter_domain(read_pinsoro_window_manifests(args.manifest, PROJECT_ROOT, args.test_split), args.domain_scope)
        if test_windows:
            test_dataset = PinSoRoWindowDataset(test_windows, args.max_cached_tensors, args.mmap_cache_root, PROJECT_ROOT)
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
