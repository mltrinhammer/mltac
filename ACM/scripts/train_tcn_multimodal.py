"""Train multimodal turn-level TCN baselines from paired multimodal turn manifests.

This trainer mirrors the unimodal turn-level training contract but consumes a
joined multimodal turn manifest and applies within-role multimodal fusion
before one of the retained TCN backbones predicts both novice and expert
engagement on every interval.
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

from src.acm_pipeline.dyadic_train_utils import (
    grouped_dyadic_metric_outputs,
    write_csv,
    write_dyadic_prediction_csv,
    write_organizer_submission_tree,
)
from src.acm_pipeline.metrics import ccc_loss, masked_mse_loss
from src.acm_pipeline.models_tcn import MultimodalTurnTCNRegressor
from src.acm_pipeline.turn_data import (
    MultimodalManifestTurnSample,
    MultimodalTurnDataset,
    multimodal_turn_collate_fn,
    read_multimodal_turn_manifest,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal turn-segmented TCN models from paired multimodal supervision.")
    parser.add_argument("--manifest", type=Path, required=True, help="Multimodal paired turn manifest CSV.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--train-split", default="train_internal")
    parser.add_argument("--val-split", default="val_internal")
    parser.add_argument("--test-splits", nargs="*", default=["test_internal", "test_additional"],
                        help="Splits to run test inference on after training.")

    parser.add_argument("--backbone", choices=["simple", "dyadic_shared", "attention"], default="dyadic_shared")
    parser.add_argument("--fusion-mode", choices=["gated", "concat"], default="gated")
    parser.add_argument("--fusion-channels", type=int, default=64)
    parser.add_argument("--modality-dropout", type=float, default=0.1)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attention-context", choices=["self", "partner", "joint"], default="joint")
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-past-frames", type=int, default=1500)
    parser.add_argument("--exclude-current-frame", action="store_true")

    parser.add_argument("--min-turn-frames", type=int, default=5)
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
    run_name = args.run_name.strip() or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_tcn_multimodal_{args.backbone}_{args.fusion_mode}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def move_modalities_to_device(x_modalities: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in x_modalities.items()}


def infer_layout(turns: list[MultimodalManifestTurnSample]) -> tuple[str, tuple[str, ...], dict[str, int]]:
    if not turns:
        raise RuntimeError("At least one multimodal turn row is required.")

    combo_name = turns[0].combo_name
    modality_order = turns[0].modality_order
    modality_dims: dict[str, int] = {}
    for turn in turns:
        if turn.combo_name != combo_name:
            raise RuntimeError(f"Expected one combo_name, got {combo_name!r} and {turn.combo_name!r}")
        if turn.modality_order != modality_order:
            raise RuntimeError("All multimodal turns must share the same modality order.")
        for modality_name in modality_order:
            novice_dim = turn.novice_examples[modality_name].n_features
            expert_dim = turn.expert_examples[modality_name].n_features
            if novice_dim != expert_dim:
                raise RuntimeError(
                    f"Modality {modality_name!r} has mismatched role dimensions: novice={novice_dim}, expert={expert_dim}"
                )
            current = modality_dims.get(modality_name)
            if current is None:
                modality_dims[modality_name] = novice_dim
            elif current != novice_dim:
                raise RuntimeError(
                    f"Modality {modality_name!r} has inconsistent dimensions across turns: {current} vs {novice_dim}"
                )
    return combo_name, modality_order, modality_dims


def build_model(args: argparse.Namespace, modality_dims: dict[str, int]) -> torch.nn.Module:
    return MultimodalTurnTCNRegressor(
        modality_dims=modality_dims,
        backbone_model=args.backbone,
        fusion_channels=args.fusion_channels,
        fusion_mode=args.fusion_mode,
        modality_dropout=args.modality_dropout,
        hidden_channels=args.hidden_channels,
        levels=args.levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        attention_context=args.attention_context,
        attention_heads=args.attention_heads,
        attention_past_frames=args.attention_past_frames,
        exclude_current_frame=args.exclude_current_frame,
    )


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
        x_modalities = move_modalities_to_device(batch["x_modalities"], device)
        y = batch["y"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_modalities)
        loss = masked_mse_loss(pred, y, loss_mask) + ccc_weight * ccc_loss(pred, y, loss_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def reconstruct_validation(
    model: torch.nn.Module,
    dataset: MultimodalTurnDataset,
    loader: DataLoader,
    device: torch.device,
    collect_gate_weights: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model.eval()

    session_info: dict[str, dict[str, object]] = {}
    for turn in dataset.turns:
        key = turn.session_key
        if key in session_info:
            continue
        reference_modality = turn.modality_order[0]
        novice_example = turn.novice_examples[reference_modality]
        expert_example = turn.expert_examples[reference_modality]
        session_len = min(novice_example.aligned_len, expert_example.aligned_len)
        session_info[key] = {
            "aligned_len": session_len,
            "novice_example": novice_example,
            "expert_example": expert_example,
            "dataset": turn.dataset,
            "session_id": turn.session_id,
            "model_split": turn.model_split,
        }

    sums: dict[str, np.ndarray] = {key: np.zeros((info["aligned_len"], 2), dtype=np.float64) for key, info in session_info.items()}
    counts: dict[str, np.ndarray] = {key: np.zeros(info["aligned_len"], dtype=np.float64) for key, info in session_info.items()}

    gate_sums: dict[str, np.ndarray] = {}
    gate_counts: dict[str, float] = {}
    modality_order: list[str] = []

    for batch in loader:
        x_modalities = move_modalities_to_device(batch["x_modalities"], device)
        if collect_gate_weights:
            pred_tensor, gate_info = model(x_modalities, return_gate_weights=True)
        else:
            pred_tensor = model(x_modalities)
            gate_info = None

        pred = pred_tensor.detach().cpu().numpy()
        frame_mask = batch["frame_mask"].numpy()
        session_keys = batch["session_keys"]
        start_frames = batch["start_frames"].numpy()

        if collect_gate_weights and gate_info is not None and gate_info.get("novice_weights") is not None:
            modality_order = list(gate_info["modality_order"])
            novice_weights = gate_info["novice_weights"].detach().cpu().numpy()
            expert_weights = gate_info["expert_weights"].detach().cpu().numpy()
            if not gate_sums:
                gate_sums = {
                    "novice": np.zeros(len(modality_order), dtype=np.float64),
                    "expert": np.zeros(len(modality_order), dtype=np.float64),
                }
                gate_counts = {"novice": 0.0, "expert": 0.0}

            for row in range(pred.shape[0]):
                valid_len = int(frame_mask[row].sum())
                if valid_len <= 0:
                    continue
                gate_sums["novice"] += novice_weights[row, :valid_len].sum(axis=0)
                gate_sums["expert"] += expert_weights[row, :valid_len].sum(axis=0)
                gate_counts["novice"] += valid_len
                gate_counts["expert"] += valid_len

        for row in range(pred.shape[0]):
            key = session_keys[row]
            start = int(start_frames[row])
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
            [novice_session.target_mask[:aligned_len], expert_session.target_mask[:aligned_len]],
            axis=1,
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

    gate_rows: list[dict[str, object]] = []
    if gate_sums and modality_order:
        for role in ("novice", "expert"):
            denominator = gate_counts[role] if gate_counts[role] > 0 else 1.0
            for idx, modality_name in enumerate(modality_order):
                gate_rows.append(
                    {
                        "role": role,
                        "modality": modality_name,
                        "mean_gate_weight": gate_sums[role][idx] / denominator,
                    }
                )

    return reconstructed, gate_rows


class _SessionStub:
    def __init__(self, dataset: str, session_id: str, model_split: str, role_names: tuple[str, ...] = ("novice", "expert")) -> None:
        self.dataset = dataset
        self.session_id = session_id
        self.model_split = model_split
        self.role_names = role_names


def main() -> None:
    args = parse_args()
    if args.min_epochs < 0:
        raise ValueError("--min-epochs must be non-negative.")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be non-negative.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = make_run_dir(args)

    train_turns = read_multimodal_turn_manifest(args.manifest, PROJECT_ROOT, split=args.train_split)
    val_turns = read_multimodal_turn_manifest(args.manifest, PROJECT_ROOT, split=args.val_split)
    if not train_turns or not val_turns:
        raise RuntimeError("Both train and validation multimodal turn rows are required.")

    combo_name, modality_order, modality_dims = infer_layout(train_turns + val_turns)
    train_dataset = MultimodalTurnDataset(train_turns, min_frames=args.min_turn_frames)
    val_dataset = MultimodalTurnDataset(val_turns, min_frames=args.min_turn_frames)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=multimodal_turn_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=multimodal_turn_collate_fn,
    )

    config = serializable_args(args)
    config.update(
        {
            "run_dir": str(run_dir),
            "combo_name": combo_name,
            "modality_order": list(modality_order),
            "modality_dims": modality_dims,
            "n_modalities": len(modality_order),
            "output_dim": 2,
            "n_train_turns": len(train_dataset),
            "n_val_turns": len(val_dataset),
        }
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    model = build_model(args, modality_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ccc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    log_rows: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, ccc_weight=args.ccc_weight)
        reconstructed, _gate_rows = reconstruct_validation(model, val_dataset, val_loader, device, collect_gate_weights=False)
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
                    "combo_name": combo_name,
                    "modality_dims": modality_dims,
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
        reconstructed, gate_rows = reconstruct_validation(model, val_dataset, val_loader, device, collect_gate_weights=True)
        grouped_dyadic_metric_outputs(run_dir, reconstructed)
        write_dyadic_prediction_csv(run_dir / "val_predictions.csv", reconstructed)
        write_organizer_submission_tree(run_dir / "val_submission_format", reconstructed)
        if gate_rows:
            write_csv(run_dir / "val_gate_weights.csv", list(gate_rows[0].keys()), gate_rows)

        # Test-split inference: generate submission-format predictions for
        # held-out test sessions (labels withheld by organizers).
        for test_split in (args.test_splits or []):
            test_turns = read_multimodal_turn_manifest(args.manifest, PROJECT_ROOT, split=test_split)
            if not test_turns:
                continue
            test_dataset = MultimodalTurnDataset(test_turns, min_frames=args.min_turn_frames)
            if len(test_dataset) == 0:
                continue
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=multimodal_turn_collate_fn,
            )
            test_reconstructed, _ = reconstruct_validation(model, test_dataset, test_loader, device)
            write_organizer_submission_tree(run_dir / "test_submission_format", test_reconstructed)
            print(f"test_split={test_split}  sessions={len(test_reconstructed)}", flush=True)

    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()