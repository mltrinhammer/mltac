"""Run test-only multimodal turn inference from a trained checkpoint.

Supports an ``--aggregate-pairs`` mode for datasets preprocessed with
``build_allpairs_turn_manifest.py``: composite session IDs are parsed back
into real session IDs and per-participant predictions are averaged across all
C(N,2) pairs before writing the organizer submission tree.
"""

from __future__ import annotations

import argparse
import json
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

from src.acm_pipeline.dyadic_train_utils import write_csv, write_organizer_submission_tree
from src.acm_pipeline.models_tcn import MultimodalTurnTCNRegressor
from src.acm_pipeline.turn_data import (
    MultimodalTurnDataset,
    multimodal_turn_collate_fn,
    read_multimodal_turn_manifest,
)

# Import from sibling script — works because PROJECT_ROOT/scripts is on sys.path
# via the ``if str(PROJECT_ROOT) not in sys.path`` block above and CWD convention.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from train_tcn_multimodal import infer_layout, reconstruct_validation, _SessionStub


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test-only multimodal turn inference from a trained checkpoint.")
    parser.add_argument("--manifest", type=Path, required=True, help="Multimodal paired turn manifest CSV.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model_best.pt from train_tcn_multimodal.py")
    parser.add_argument("--test-split", default="test", help="Manifest split label to export.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--min-turn-frames", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-gates", action="store_true", help="Save mean gate weights (when checkpoint uses gated fusion).")
    parser.add_argument("--aggregate-pairs", action="store_true",
                        help="Aggregate predictions from all-pairs composite session IDs back to per-participant output.")
    parser.add_argument("--pair-separator", default="__pair__",
                        help="Separator used in composite session IDs (must match build_allpairs_turn_manifest.py).")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(name)


def make_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name.strip() or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_tcn_multimodal_test_infer"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _as_int(value: object, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _as_float(value: object, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def build_model_from_checkpoint(
    checkpoint: dict[str, object],
    modality_dims: dict[str, int],
) -> MultimodalTurnTCNRegressor:
    ckpt_args = checkpoint.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}

    model = MultimodalTurnTCNRegressor(
        modality_dims=modality_dims,
        backbone_model=str(ckpt_args.get("backbone", "dyadic_shared")),
        fusion_channels=_as_int(ckpt_args.get("fusion_channels"), 64),
        fusion_mode=str(ckpt_args.get("fusion_mode", "gated")),
        modality_dropout=_as_float(ckpt_args.get("modality_dropout"), 0.0),
        hidden_channels=_as_int(ckpt_args.get("hidden_channels"), 64),
        levels=_as_int(ckpt_args.get("levels"), 4),
        kernel_size=_as_int(ckpt_args.get("kernel_size"), 5),
        dropout=_as_float(ckpt_args.get("dropout"), 0.2),
        attention_context=str(ckpt_args.get("attention_context", "joint")),
        attention_heads=_as_int(ckpt_args.get("attention_heads"), 4),
        attention_past_frames=_as_int(ckpt_args.get("attention_past_frames"), 1500),
        exclude_current_frame=_as_bool(ckpt_args.get("exclude_current_frame"), False),
    )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain model_state_dict.")
    model.load_state_dict(state_dict)
    return model


# ---------------------------------------------------------------------------
# All-pairs aggregation
# ---------------------------------------------------------------------------


def _fill_prediction_gaps(y_pred: np.ndarray) -> np.ndarray:
    """Fill NaN gaps in prediction columns via forward-fill then backward-fill.

    This ensures every frame has a valid prediction value so the organizer
    submission passes the minimum-coverage threshold required by the evaluator.
    """
    out = y_pred.copy()
    for col in range(out.shape[1]):
        series = out[:, col]
        # Forward-fill: propagate last valid value forward.
        mask = np.isnan(series)
        if not mask.any():
            continue
        valid_idx = np.where(~mask)[0]
        if len(valid_idx) == 0:
            # No valid values at all — fill with zero.
            series[:] = 0.0
            continue
        # Forward-fill.
        for i in range(1, len(series)):
            if mask[i] and not mask[i - 1]:
                series[i] = series[i - 1]
                mask[i] = False
        # Backward-fill remaining leading NaNs.
        mask = np.isnan(series)
        if mask.any():
            first_valid = valid_idx[0]
            series[:first_valid] = series[first_valid]
    return out


def aggregate_pair_predictions(
    reconstructed: list[dict[str, object]],
    pair_separator: str,
) -> list[dict[str, object]]:
    """Merge per-pair reconstructed sessions into per-participant predictions.

    Each reconstructed item has a composite ``session_id`` of the form
    ``{real_id}{sep}{role_a}{sep}{role_b}`` where channel 0 = role_a (novice
    slot) and channel 1 = role_b (expert slot).

    For each real session, every participant's predictions are averaged across
    all C(N,2) pairs they appeared in.  The output items have ``y_pred`` of
    shape ``[aligned_len, N_participants]`` and ``role_names`` set to the
    sorted tuple of participant names.
    """

    # Group pair-level items by (dataset, real_session_id).
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for item in reconstructed:
        example = item["example"]
        composite_id = str(example.session_id)
        parts = composite_id.split(pair_separator)
        if len(parts) != 3:
            raise RuntimeError(
                f"Cannot parse composite session_id {composite_id!r} with "
                f"separator {pair_separator!r}. Expected 3 parts, got {len(parts)}."
            )
        real_session_id = parts[0]
        groups[(str(example.dataset), real_session_id)].append(item)

    aggregated: list[dict[str, object]] = []
    for (dataset, real_session_id), pair_items in sorted(groups.items()):
        # Discover all participants and determine shared aligned length.
        all_roles: set[str] = set()
        model_split = ""
        for item in pair_items:
            composite_id = str(item["example"].session_id)
            parts = composite_id.split(pair_separator)
            all_roles.add(parts[1])
            all_roles.add(parts[2])
            if not model_split:
                model_split = str(item["example"].model_split)

        sorted_roles = sorted(all_roles)
        role_to_idx = {role: idx for idx, role in enumerate(sorted_roles)}
        n_participants = len(sorted_roles)

        # Use the minimum aligned length across all pairs (should be identical
        # for participants from the same recording).
        aligned_len = min(item["y_pred"].shape[0] for item in pair_items)

        # Accumulate weighted sums per participant.
        pred_sums = np.zeros((aligned_len, n_participants), dtype=np.float64)
        pred_counts = np.zeros((aligned_len, n_participants), dtype=np.float64)
        covered_any = np.zeros(aligned_len, dtype=bool)

        for item in pair_items:
            composite_id = str(item["example"].session_id)
            parts = composite_id.split(pair_separator)
            role_a, role_b = parts[1], parts[2]
            y_pred = item["y_pred"][:aligned_len]  # [L, 2]
            covered = item["covered"][:aligned_len].astype(bool)

            covered_any |= covered

            # Channel 0 → role_a, channel 1 → role_b.
            idx_a = role_to_idx[role_a]
            idx_b = role_to_idx[role_b]

            valid_a = covered & np.isfinite(y_pred[:, 0])
            valid_b = covered & np.isfinite(y_pred[:, 1])

            pred_sums[valid_a, idx_a] += y_pred[valid_a, 0]
            pred_counts[valid_a, idx_a] += 1.0

            pred_sums[valid_b, idx_b] += y_pred[valid_b, 1]
            pred_counts[valid_b, idx_b] += 1.0

        # Average predictions per participant.
        y_pred_agg = np.full((aligned_len, n_participants), np.nan, dtype=np.float32)
        has_pred = pred_counts > 0
        y_pred_agg[has_pred] = (pred_sums[has_pred] / pred_counts[has_pred]).astype(np.float32)

        # Fill gaps between turns so every frame has a valid prediction.
        # The organizer evaluator requires ~99% valid frames; without this,
        # between-turn frames are written as empty and the submission is rejected.
        y_pred_agg = _fill_prediction_gaps(y_pred_agg)

        aggregated.append(
            {
                "example": _SessionStub(
                    dataset=dataset,
                    session_id=real_session_id,
                    model_split=model_split,
                    role_names=tuple(sorted_roles),
                ),
                "y_true": np.zeros((aligned_len, n_participants), dtype=np.float32),
                "target_mask": np.zeros((aligned_len, n_participants), dtype=np.float32),
                "y_pred": y_pred_agg,
                "covered": np.ones(aligned_len, dtype=np.float32),
            }
        )

    print(f"Aggregated {len(reconstructed)} pair sessions → {len(aggregated)} real sessions", flush=True)
    return aggregated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    run_dir = make_run_dir(args)

    test_turns = read_multimodal_turn_manifest(args.manifest, PROJECT_ROOT, split=args.test_split)
    if not test_turns:
        raise RuntimeError(f"No multimodal turn rows found for split {args.test_split!r} in {args.manifest}")

    combo_name, modality_order, modality_dims = infer_layout(test_turns)
    test_dataset = MultimodalTurnDataset(test_turns, min_frames=args.min_turn_frames)
    if len(test_dataset) == 0:
        raise RuntimeError("No test turns remain after min-turn-frames filtering.")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=multimodal_turn_collate_fn,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_dims = checkpoint.get("modality_dims")
    if isinstance(checkpoint_dims, dict):
        checkpoint_dims_cast = {str(k): int(v) for k, v in checkpoint_dims.items()}
        if checkpoint_dims_cast != modality_dims:
            raise RuntimeError(
                "Checkpoint modality dimensions do not match manifest layout: "
                f"checkpoint={checkpoint_dims_cast} manifest={modality_dims}"
            )

    model = build_model_from_checkpoint(checkpoint, modality_dims).to(device)
    reconstructed, gate_rows = reconstruct_validation(
        model,
        test_dataset,
        test_loader,
        device,
        collect_gate_weights=args.save_gates,
    )

    if args.aggregate_pairs:
        reconstructed = aggregate_pair_predictions(reconstructed, args.pair_separator)

    write_organizer_submission_tree(run_dir / "test_submission_format", reconstructed)

    config = {
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "test_split": args.test_split,
        "combo_name": combo_name,
        "modality_order": list(modality_order),
        "modality_dims": modality_dims,
        "n_test_turns": len(test_dataset),
        "n_test_sessions": len(reconstructed),
        "aggregate_pairs": args.aggregate_pairs,
        "device": str(device),
    }
    with (run_dir / "inference_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    if args.save_gates and gate_rows:
        write_csv(run_dir / "test_gate_weights.csv", list(gate_rows[0].keys()), gate_rows)

    print(f"Run directory: {run_dir}", flush=True)
    print(f"test_split={args.test_split}  sessions={len(reconstructed)}", flush=True)


if __name__ == "__main__":
    main()
