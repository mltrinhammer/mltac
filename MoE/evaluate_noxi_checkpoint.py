"""Evaluate a trained NOXI turn-TCN checkpoint on any manifest split."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_tcn_turns import build_model, reconstruct_validation
from src.acm_pipeline.dyadic_train_utils import (
    grouped_dyadic_metric_outputs,
    write_dyadic_prediction_csv,
    write_organizer_submission_tree,
)
from src.acm_pipeline.turn_data import TurnDataset, read_turn_manifest, turn_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a NOXI/NOXI-J dyadic turn checkpoint with full-session reconstruction."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="train_internal")
    parser.add_argument("--checkpoint", default="model_best.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-turn-frames", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(name)


def model_args_from_config(config: dict[str, object]) -> Namespace:
    return Namespace(
        model=str(config["model"]),
        hidden_channels=int(config["hidden_channels"]),
        levels=int(config["levels"]),
        kernel_size=int(config["kernel_size"]),
        dropout=float(config["dropout"]),
        attention_context=str(config.get("attention_context", "joint")),
        attention_heads=int(config.get("attention_heads", 4)),
        attention_past_frames=int(config.get("attention_past_frames", 1500)),
        exclude_current_frame=bool(config.get("exclude_current_frame", False)),
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = Path(str(config["manifest"]))
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest

    turns = read_turn_manifest(manifest, PROJECT_ROOT, split=args.split)
    if not turns:
        raise RuntimeError(f"No turn rows found for split {args.split!r}.")

    min_turn_frames = (
        args.min_turn_frames
        if args.min_turn_frames is not None
        else int(config.get("min_turn_frames", 5))
    )
    dataset = TurnDataset(turns, min_frames=min_turn_frames)
    if len(dataset) == 0:
        raise RuntimeError(f"No usable turn rows for split {args.split!r}.")

    device = resolve_device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=turn_collate_fn,
    )
    model = build_model(model_args_from_config(config), int(config["n_features_per_role"])).to(device)
    checkpoint = torch.load(run_dir / args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    output_dir = args.output_dir or run_dir / "diagnostics" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstructed = reconstruct_validation(model, dataset, loader, device)
    metrics = grouped_dyadic_metric_outputs(output_dir, reconstructed)
    write_dyadic_prediction_csv(output_dir / "val_predictions.csv", reconstructed)
    write_organizer_submission_tree(output_dir / "submission_format", reconstructed)
    print(
        f"split={args.split} turns={len(dataset)} ccc={metrics['ccc']:.6f} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
