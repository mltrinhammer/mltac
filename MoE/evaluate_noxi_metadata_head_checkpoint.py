"""Evaluate a trained NOXI metadata-head checkpoint on any manifest split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MoE.train_noxi_metadata_head_tcn import (
    MetadataStats,
    MetadataTurnDataset,
    build_model,
    metadata_collate_fn,
    read_metadata,
    reconstruct_validation,
)
from src.acm_pipeline.dyadic_train_utils import (
    grouped_dyadic_metric_outputs,
    write_dyadic_prediction_csv,
    write_organizer_submission_tree,
)
from src.acm_pipeline.turn_data import read_turn_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a NOXI metadata-head checkpoint.")
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


def resolve_config_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    project_candidate = PROJECT_ROOT / path
    if project_candidate.exists():
        return project_candidate
    repo_candidate = PROJECT_ROOT.parent / path
    if repo_candidate.exists():
        return repo_candidate
    return project_candidate


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = resolve_config_path(str(config["manifest"]))
    metadata_path = resolve_config_path(str(config["metadata"]))

    turns = read_turn_manifest(manifest, PROJECT_ROOT, split=args.split)
    if not turns:
        raise RuntimeError(f"No turn rows found for split {args.split!r}.")
    stats_config = config["metadata_stats"]
    stats = MetadataStats(
        age_mean=float(stats_config["age_mean"]),
        age_std=float(stats_config["age_std"]),
        languages=tuple(str(item) for item in stats_config["languages"]),
    )
    metadata_table = read_metadata(metadata_path)
    min_turn_frames = args.min_turn_frames if args.min_turn_frames is not None else int(config.get("min_turn_frames", 5))
    dataset = MetadataTurnDataset(
        turns,
        metadata_table,
        stats,
        str(config.get("metadata_mode", "age_gender_language")),
        min_frames=min_turn_frames,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No usable turn rows for split {args.split!r}.")

    device = resolve_device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=metadata_collate_fn,
    )
    model = build_model(argparse.Namespace(**config), int(config["n_features_per_role"]), int(config["metadata_dim"])).to(device)
    checkpoint = torch.load(run_dir / args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    output_dir = args.output_dir or run_dir / "diagnostics" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstructed = reconstruct_validation(model, dataset, loader, device)
    metrics = grouped_dyadic_metric_outputs(output_dir, reconstructed)
    write_dyadic_prediction_csv(output_dir / "val_predictions.csv", reconstructed)
    write_organizer_submission_tree(output_dir / "submission_format", reconstructed)
    print(f"split={args.split} turns={len(dataset)} ccc={metrics['ccc']:.6f} output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
