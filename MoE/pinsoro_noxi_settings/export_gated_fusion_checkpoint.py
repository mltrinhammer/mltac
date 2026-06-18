"""Export validation/test predictions from a projected-fusion checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from src.acm_pipeline.pinsoro_data import PinSoRoWindowDataset, read_pinsoro_window_manifests  # noqa: E402
from src.acm_pipeline.pinsoro_train_utils import (  # noqa: E402
    prediction_coverage_rows,
    write_csv,
    write_metric_outputs,
    write_pinsoro_submission_tree,
    write_prediction_scores,
    write_predictions,
    write_test_predictions,
)
from train_pinsoro_tcn import load_checkpoint, make_loader, resolve_device  # noqa: E402
from train_gated_fusion import ProjectedFusionDyadicSharedTCN, filter_domain, modality_dims, reconstruct  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export projected-fusion checkpoint predictions.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="model_best.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--include-train", action="store_true", help="Also export train_internal prediction scores under diagnostics/train_internal.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    manifests = [Path(path) for path in config["manifest"]]
    device = resolve_device(args.device)
    dims = modality_dims(manifests, str(config["train_split"]))
    model = ProjectedFusionDyadicSharedTCN(
        modality_dims=dims,
        fusion_mode=str(config["fusion_mode"]),
        fusion_channels=int(config["fusion_channels"]),
        hidden_channels=int(config["hidden_channels"]),
        levels=int(config["levels"]),
        kernel_size=int(config["kernel_size"]),
        dropout=float(config["dropout"]),
        modality_dropout=float(config["modality_dropout"]),
        causal_tcn=bool(config["causal_tcn"]),
    ).to(device)
    model.load_state_dict(load_checkpoint(args.run_dir / args.checkpoint, device)["model_state_dict"])
    pin_memory = device.type == "cuda"
    coverage_rows = []
    split_jobs = []
    if args.include_train:
        split_jobs.append((str(config["train_split"]), write_predictions, "diagnostics/train_internal/val_predictions.csv", "diagnostics/train_internal/val_prediction_scores.csv.gz", "train_internal"))
    split_jobs.extend([
        (str(config["val_split"]), write_predictions, "val_predictions.csv", "val_prediction_scores.csv.gz", "validation"),
        (str(config["test_split"]), write_test_predictions, "test_predictions.csv", "test_prediction_scores.csv.gz", "test"),
    ])
    for split_name, writer, pred_name, score_name, coverage_name in split_jobs:
        windows = filter_domain(read_pinsoro_window_manifests(manifests, PROJECT_ROOT, split_name), str(config["domain_scope"]))
        if not windows:
            continue
        dataset = PinSoRoWindowDataset(
            windows,
            max_cached_tensors=int(config["max_cached_tensors"]),
            mmap_cache_root=Path(config["mmap_cache_root"]) if config.get("mmap_cache_root") else None,
            project_root=PROJECT_ROOT,
        )
        loader = make_loader(dataset, argparse.Namespace(**config), shuffle=False, pin_memory=pin_memory)
        reconstructed = reconstruct(model, dataset, loader, device)
        if coverage_name == "validation":
            write_metric_outputs(args.run_dir, reconstructed)
            writer(args.run_dir / pred_name, reconstructed)
            write_prediction_scores(args.run_dir / score_name, reconstructed)
        elif coverage_name == "test":
            writer(args.run_dir / pred_name, reconstructed)
            write_prediction_scores(args.run_dir / score_name, reconstructed, supervised_only=False)
            write_pinsoro_submission_tree(args.run_dir / "test_submission_format", reconstructed)
        else:
            writer(args.run_dir / pred_name, reconstructed)
            write_prediction_scores(args.run_dir / score_name, reconstructed)
        coverage_rows.extend(prediction_coverage_rows(reconstructed, coverage_name))
    write_csv(args.run_dir / "prediction_coverage.csv", coverage_rows)
    print(f"Exported {args.checkpoint} for {args.run_dir}", flush=True)


if __name__ == "__main__":
    main()
