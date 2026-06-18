"""Evaluate a metadata-head MoE expert checkpoint on one split."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MOE_ROOT = PROJECT_ROOT / "MoE"
if str(MOE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOE_ROOT))

from train_moe1_metadata_head_tcn import (  # noqa: E402
    METADATA_FIELDS,
    MetadataWindowDataset,
    PinSoRoDyadicMetadataHeadTCN,
    make_loader,
    read_metadata,
    reconstruct,
)
from src.acm_pipeline.pinsoro_data import read_pinsoro_window_manifests  # noqa: E402
from src.acm_pipeline.pinsoro_train_utils import (  # noqa: E402
    prediction_coverage_rows,
    write_csv,
    write_metric_outputs,
    write_prediction_scores,
    write_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a metadata-head PinSoRo checkpoint.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="train_internal")
    parser.add_argument("--checkpoint", default="model_best.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-cached-tensors", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--supervised-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When writing prediction scores, keep only canonical-supervised frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifests = [Path(path) for path in config["manifest"]]
    windows = read_pinsoro_window_manifests(manifests, PROJECT_ROOT, args.split)
    if not windows:
        raise RuntimeError(f"No windows found for split {args.split!r}.")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    metadata_path = Path(config["metadata"])
    table = read_metadata(metadata_path)
    mmap_config = config.get("mmap_cache_root")
    mmap_cache_root = None if mmap_config is None else Path(mmap_config)
    dataset = MetadataWindowDataset(
        windows,
        args.max_cached_tensors,
        mmap_cache_root,
        PROJECT_ROOT,
        metadata_table=table,
        metadata_stats=config["metadata_stats"],
        metadata_mode=config["metadata_mode"],
    )
    loader_args = Namespace(batch_size=args.batch_size, num_workers=args.num_workers, seed=int(config["seed"]))
    loader = make_loader(dataset, loader_args, shuffle=False, pin_memory=device.type == "cuda")
    model = PinSoRoDyadicMetadataHeadTCN(
        int(config["n_features_per_role"]),
        len(METADATA_FIELDS),
        int(config["hidden_channels"]),
        int(config["levels"]),
        int(config["kernel_size"]),
        float(config["dropout"]),
        float(config["metadata_dropout"]),
        bool(config.get("causal_tcn", True)),
    ).to(device)
    checkpoint = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    output_dir = args.output_dir or run_dir / "diagnostics" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstructed = reconstruct(model, dataset, loader, device)
    metrics = write_metric_outputs(output_dir, reconstructed)
    write_predictions(output_dir / "val_predictions.csv", reconstructed)
    write_prediction_scores(
        output_dir / "val_prediction_scores.csv.gz",
        reconstructed,
        supervised_only=args.supervised_only,
    )
    write_csv(output_dir / "prediction_coverage.csv", prediction_coverage_rows(reconstructed, args.split))
    print(
        f"split={args.split} windows={len(windows)} organizer_score={metrics['organizer_score']:.6f} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
