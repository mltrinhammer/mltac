"""Fit train-only normalization and export PinSoRo session tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.io import read_csv, write_csv
from src.acm_pipeline.transforms import FeatureNormalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize PinSoRo tensors using train-only statistics.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--transform-dir", type=Path, default=None)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_manifest)
    if not rows:
        raise RuntimeError(f"No rows in {args.input_manifest}")
    feature_sets = sorted({row["feature_set"] for row in rows})
    if len(feature_sets) != 1:
        raise RuntimeError(f"Expected one feature set, got {feature_sets}")
    feature_set = feature_sets[0]
    out_root = args.out_root or PROJECT_ROOT / "processed" / "pinsoro" / "transformed" / f"{feature_set}_raw"
    output_manifest = args.output_manifest or PROJECT_ROOT / "outputs" / "pinsoro" / "manifests" / f"{feature_set}_raw.csv"
    transform_dir = args.transform_dir or PROJECT_ROOT / "outputs" / "pinsoro" / "transforms" / f"{feature_set}_raw"

    train_paths = [resolve(row["tensor_relative_path"]) for row in rows if row["model_split"] == "train_internal"]
    normalizer = FeatureNormalizer.fit_npz_paths(train_paths)
    normalizer_path = transform_dir / "normalizer.npz"
    normalizer.save(normalizer_path)

    output_rows: list[dict[str, object]] = []
    for row in rows:
        source_path = resolve(row["tensor_relative_path"])
        out_dir = out_root / row["source_split"] / row["session_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{row['role']}.{feature_set}.raw.npz"
        with np.load(source_path, allow_pickle=True) as data:
            np.savez_compressed(
                out_path,
                x=normalizer.transform(np.asarray(data["x"], dtype=np.float32)),
                task_y=np.asarray(data["task_y"], dtype=np.int64),
                task_mask=np.asarray(data["task_mask"], dtype=np.float32),
                social_y=np.asarray(data["social_y"], dtype=np.int64),
                social_mask=np.asarray(data["social_mask"], dtype=np.float32),
                sample_rate_hz=np.asarray(data["sample_rate_hz"], dtype=np.float32),
                domain=np.asarray(data["domain"]),
                role=np.asarray(data["role"]),
                feature_set=np.asarray(data["feature_set"]),
            )
        output_rows.append({
            **row,
            "transform_method": "raw",
            "source_tensor_relative_path": row["tensor_relative_path"],
            "tensor_relative_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "normalizer_path": str(normalizer_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        })
    write_csv(output_manifest, list(output_rows[0].keys()), output_rows)
    transform_dir.mkdir(parents=True, exist_ok=True)
    with (transform_dir / "transform_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"feature_set": feature_set, "train_split": "train_internal", "input_manifest": str(args.input_manifest)}, handle, indent=2)
    print(f"Feature set: {feature_set}; transformed tensors: {len(output_rows)}")
    print(f"Manifest: {output_manifest}")


if __name__ == "__main__":
    main()
