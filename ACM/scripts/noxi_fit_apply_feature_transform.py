from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# This script is the second preprocessing stage. It takes aligned 25 Hz tensors
# and creates the active model-input branch: normalized raw features.

from src.acm_pipeline.io import read_csv, write_csv
from src.acm_pipeline.transforms import FeatureNormalizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for train-only normalization export."""

    parser = argparse.ArgumentParser(description="Fit train-only normalization and export transformed tensors.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--method", choices=["raw"], default="raw")
    parser.add_argument("--train-split", default="train_internal")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--transform-dir", type=Path, default=None)
    return parser.parse_args()


def tensor_path(row: dict[str, str]) -> Path:
    """Resolve tensor paths from a processed manifest row."""

    path = Path(row["tensor_relative_path"])
    return path if path.is_absolute() else PROJECT_ROOT / path


def infer_feature_set(rows: list[dict[str, str]]) -> str:
    """Infer the feature-set name from a processed manifest."""

    values = sorted({row.get("feature_set", "") for row in rows if row.get("feature_set", "")})
    if len(values) == 1:
        return values[0]
    return "features"


def method_suffix(args: argparse.Namespace) -> str:
    """Create a stable suffix for output folders and manifests."""

    return "raw"


def default_out_root(feature_set: str, suffix: str) -> Path:
    return PROJECT_ROOT / "processed" / "transformed" / f"{feature_set}_{suffix}"


def default_manifest_path(feature_set: str, suffix: str) -> Path:
    return PROJECT_ROOT / "outputs" / "manifests" / f"model_processed_manifest_{feature_set}_{suffix}.csv"


def transform_matrix(x: np.ndarray, normalizer: FeatureNormalizer) -> np.ndarray:
    """Normalize one sequence with the train-fitted normalizer."""

    return normalizer.transform(x).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_manifest)
    if not rows:
        raise RuntimeError(f"No rows in manifest: {args.input_manifest}")
    # The output names inherit the feature-set and transform method so many
    # branches can coexist without overwriting each other.
    feature_set = infer_feature_set(rows)
    suffix = method_suffix(args)
    out_root = args.out_root or default_out_root(feature_set, suffix)
    output_manifest = args.output_manifest or default_manifest_path(feature_set, suffix)
    transform_dir = args.transform_dir or PROJECT_ROOT / "outputs" / "transforms" / f"{feature_set}_{suffix}"

    # Fit statistics only on the training split. Validation/test tensors are
    # transformed with the same saved objects to avoid leakage.
    train_paths = [tensor_path(row) for row in rows if row["model_split"] == args.train_split]
    if not train_paths:
        raise RuntimeError(f"No train rows found for split {args.train_split!r}.")

    # All transforms are fit on the training split only, then applied unchanged
    # to every row in the input manifest.
    # The normalizer is always fitted, including for "raw". Raw means no
    # dimensionality reduction, not unnormalized model input.
    normalizer = FeatureNormalizer.fit_npz_paths(train_paths)
    normalizer_path = transform_dir / "normalizer.npz"
    normalizer.save(normalizer_path)

    reducer_path = ""
    variance_path = ""

    processed_rows: list[dict[str, object]] = []
    for row in rows:
        # Export transformed session-role tensors with the same y/mask metadata
        # so downstream model scripts can consume all transform branches equally.
        in_path = tensor_path(row)
        with np.load(in_path, allow_pickle=True) as data:
            x = np.asarray(data["x"], dtype=np.float32)
            x_out = transform_matrix(x, normalizer)
            out_dir = out_root / row["dataset"] / row["session_id"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{row['role']}.{feature_set}.{suffix}.npz"
            # Preserve y/target_mask unchanged so every transformed branch can
            # be consumed by the same downstream training code.
            np.savez_compressed(
                out_path,
                x=x_out,
                y=np.asarray(data["y"], dtype=np.float32),
                target_mask=np.asarray(data["target_mask"], dtype=np.float32),
                source_tensor_relative_path=np.asarray([row["tensor_relative_path"]]),
                feature_set=np.asarray([feature_set]),
                transform_method=np.asarray([args.method]),
                normalizer_path=np.asarray([str(normalizer_path.relative_to(PROJECT_ROOT)).replace("\\", "/")]),
                reducer_path=np.asarray([""]),
                sample_rate_hz=np.asarray(data["sample_rate_hz"], dtype=np.float32),
            )
        processed_rows.append(
            {
                **row,
                "transform_method": args.method,
                "transform_suffix": suffix,
                "tensor_relative_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "source_tensor_relative_path": row["tensor_relative_path"],
                "n_features": str(x_out.shape[1]),
                "normalizer_path": str(normalizer_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "reducer_path": reducer_path,
                "variance_path": variance_path,
            }
        )

    base_fields = list(processed_rows[0].keys())
    write_csv(output_manifest, base_fields, processed_rows)
    # Persist transform configuration next to the fitted objects. This is the
    # minimum metadata needed to reproduce a transform branch on UCloud.
    config = {
        "input_manifest": str(args.input_manifest),
        "output_manifest": str(output_manifest),
        "feature_set": feature_set,
        "method": "raw",
        "train_split": args.train_split,
        "normalizer_path": str(normalizer_path),
        "reducer_path": "",
    }
    transform_dir.mkdir(parents=True, exist_ok=True)
    with (transform_dir / "transform_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"Input manifest: {args.input_manifest}")
    print("Method: raw")
    print(f"Wrote transformed rows: {len(processed_rows)}")
    print(f"Output manifest: {output_manifest}")
    print(f"Transform dir: {transform_dir}")


if __name__ == "__main__":
    main()
