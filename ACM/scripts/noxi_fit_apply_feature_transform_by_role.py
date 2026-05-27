from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Role-specific transform branch.
#
# This is parallel to noxi_fit_apply_feature_transform.py, but it fits one
# normalizer/reducer per role. For dyadic modelling, this lets novice and expert
# streams be compressed in role-specific feature spaces before they are fused
# into [novice_features_t, expert_features_t].
from src.acm_pipeline.io import read_csv, write_csv
from src.acm_pipeline.transforms import (
    FeatureNormalizer,
    fit_pca,
    fit_random_projection,
    sample_normalized_frames,
    save_pickle,
)


ROLES = ("novice", "expert")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for role-wise normalization/reduction."""

    parser = argparse.ArgumentParser(description="Fit role-specific normalization/reduction and export transformed tensors.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--method", choices=["raw", "pca", "random_projection"], required=True)
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--train-split", default="train_internal")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--transform-dir", type=Path, default=None)
    parser.add_argument("--max-fit-frames", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=13)
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
    """Create a stable suffix for role-specific output folders/manifests."""

    if args.method == "raw":
        return "raw_by_role"
    if args.n_components is None:
        raise ValueError(f"--n-components is required for method {args.method!r}")
    short = "rp" if args.method == "random_projection" else args.method
    return f"{short}{args.n_components}_by_role"


def default_out_root(feature_set: str, suffix: str) -> Path:
    return PROJECT_ROOT / "processed" / "transformed" / f"{feature_set}_{suffix}"


def default_manifest_path(feature_set: str, suffix: str) -> Path:
    return PROJECT_ROOT / "outputs" / "manifests" / f"model_processed_manifest_{feature_set}_{suffix}.csv"


def export_variance(path: Path, reducer: object) -> None:
    """Write PCA explained-variance diagnostics when available."""

    ratios = getattr(reducer, "explained_variance_ratio_", None)
    if ratios is None:
        return
    rows = []
    cumulative = 0.0
    for idx, ratio in enumerate(ratios, start=1):
        cumulative += float(ratio)
        rows.append(
            {
                "component_idx": idx,
                "explained_variance_ratio": float(ratio),
                "cumulative_explained_variance": cumulative,
            }
        )
    write_csv(path, ["component_idx", "explained_variance_ratio", "cumulative_explained_variance"], rows)


def fit_role_transform(
    role: str,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    transform_dir: Path,
) -> tuple[FeatureNormalizer, object | None, str, str, str]:
    """Fit normalizer and optional reducer for one role only."""

    # This is the key distinction from the shared transform script: the fit set
    # is restricted to one role's training examples. Novice and expert therefore
    # get separate scaling and separate PCA/RP objects.
    train_paths = [
        tensor_path(row)
        for row in rows
        if row["model_split"] == args.train_split and row["role"] == role
    ]
    if not train_paths:
        raise RuntimeError(f"No training rows found for role={role!r} and split={args.train_split!r}.")

    role_dir = transform_dir / role
    normalizer = FeatureNormalizer.fit_npz_paths(train_paths)
    normalizer_path = role_dir / "normalizer.npz"
    normalizer.save(normalizer_path)

    reducer = None
    reducer_path = ""
    variance_path = ""
    if args.method in {"pca", "random_projection"}:
        frames = sample_normalized_frames(
            train_paths,
            normalizer,
            max_frames=args.max_fit_frames,
            seed=args.seed,
        )
        if args.n_components is None or args.n_components <= 0:
            raise ValueError("--n-components must be positive for dimensionality reduction.")
        if args.n_components > frames.shape[1]:
            raise ValueError(f"n_components={args.n_components} exceeds input dimension {frames.shape[1]} for role {role}.")

        if args.method == "pca":
            reducer = fit_pca(frames, n_components=args.n_components, seed=args.seed)
            variance_csv = role_dir / "pca_explained_variance.csv"
            export_variance(variance_csv, reducer)
            variance_path = str(variance_csv.relative_to(PROJECT_ROOT)).replace("\\", "/")
        else:
            reducer = fit_random_projection(frames, n_components=args.n_components, seed=args.seed)
        reducer_file = role_dir / f"{args.method}.pkl"
        save_pickle(reducer, reducer_file)
        reducer_path = str(reducer_file.relative_to(PROJECT_ROOT)).replace("\\", "/")

    return (
        normalizer,
        reducer,
        str(normalizer_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        reducer_path,
        variance_path,
    )


def transform_matrix(x: np.ndarray, normalizer: FeatureNormalizer, reducer: object | None) -> np.ndarray:
    """Normalize one sequence and optionally apply the role's fitted reducer."""

    x_norm = normalizer.transform(x)
    if reducer is None:
        return x_norm.astype(np.float32, copy=False)
    return reducer.transform(x_norm).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_manifest)
    if not rows:
        raise RuntimeError(f"No rows in manifest: {args.input_manifest}")

    feature_set = infer_feature_set(rows)
    suffix = method_suffix(args)
    out_root = args.out_root or default_out_root(feature_set, suffix)
    output_manifest = args.output_manifest or default_manifest_path(feature_set, suffix)
    transform_dir = args.transform_dir or PROJECT_ROOT / "outputs" / "transforms" / f"{feature_set}_{suffix}"

    # Fit one transform bundle per role. Stored paths are carried into every
    # output manifest row so dyadic builders can trace provenance later.
    transforms: dict[str, tuple[FeatureNormalizer, object | None, str, str]] = {}
    for role in ROLES:
        transforms[role] = fit_role_transform(role, rows, args, transform_dir)

    processed_rows: list[dict[str, object]] = []
    for row in rows:
        role = row["role"]
        if role not in transforms:
            continue
        normalizer, reducer, normalizer_path, reducer_path, variance_path = transforms[role]
        in_path = tensor_path(row)
        with np.load(in_path, allow_pickle=True) as data:
            x = np.asarray(data["x"], dtype=np.float32)
            x_out = transform_matrix(x, normalizer, reducer)
            out_dir = out_root / row["dataset"] / row["session_id"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{row['role']}.{feature_set}.{suffix}.npz"
            np.savez_compressed(
                out_path,
                x=x_out,
                y=np.asarray(data["y"], dtype=np.float32),
                target_mask=np.asarray(data["target_mask"], dtype=np.float32),
                source_tensor_relative_path=np.asarray([row["tensor_relative_path"]]),
                feature_set=np.asarray([feature_set]),
                transform_method=np.asarray([args.method]),
                transform_scope=np.asarray(["role_specific"]),
                role=np.asarray([role]),
                normalizer_path=np.asarray([normalizer_path]),
                reducer_path=np.asarray([reducer_path]),
                variance_path=np.asarray([variance_path]),
                sample_rate_hz=np.asarray(data["sample_rate_hz"], dtype=np.float32),
            )

        processed_rows.append(
            {
                **row,
                "transform_method": args.method,
                "transform_scope": "role_specific",
                "transform_suffix": suffix,
                "tensor_relative_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "source_tensor_relative_path": row["tensor_relative_path"],
                "n_features": str(x_out.shape[1]),
                "normalizer_path": normalizer_path,
                "reducer_path": reducer_path,
                "variance_path": variance_path,
            }
        )

    if not processed_rows:
        raise RuntimeError("No role-specific transformed rows were produced.")

    write_csv(output_manifest, list(processed_rows[0].keys()), processed_rows)
    config = {
        "input_manifest": str(args.input_manifest),
        "output_manifest": str(output_manifest),
        "feature_set": feature_set,
        "method": args.method,
        "transform_scope": "role_specific",
        "n_components": args.n_components,
        "train_split": args.train_split,
        "max_fit_frames": args.max_fit_frames,
        "seed": args.seed,
        "roles": list(ROLES),
    }
    transform_dir.mkdir(parents=True, exist_ok=True)
    with (transform_dir / "transform_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"Input manifest: {args.input_manifest}")
    print(f"Method: {args.method}")
    print("Transform scope: role_specific")
    print(f"Wrote transformed rows: {len(processed_rows)}")
    print(f"Output manifest: {output_manifest}")
    print(f"Transform dir: {transform_dir}")


if __name__ == "__main__":
    main()
