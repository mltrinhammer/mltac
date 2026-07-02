"""Fit and apply PinSoRo feature normalization separately by domain."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.io import read_csv, write_csv
from src.acm_pipeline.transforms import FeatureNormalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize PinSoRo tensors with separate train-only CC and CR "
            "feature statistics."
        )
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--transform-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train_internal")
    parser.add_argument("--domains", nargs="+", default=("CC", "CR"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report fit/apply counts without writing tensors or manifests.",
    )
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def manifest_path(path: Path) -> str:
    try:
        path = path.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    return str(path).replace("\\", "/")


def copy_normalized_tensor(
    source_path: Path,
    out_path: Path,
    normalizer: FeatureNormalizer,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source_path, allow_pickle=True) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "x"}
        arrays.update(
            {
                "x": normalizer.transform(np.asarray(data["x"], dtype=np.float32)),
                "task_y": np.asarray(data["task_y"], dtype=np.int64),
                "task_mask": np.asarray(data["task_mask"], dtype=np.float32),
                "social_y": np.asarray(data["social_y"], dtype=np.int64),
                "social_mask": np.asarray(data["social_mask"], dtype=np.float32),
                "sample_rate_hz": np.asarray(data["sample_rate_hz"], dtype=np.float32),
            }
        )
        np.savez_compressed(out_path, **arrays)


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_manifest)
    if not rows:
        raise RuntimeError(f"No rows in {args.input_manifest}")
    feature_sets = sorted({row["feature_set"] for row in rows})
    if len(feature_sets) != 1:
        raise RuntimeError(f"Expected one feature set, got {feature_sets}")
    feature_set = feature_sets[0]
    domains = tuple(args.domains)
    unknown_domains = sorted({row["domain"] for row in rows} - set(domains))
    if unknown_domains:
        raise RuntimeError(f"Input manifest contains unexpected domains: {unknown_domains}")
    if args.output_manifest.exists() and not args.force and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite {args.output_manifest}; pass --force")

    train_paths_by_domain: dict[str, list[Path]] = {}
    for domain in domains:
        paths = [
            resolve(row["tensor_relative_path"])
            for row in rows
            if row["domain"] == domain and row["model_split"] == args.train_split
        ]
        if not paths:
            raise RuntimeError(
                f"No {args.train_split} tensor rows found for domain {domain}"
            )
        train_paths_by_domain[domain] = paths

    row_counts = Counter(row["domain"] for row in rows)
    train_counts = {domain: len(paths) for domain, paths in train_paths_by_domain.items()}
    print(
        f"feature_set={feature_set} rows={len(rows)} row_counts={dict(row_counts)} "
        f"train_fit_counts={train_counts}",
        flush=True,
    )
    if args.dry_run:
        print(f"would_write_manifest={args.output_manifest}", flush=True)
        print(f"would_write_tensors_under={args.out_root}", flush=True)
        print(f"would_write_transforms_under={args.transform_dir}", flush=True)
        return

    normalizers: dict[str, FeatureNormalizer] = {}
    normalizer_paths: dict[str, Path] = {}
    for domain, paths in train_paths_by_domain.items():
        normalizer = FeatureNormalizer.fit_npz_paths(paths)
        normalizer_path = args.transform_dir / domain / "normalizer.npz"
        normalizer.save(normalizer_path)
        normalizers[domain] = normalizer
        normalizer_paths[domain] = normalizer_path

    output_rows: list[dict[str, object]] = []
    for row in rows:
        domain = row["domain"]
        source_path = resolve(row["tensor_relative_path"])
        out_path = (
            args.out_root
            / row["source_split"]
            / row["session_id"]
            / f"{row['role']}.{feature_set}.raw.npz"
        )
        copy_normalized_tensor(source_path, out_path, normalizers[domain])
        output_rows.append(
            {
                **row,
                "transform_method": "domain_raw",
                "transform_domain": domain,
                "source_tensor_relative_path": row["tensor_relative_path"],
                "tensor_relative_path": manifest_path(out_path),
                "normalizer_path": manifest_path(normalizer_paths[domain]),
            }
        )

    write_csv(args.output_manifest, list(output_rows[0].keys()), output_rows)
    args.transform_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "feature_set": feature_set,
        "train_split": args.train_split,
        "domains": list(domains),
        "fit_scope": "per fold and per domain, using rows where model_split matches train_split",
        "input_manifest": str(args.input_manifest),
        "output_manifest": str(args.output_manifest),
        "out_root": str(args.out_root),
        "row_counts_by_domain": dict(row_counts),
        "train_fit_counts_by_domain": train_counts,
        "normalizers": {
            domain: manifest_path(path) for domain, path in normalizer_paths.items()
        },
    }
    with (args.transform_dir / "transform_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"transformed_tensors={len(output_rows)}", flush=True)
    print(f"manifest={args.output_manifest}", flush=True)


if __name__ == "__main__":
    main()
