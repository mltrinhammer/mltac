"""Combine NoXi and MPIIGI group-window manifests for joint DAPA training.

Both input manifests must be in group-window format (as produced by
``build_mpii_group_window_manifest.py``).  The script validates that the
column schemas and modality orders match, then concatenates the rows into a
single output manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.io import read_csv, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine group-window manifests for joint NoXi + MPIIGI training.",
    )
    parser.add_argument(
        "--input-manifests",
        nargs="+",
        type=Path,
        required=True,
        help="Two or more group-window manifests to combine.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Path for the combined output manifest.",
    )
    parser.add_argument(
        "--require-same-combo",
        action="store_true",
        default=False,
        help="If set, all inputs must share the same combo_name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_rows: list[dict[str, str]] = []
    reference_columns: list[str] | None = None
    reference_modality_order: list[str] | None = None
    combo_names: set[str] = set()

    for manifest_path in args.input_manifests:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        rows = read_csv(manifest_path)
        if not rows:
            print(f"Warning: {manifest_path} has no rows, skipping.")
            continue

        columns = list(rows[0].keys())
        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            raise RuntimeError(
                f"Column mismatch between manifests.\n"
                f"  Expected: {reference_columns}\n"
                f"  Got:      {columns}\n"
                f"  From:     {manifest_path}"
            )

        # Validate modality order consistency
        for row in rows:
            modality_order = json.loads(row["modality_order_json"])
            if reference_modality_order is None:
                reference_modality_order = modality_order
            elif modality_order != reference_modality_order:
                raise RuntimeError(
                    f"Modality order mismatch in {manifest_path}: "
                    f"{modality_order} vs {reference_modality_order}"
                )
            combo_names.add(row["combo_name"])

        all_rows.extend(rows)
        print(f"  {manifest_path.name}: {len(rows)} rows")

    if not all_rows:
        raise RuntimeError("No rows found in any input manifest.")

    if args.require_same_combo and len(combo_names) > 1:
        raise RuntimeError(f"Multiple combo_names found: {sorted(combo_names)}")

    # Count per-dataset stats
    dataset_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for row in all_rows:
        dataset_counts[row["dataset"]] = dataset_counts.get(row["dataset"], 0) + 1
        split_counts[row["model_split"]] = split_counts.get(row["model_split"], 0) + 1

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_manifest, reference_columns, all_rows)

    print(f"\nCombined manifest: {args.output_manifest}")
    print(f"Total rows: {len(all_rows)}")
    print(f"Datasets: {', '.join(f'{k}={v}' for k, v in sorted(dataset_counts.items()))}")
    print(f"Splits: {', '.join(f'{k}={v}' for k, v in sorted(split_counts.items()))}")
    print(f"Modalities: {', '.join(reference_modality_order or [])}")


if __name__ == "__main__":
    main()
