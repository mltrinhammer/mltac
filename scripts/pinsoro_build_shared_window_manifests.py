"""Build one canonical PinSoRo window grid shared by every modality."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.io import read_csv, write_csv
from src.acm_pipeline.pinsoro import ROLE_ORDER
from src.acm_pipeline.turns import compute_window_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shared PinSoRo windows across modality manifests.")
    parser.add_argument("--input-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--window-size", type=int, default=250)
    parser.add_argument("--stride", type=int, default=62)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "pinsoro" / "windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_feature: dict[str, dict[tuple[str, str], dict[str, dict[str, str]]]] = {}
    for manifest in args.input_manifests:
        rows = read_csv(manifest)
        feature_sets = sorted({row["feature_set"] for row in rows})
        if len(feature_sets) != 1:
            raise RuntimeError(f"Expected one feature set in {manifest}, got {feature_sets}")
        grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
        for row in rows:
            grouped[(row["source_split"], row["session_id"])][row["role"]] = row
        by_feature[feature_sets[0]] = dict(grouped)

    shared_sessions = set.intersection(*(set(grouped) for grouped in by_feature.values()))
    canonical: list[dict[str, object]] = []
    individual_by_feature: dict[str, list[dict[str, object]]] = {name: [] for name in by_feature}
    dyadic_by_feature: dict[str, list[dict[str, object]]] = {name: [] for name in by_feature}
    for source_split, session_id in sorted(shared_sessions):
        if any(any(role not in grouped[(source_split, session_id)] for role in ROLE_ORDER) for grouped in by_feature.values()):
            continue
        all_rows = [
            by_feature[feature_set][(source_split, session_id)][role]
            for feature_set in sorted(by_feature)
            for role in ROLE_ORDER
        ]
        shared_len = min(int(row["aligned_len"]) for row in all_rows)
        base_row = all_rows[0]
        for window_idx, segment in enumerate(compute_window_segments(shared_len, args.window_size, args.stride)):
            common = {
                "dataset": "pinsoro",
                "domain": base_row["domain"],
                "source_split": source_split,
                "model_split": base_row["model_split"],
                "session_id": session_id,
                "window_idx": window_idx,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "window_len": segment.end_frame - segment.start_frame,
                "session_aligned_len": shared_len,
            }
            canonical.append(common)
            for feature_set, grouped in by_feature.items():
                purple = grouped[(source_split, session_id)]["purple"]
                yellow = grouped[(source_split, session_id)]["yellow"]
                for role, row in (("purple", purple), ("yellow", yellow)):
                    if row["domain"] == "CR" and role == "yellow":
                        continue
                    individual_by_feature[feature_set].append({
                        **common, "feature_set": feature_set, "role": role, "supervised": row["supervised"],
                        "tensor_relative_path": row["tensor_relative_path"], "n_features": row["n_features"],
                    })
                dyadic_by_feature[feature_set].append({
                    **common, "feature_set": feature_set,
                    "purple_supervised": purple["supervised"], "yellow_supervised": yellow["supervised"],
                    "purple_tensor_relative_path": purple["tensor_relative_path"],
                    "yellow_tensor_relative_path": yellow["tensor_relative_path"],
                    "n_features_per_role": purple["n_features"],
                })

    if not canonical:
        raise RuntimeError("No windows shared across all requested modalities.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"w{args.window_size}_s{args.stride}"
    write_csv(args.out_dir / f"shared_{suffix}_canonical.csv", list(canonical[0].keys()), canonical)
    for feature_set in sorted(by_feature):
        individual = individual_by_feature[feature_set]
        dyadic = dyadic_by_feature[feature_set]
        write_csv(args.out_dir / f"{feature_set}_{suffix}_individual.csv", list(individual[0].keys()), individual)
        write_csv(args.out_dir / f"{feature_set}_{suffix}_dyadic.csv", list(dyadic[0].keys()), dyadic)
    print(f"Modalities: {len(by_feature)}; shared sessions: {len(shared_sessions)}; canonical windows: {len(canonical)}")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
