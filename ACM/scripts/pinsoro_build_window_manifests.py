"""Build canonical, individual, and dyadic PinSoRo window manifests."""

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
    parser = argparse.ArgumentParser(description="Build PinSoRo fixed-window manifests.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=250)
    parser.add_argument("--stride", type=int, default=62)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "pinsoro" / "windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_manifest)
    if not rows:
        raise RuntimeError(f"No rows in {args.input_manifest}")
    feature_sets = sorted({row["feature_set"] for row in rows})
    if len(feature_sets) != 1:
        raise RuntimeError(f"Expected one feature set, got {feature_sets}")
    feature_set = feature_sets[0]
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["source_split"], row["session_id"])][row["role"]] = row

    canonical: list[dict[str, object]] = []
    individual: list[dict[str, object]] = []
    dyadic: list[dict[str, object]] = []
    for (source_split, session_id), role_rows in sorted(grouped.items()):
        if any(role not in role_rows for role in ROLE_ORDER):
            continue
        purple, yellow = role_rows["purple"], role_rows["yellow"]
        shared_len = min(int(purple["aligned_len"]), int(yellow["aligned_len"]))
        segments = compute_window_segments(shared_len, window_size=args.window_size, stride=args.stride)
        for window_idx, segment in enumerate(segments):
            common = {
                "dataset": "pinsoro",
                "domain": purple["domain"],
                "source_split": source_split,
                "model_split": purple["model_split"],
                "session_id": session_id,
                "feature_set": feature_set,
                "window_idx": window_idx,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "window_len": segment.end_frame - segment.start_frame,
                "session_aligned_len": shared_len,
            }
            canonical.append(common)
            for role, row in (("purple", purple), ("yellow", yellow)):
                if row["domain"] == "CR" and role == "yellow":
                    continue
                individual.append({
                    **common,
                    "role": role,
                    "supervised": row["supervised"],
                    "tensor_relative_path": row["tensor_relative_path"],
                    "n_features": row["n_features"],
                })
            dyadic.append({
                **common,
                "purple_supervised": purple["supervised"],
                "yellow_supervised": yellow["supervised"],
                "purple_tensor_relative_path": purple["tensor_relative_path"],
                "yellow_tensor_relative_path": yellow["tensor_relative_path"],
                "n_features_per_role": purple["n_features"],
            })

    if not canonical:
        raise RuntimeError("No paired windows were produced.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{feature_set}_w{args.window_size}_s{args.stride}"
    write_csv(args.out_dir / f"{prefix}_canonical.csv", list(canonical[0].keys()), canonical)
    write_csv(args.out_dir / f"{prefix}_individual.csv", list(individual[0].keys()), individual)
    write_csv(args.out_dir / f"{prefix}_dyadic.csv", list(dyadic[0].keys()), dyadic)
    print(f"Feature set: {feature_set}; canonical: {len(canonical)}; individual: {len(individual)}; dyadic: {len(dyadic)}")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
