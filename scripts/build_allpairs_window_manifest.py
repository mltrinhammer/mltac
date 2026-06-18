"""Build paired fixed-window manifests for all C(N,2) participant pairs per session.

Like ``build_allpairs_turn_manifest.py`` but uses geometric sliding windows
instead of speech-turn boundaries.  No transcript files are needed — the
segmentation is purely geometric.

Composite session ID format::

    {session_id}__pair__{role_a}__{role_b}

Usage::

    python scripts/build_allpairs_window_manifest.py \\
        --input-manifest outputs/mpiii_eval/manifests/model_processed_manifest_audio_w2vbert2_raw.csv \\
        --output-manifest outputs/mpiii_eval/manifests/model_processed_manifest_audio_w2vbert2_raw_turns.csv \\
        --window-size 500 --stride 125
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.io import read_csv, write_csv
from src.acm_pipeline.turns import compute_window_segments


PAIR_SEPARATOR = "__pair__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all-pairs fixed-window manifests for N-participant sessions."
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--stride", type=int, default=125)
    parser.add_argument("--pair-separator", default=PAIR_SEPARATOR)
    parser.add_argument("--min-window-frames", type=int, default=1)
    return parser.parse_args()


def infer_branch_name(rows: list[dict[str, str]]) -> str:
    feature_sets = sorted({row.get("feature_set", "features") for row in rows})
    suffixes = sorted({row.get("transform_suffix", row.get("transform_method", "raw")) for row in rows})
    feature_set = feature_sets[0] if len(feature_sets) == 1 else "features"
    suffix = suffixes[0] if len(suffixes) == 1 else "mixed"
    return f"{feature_set}_{suffix}_turns"


def default_manifest_path(branch_name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "manifests" / f"model_processed_manifest_{branch_name}.csv"


def group_rows_by_session(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    """Group manifest rows by (dataset, session_id) -> {role: row}."""
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        role = row.get("role", "")
        if not role:
            continue
        grouped[(row["dataset"], row["session_id"])][role] = row
    return dict(grouped)


def build_pair_window_rows(
    dataset: str,
    session_id: str,
    role_a: str,
    role_b: str,
    row_a: dict[str, str],
    row_b: dict[str, str],
    window_size: int,
    stride: int,
    pair_separator: str,
    min_window_frames: int,
) -> list[dict[str, object]]:
    """Build window rows for one (role_a, role_b) pair.

    role_a is placed in the "novice" model slot, role_b in the "expert" slot.
    """
    composite_id = f"{session_id}{pair_separator}{role_a}{pair_separator}{role_b}"
    session_len = min(int(row_a["aligned_len"]), int(row_b["aligned_len"]))
    segments = compute_window_segments(session_len, window_size=window_size, stride=stride)

    rows: list[dict[str, object]] = []
    for window_idx, segment in enumerate(segments):
        window_len = segment.end_frame - segment.start_frame
        if window_len < min_window_frames:
            continue
        rows.append(
            {
                "dataset": dataset,
                "session_id": composite_id,
                "model_split": row_a["model_split"],
                "feature_set": row_a.get("feature_set", ""),
                "transform_method": row_a.get("transform_method", ""),
                "transform_scope": row_a.get("transform_scope", "shared"),
                "transform_suffix": row_a.get("transform_suffix", row_a.get("transform_method", "raw")),
                "turn_idx": window_idx,
                "speaker": segment.speaker,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "turn_len": window_len,
                "session_aligned_len": session_len,
                "novice_tensor_relative_path": row_a["tensor_relative_path"],
                "expert_tensor_relative_path": row_b["tensor_relative_path"],
                "novice_aligned_len": row_a["aligned_len"],
                "expert_aligned_len": row_b["aligned_len"],
                "n_features_per_role": row_a["n_features"],
                "metadata_json": json.dumps(
                    {
                        "novice_role": role_a,
                        "expert_role": role_b,
                        "real_session_id": session_id,
                    },
                    sort_keys=True,
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_manifest)
    if not rows:
        raise RuntimeError(f"No rows in manifest: {args.input_manifest}")

    branch_name = infer_branch_name(rows)
    output_manifest = args.output_manifest or default_manifest_path(branch_name)

    session_groups = group_rows_by_session(rows)
    processed_rows: list[dict[str, object]] = []
    skipped_pairs = 0

    for (dataset, session_id), role_rows in sorted(session_groups.items()):
        roles = sorted(role_rows.keys())
        if len(roles) < 2:
            print(f"  skip {dataset}/{session_id}: only {len(roles)} role(s)")
            continue

        for role_a, role_b in combinations(roles, 2):
            pair_windows = build_pair_window_rows(
                dataset=dataset,
                session_id=session_id,
                role_a=role_a,
                role_b=role_b,
                row_a=role_rows[role_a],
                row_b=role_rows[role_b],
                window_size=args.window_size,
                stride=args.stride,
                pair_separator=args.pair_separator,
                min_window_frames=args.min_window_frames,
            )
            if not pair_windows:
                skipped_pairs += 1
                continue
            processed_rows.extend(pair_windows)

    if not processed_rows:
        raise RuntimeError("No window rows were produced from any pair.")

    write_csv(output_manifest, list(processed_rows[0].keys()), processed_rows)

    n_sessions = len(session_groups)
    n_pairs = sum(1 for roles in session_groups.values() for _ in combinations(sorted(roles.keys()), 2))
    print(f"Input manifest: {args.input_manifest}")
    print(f"Sessions: {n_sessions}  Total pairs: {n_pairs}  Skipped pairs: {skipped_pairs}")
    print(f"Window size: {args.window_size}  Stride: {args.stride}")
    print(f"Wrote window rows: {len(processed_rows)}")
    print(f"Output manifest: {output_manifest}")


if __name__ == "__main__":
    main()
