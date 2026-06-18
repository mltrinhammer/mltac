"""Build MPII group-window multimodal manifests from role-level manifests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.io import read_csv, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build group-window multimodal manifests for MPII.")
    parser.add_argument("--input-manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--combo-name", default="")
    parser.add_argument("--window-frames", type=int, default=500)
    parser.add_argument("--stride-frames", type=int, default=125)
    parser.add_argument("--min-window-frames", type=int, default=5)
    parser.add_argument(
        "--val-session-ids",
        nargs="*",
        default=[],
        help="Session IDs to relabel as val_internal for leave-one-session-out folds.",
    )
    parser.add_argument(
        "--test-session-ids",
        nargs="*",
        default=[],
        help="Session IDs to relabel as test_internal.",
    )
    return parser.parse_args()


def _feature_set(rows: list[dict[str, str]], path: Path) -> str:
    values = sorted({row.get("feature_set", "") for row in rows if row.get("feature_set", "")})
    if len(values) != 1:
        raise RuntimeError(f"Expected one feature_set in {path}, got {values}")
    return values[0]


def _group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["dataset"], row["session_id"])][row["role"]] = row
    return dict(grouped)


def _window_ranges(session_len: int, window_frames: int, stride_frames: int, min_window_frames: int) -> list[tuple[int, int]]:
    if session_len < min_window_frames:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < session_len:
        end = min(start + window_frames, session_len)
        if end - start >= min_window_frames:
            ranges.append((start, end))
        if end >= session_len:
            break
        start += stride_frames
    return ranges


def main() -> None:
    args = parse_args()
    if args.window_frames <= 0 or args.stride_frames <= 0:
        raise ValueError("--window-frames and --stride-frames must be positive.")

    rows_by_modality: dict[str, list[dict[str, str]]] = {}
    grouped_by_modality: dict[str, dict[tuple[str, str], dict[str, dict[str, str]]]] = {}
    for manifest in args.input_manifests:
        rows = read_csv(manifest)
        feature_set = _feature_set(rows, manifest)
        rows_by_modality[feature_set] = rows
        grouped_by_modality[feature_set] = _group_rows(rows)

    modality_order = tuple(rows_by_modality.keys())
    combo_name = args.combo_name.strip() or "__".join(modality_order)
    common_sessions = set.intersection(*(set(groups.keys()) for groups in grouped_by_modality.values()))
    output_rows: list[dict[str, object]] = []
    val_session_ids = set(args.val_session_ids or [])
    test_session_ids = set(args.test_session_ids or [])
    if val_session_ids & test_session_ids:
        raise RuntimeError(f"Sessions cannot be both validation and test: {sorted(val_session_ids & test_session_ids)}")

    for dataset, session_id in sorted(common_sessions):
        role_sets = [set(grouped_by_modality[modality][(dataset, session_id)].keys()) for modality in modality_order]
        roles = tuple(sorted(set.intersection(*role_sets)))
        if len(roles) < 2:
            continue

        split_values = {
            grouped_by_modality[modality][(dataset, session_id)][role]["model_split"]
            for modality in modality_order
            for role in roles
        }
        if len(split_values) != 1:
            raise RuntimeError(f"Mixed splits for {dataset}/{session_id}: {sorted(split_values)}")
        model_split = next(iter(split_values))
        if session_id in val_session_ids:
            model_split = "val_internal"
        elif session_id in test_session_ids:
            model_split = "test_internal"
        elif val_session_ids or test_session_ids:
            model_split = "train_internal"

        session_len = min(
            int(grouped_by_modality[modality][(dataset, session_id)][role]["aligned_len"])
            for modality in modality_order
            for role in roles
        )
        modality_specs: dict[str, object] = {}
        for modality in modality_order:
            role_specs: dict[str, object] = {}
            for role in roles:
                row = grouped_by_modality[modality][(dataset, session_id)][role]
                role_specs[role] = {
                    "feature_set": row.get("feature_set", modality),
                    "transform_method": row.get("transform_method", ""),
                    "transform_suffix": row.get("transform_suffix", row.get("transform_method", "")),
                    "tensor_relative_path": row["tensor_relative_path"],
                    "aligned_len": int(row["aligned_len"]),
                    "n_features": int(row["n_features"]),
                }
            modality_specs[modality] = {"roles": role_specs}

        for window_idx, (start, end) in enumerate(
            _window_ranges(session_len, args.window_frames, args.stride_frames, args.min_window_frames)
        ):
            output_rows.append(
                {
                    "dataset": dataset,
                    "session_id": session_id,
                    "model_split": model_split,
                    "combo_name": combo_name,
                    "window_idx": window_idx,
                    "start_frame": start,
                    "end_frame": end,
                    "window_len": end - start,
                    "session_aligned_len": session_len,
                    "role_order_json": json.dumps(list(roles)),
                    "modality_order_json": json.dumps(list(modality_order)),
                    "modalities_json": json.dumps(modality_specs, sort_keys=True),
                }
            )

    if not output_rows:
        raise RuntimeError("No group windows were produced.")
    write_csv(args.output_manifest, list(output_rows[0].keys()), output_rows)
    print(f"Modalities: {', '.join(modality_order)}")
    print(f"Sessions: {len(common_sessions)}")
    print(f"Windows: {len(output_rows)}")
    if val_session_ids:
        print(f"Validation sessions: {', '.join(sorted(val_session_ids))}")
    if test_session_ids:
        print(f"Test sessions: {', '.join(sorted(test_session_ids))}")
    print(f"Output manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
