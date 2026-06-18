"""Build aligned PinSoRo session-role tensors for one feature set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.alignment import TARGET_RATE_HZ, align_to_target_grid
from src.acm_pipeline.feature_registry import FEATURE_SETS, get_feature_set
from src.acm_pipeline.io import read_csv, read_stream_matrix, write_csv
from src.acm_pipeline.pinsoro import cache_path, read_class_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PinSoRo feature tensors aligned to 25 Hz.")
    parser.add_argument("--feature-set", required=True, choices=sorted(FEATURE_SETS))
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "cache" / "pinsoro")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "outputs" / "pinsoro" / "raw_manifest.csv")
    parser.add_argument("--streams", type=Path, default=PROJECT_ROOT / "outputs" / "pinsoro" / "raw_stream_manifest.csv")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--status-out", type=Path, default=None)
    return parser.parse_args()


def stream_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["source_split"], row["session_id"], row["role"]


def load_targets(row: dict[str, str], cache_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    task_rel, social_rel = row.get("task_target_relative_path", ""), row.get("social_target_relative_path", "")
    if not task_rel or not social_rel:
        return None
    task_y, task_mask = read_class_labels(cache_path(cache_root, task_rel), "task_engagement")
    social_y, social_mask = read_class_labels(cache_path(cache_root, social_rel), "social_engagement")
    length = min(len(task_y), len(social_y))
    return task_y[:length], task_mask[:length], social_y[:length], social_mask[:length]


def process_row(
    row: dict[str, str],
    lookup: dict[tuple[str, str, str], dict[str, dict[str, str]]],
    required_streams: tuple[str, ...],
    feature_set: str,
    cache_root: Path,
    out_root: Path,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    key = (row["source_split"], row["session_id"], row["role"])
    status: dict[str, object] = {**{k: row[k] for k in ("domain", "source_split", "model_split", "session_id", "role")}, "feature_set": feature_set}
    stream_specs = lookup.get(key, {})
    matrices: list[tuple[str, np.ndarray, float, int]] = []
    for stream_name in required_streams:
        spec = stream_specs.get(stream_name)
        if spec is None:
            status.update({"status": "skipped", "reason": f"missing_stream:{stream_name}"})
            return None, status
        header = cache_path(cache_root, spec["local_relative_path"])
        binary = cache_path(cache_root, spec["binary_local_relative_path"])
        if not header.exists() or not binary.exists():
            status.update({"status": "skipped", "reason": f"missing_stream_files:{stream_name}"})
            return None, status
        matrix, rate, dim = read_stream_matrix(header, binary)
        matrices.append((stream_name, matrix, rate, dim))

    targets = load_targets(row, cache_root)
    if targets is None:
        durations = [len(matrix) / rate for _, matrix, rate, _ in matrices if rate > 0]
        reference_len = round(min(durations) * TARGET_RATE_HZ)
        task_y = np.full(reference_len, -1, dtype=np.int64)
        social_y = np.full(reference_len, -1, dtype=np.int64)
        task_mask = np.zeros(reference_len, dtype=np.float32)
        social_mask = np.zeros(reference_len, dtype=np.float32)
    else:
        task_y, task_mask, social_y, social_mask = targets
        reference_len = len(task_y)

    if row["supervised"] != "yes":
        task_mask[:] = 0.0
        social_mask[:] = 0.0

    aligned_parts: list[np.ndarray] = []
    methods: list[str] = []
    dims: list[str] = []
    for stream_name, matrix, rate, dim in matrices:
        aligned, method = align_to_target_grid(matrix, rate, reference_len)
        if len(aligned) == 0:
            status.update({"status": "skipped", "reason": f"empty_after_alignment:{stream_name}"})
            return None, status
        aligned_parts.append(aligned)
        methods.append(f"{stream_name}:{method}")
        dims.append(str(dim))

    aligned_len = min(reference_len, *(len(part) for part in aligned_parts))
    x = np.concatenate([part[:aligned_len] for part in aligned_parts], axis=1).astype(np.float32, copy=False)
    out_dir = out_root / row["source_split"] / row["session_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{row['role']}.{feature_set}.25hz.npz"
    np.savez_compressed(
        out_path,
        x=x,
        task_y=task_y[:aligned_len],
        task_mask=task_mask[:aligned_len],
        social_y=social_y[:aligned_len],
        social_mask=social_mask[:aligned_len],
        sample_rate_hz=np.asarray([TARGET_RATE_HZ], dtype=np.float32),
        domain=np.asarray([row["domain"]]),
        role=np.asarray([row["role"]]),
        feature_set=np.asarray([feature_set]),
    )
    processed = {
        **row,
        "feature_set": feature_set,
        "tensor_relative_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "included_streams": ";".join(required_streams),
        "stream_dims": ";".join(dims),
        "alignment_methods": ";".join(methods),
        "sample_rate_hz": f"{TARGET_RATE_HZ:.3f}",
        "n_features": str(x.shape[1]),
        "aligned_len": str(aligned_len),
        "task_valid_count": str(int(task_mask[:aligned_len].sum())),
        "social_valid_count": str(int(social_mask[:aligned_len].sum())),
    }
    status.update({"status": "ok", "reason": "", "aligned_len": aligned_len, "n_features": x.shape[1]})
    return processed, status


def main() -> None:
    args = parse_args()
    definition = get_feature_set(args.feature_set)
    out_root = args.out_root or PROJECT_ROOT / "processed" / "pinsoro" / f"{args.feature_set}_25hz"
    output_manifest = args.output_manifest or PROJECT_ROOT / "outputs" / "pinsoro" / "manifests" / f"{args.feature_set}_25hz.csv"
    status_out = args.status_out or PROJECT_ROOT / "outputs" / "pinsoro" / "manifests" / f"{args.feature_set}_25hz_status.csv"
    lookup: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for row in read_csv(args.streams):
        lookup.setdefault(stream_key(row), {})[row["stream_name"]] = row
    processed_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for row in read_csv(args.manifest):
        processed, status = process_row(row, lookup, definition.streams, args.feature_set, args.cache_root, out_root)
        status_rows.append(status)
        if processed is not None:
            processed_rows.append(processed)
    if not processed_rows:
        raise RuntimeError(f"No tensors produced for {args.feature_set}.")
    write_csv(output_manifest, list(processed_rows[0].keys()), processed_rows)
    write_csv(status_out, list(status_rows[0].keys()), status_rows)
    print(f"Feature set: {args.feature_set}; tensors: {len(processed_rows)}")
    print(f"Manifest: {output_manifest}")


if __name__ == "__main__":
    main()
