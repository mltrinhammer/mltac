"""Build PinSoRo session-role tensors on the native 30 Hz label timeline."""

from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.acm_pipeline.feature_registry import FEATURE_SETS, get_feature_set
from src.acm_pipeline.io import read_csv, write_csv
from src.acm_pipeline.pinsoro import (
    LABEL_RATE_HZ,
    cache_path,
    pinsoro_stream_metadata,
    read_class_labels,
    read_pinsoro_stream,
)

TARGET_RATE_HZ = LABEL_RATE_HZ
REFERENCE_STREAMS = (
    "clip",
    "dino",
    "openface2",
    "openface3",
    "openpose",
    "swin",
    "videomae",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare PinSoRo feature tensors on the native 30 Hz label grid."
    )
    p.add_argument("--feature-set", required=True, choices=sorted(FEATURE_SETS))
    p.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "cache/pinsoro")
    p.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/pinsoro/raw_manifest.csv",
    )
    p.add_argument(
        "--streams",
        type=Path,
        default=PROJECT_ROOT / "outputs/pinsoro/raw_stream_manifest.csv",
    )
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--output-manifest", type=Path, default=None)
    p.add_argument("--status-out", type=Path, default=None)
    return p.parse_args()


def stream_key(r: dict[str, str]) -> tuple[str, str, str]:
    return r["source_split"], r["session_id"], r["role"]


def manifest_path(path: Path) -> str:
    try:
        path = path.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    return str(path).replace("\\", "/")


def load_targets(row: dict[str, str], cache_root: Path):
    tr, sr = (
        row.get("task_target_relative_path", ""),
        row.get("social_target_relative_path", ""),
    )
    if not tr or not sr:
        return None
    ty, tm = read_class_labels(cache_path(cache_root, tr), "task_engagement")
    sy, sm = read_class_labels(cache_path(cache_root, sr), "social_engagement")
    n = min(len(ty), len(sy))
    return ty[:n], tm[:n], sy[:n], sm[:n]


def test_target_len(
    specs: dict[str, dict[str, str]], cache_root: Path
) -> tuple[int, float]:
    durations = []
    for name in REFERENCE_STREAMS:
        spec = specs.get(name)
        if spec is None:
            continue
        rate, _, frames = pinsoro_stream_metadata(
            cache_path(cache_root, spec["local_relative_path"])
        )
        durations.append(frames / rate)
    if not durations:
        raise RuntimeError("Test role has no native-rate reference stream.")
    duration = min(durations)
    return int(math.floor(duration * TARGET_RATE_HZ + 1e-6)), duration


def resample_features(
    matrix: np.ndarray, source_rate: float, target_len: int
) -> tuple[np.ndarray, str]:
    if len(matrix) == target_len and abs(source_rate - TARGET_RATE_HZ) < 1e-6:
        return matrix.astype(np.float32, copy=False), "native_grid"
    source_t = np.arange(len(matrix), dtype=np.float64) / source_rate
    target_t = np.arange(target_len, dtype=np.float64) / TARGET_RATE_HZ
    if target_t[-1] - source_t[-1] > 1.01:
        raise ValueError(f"source_duration_shortfall:{source_t[-1]}:{target_t[-1]}")
    out = np.empty((target_len, matrix.shape[1]), dtype=np.float32)
    for i in range(matrix.shape[1]):
        col = matrix[:, i].astype(np.float64, copy=False)
        finite = np.isfinite(col)
        if not finite.any():
            out[:, i] = np.nan
        else:
            out[:, i] = np.interp(target_t, source_t[finite], col[finite]).astype(
                np.float32
            )
    return out, "linear_interpolation"


def process_row(row, lookup, required_streams, feature_set, cache_root, out_root):
    specs = lookup.get(stream_key(row), {})
    status = {
        **{
            k: row[k]
            for k in ("domain", "source_split", "model_split", "session_id", "role")
        },
        "feature_set": feature_set,
    }
    if any(name not in specs for name in required_streams):
        status.update({"status": "skipped", "reason": "missing_required_stream"})
        return None, status
    targets = load_targets(row, cache_root)
    if targets is None:
        target_len, duration = test_target_len(specs, cache_root)
        ty = sy = np.full(target_len, -1, dtype=np.int64)
        tm = sm = np.zeros(target_len, dtype=np.float32)
    else:
        ty, tm, sy, sm = targets
        target_len = len(ty)
        duration = target_len / TARGET_RATE_HZ
    if row["supervised"] != "yes":
        tm[:] = 0
        sm[:] = 0
    parts = []
    methods = []
    dims = []
    rates = []
    counts = []
    try:
        for name in required_streams:
            spec = specs[name]
            matrix, rate, dim, frames = read_pinsoro_stream(
                cache_path(cache_root, spec["local_relative_path"]),
                cache_path(cache_root, spec["binary_local_relative_path"]),
            )
            aligned, method = resample_features(matrix, rate, target_len)
            parts.append(aligned)
            methods.append(f"{name}:{method}")
            dims.append(str(dim))
            rates.append(f"{rate:.6f}")
            counts.append(str(frames))
    except Exception as exc:
        status.update({"status": "skipped", "reason": f"{type(exc).__name__}:{exc}"})
        return None, status
    x = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    out_dir = out_root / row["source_split"] / row["session_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{row['role']}.{feature_set}.30hz.npz"
    np.savez_compressed(
        out_path,
        x=x,
        task_y=ty,
        task_mask=tm,
        social_y=sy,
        social_mask=sm,
        sample_rate_hz=np.asarray([TARGET_RATE_HZ], dtype=np.float32),
        domain=np.asarray([row["domain"]]),
        role=np.asarray([row["role"]]),
        feature_set=np.asarray([feature_set]),
    )
    processed = {
        **row,
        "feature_set": feature_set,
        "tensor_relative_path": manifest_path(out_path),
        "included_streams": ";".join(required_streams),
        "stream_dims": ";".join(dims),
        "source_rates_hz": ";".join(rates),
        "declared_frame_counts": ";".join(counts),
        "alignment_methods": ";".join(methods),
        "sample_rate_hz": f"{TARGET_RATE_HZ:.3f}",
        "timeline_duration_seconds": f"{duration:.6f}",
        "n_features": str(x.shape[1]),
        "aligned_len": str(target_len),
        "task_valid_count": str(int(tm.sum())),
        "social_valid_count": str(int(sm.sum())),
    }
    status.update(
        {
            "status": "ok",
            "reason": "",
            "aligned_len": target_len,
            "timeline_duration_seconds": f"{duration:.6f}",
            "n_features": x.shape[1],
        }
    )
    return processed, status


def main() -> None:
    args = parse_args()
    definition = get_feature_set(args.feature_set)
    out_root = (
        args.out_root or PROJECT_ROOT / "processed/pinsoro" / f"{args.feature_set}_30hz"
    )
    output = (
        args.output_manifest
        or PROJECT_ROOT / "outputs/pinsoro/manifests" / f"{args.feature_set}_30hz.csv"
    )
    status_out = (
        args.status_out
        or PROJECT_ROOT
        / "outputs/pinsoro/manifests"
        / f"{args.feature_set}_30hz_status.csv"
    )
    lookup = {}
    for row in read_csv(args.streams):
        lookup.setdefault(stream_key(row), {})[row["stream_name"]] = row
    processed = []
    statuses = []
    for row in read_csv(args.manifest):
        item, status = process_row(
            row, lookup, definition.streams, args.feature_set, args.cache_root, out_root
        )
        statuses.append(status)
        if item is not None:
            processed.append(item)
    if not processed:
        raise RuntimeError(f"No tensors produced for {args.feature_set}")
    write_csv(output, list(processed[0]), processed)
    write_csv(status_out, list(statuses[0]), statuses)
    print(f"Feature set: {args.feature_set}; tensors: {len(processed)}")
    print(f"Manifest: {output}")


if __name__ == "__main__":
    main()
