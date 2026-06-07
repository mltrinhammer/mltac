"""Validate PinSoRo processed tensors and window manifests."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.io import read_csv, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PinSoRo Stage 1 outputs.")
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--window-manifests", type=Path, nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "pinsoro" / "validation")
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    integrity_rows: list[dict[str, object]] = []
    label_counts: dict[tuple[str, str, str, str, int], int] = defaultdict(int)
    for manifest in args.manifests:
        for row in read_csv(manifest):
            path = resolve(row["tensor_relative_path"])
            status, reason = "ok", ""
            shapes: dict[str, tuple[int, ...]] = {}
            try:
                with np.load(path, allow_pickle=True) as data:
                    for key in ("x", "task_y", "task_mask", "social_y", "social_mask"):
                        shapes[key] = tuple(data[key].shape)
                    length = shapes["x"][0]
                    if any(shapes[key][0] != length for key in ("task_y", "task_mask", "social_y", "social_mask")):
                        status, reason = "error", "length_mismatch"
                    for head in ("task", "social"):
                        labels = np.asarray(data[f"{head}_y"], dtype=np.int64)
                        mask = np.asarray(data[f"{head}_mask"], dtype=np.float32) > 0
                        for value, count in Counter(labels[mask].tolist()).items():
                            label_counts[(row["feature_set"], row["domain"], row["model_split"], head, int(value))] += count
            except Exception as exc:
                status, reason = "error", f"{type(exc).__name__}:{exc}"
            integrity_rows.append({
                "manifest": str(manifest),
                "feature_set": row.get("feature_set", ""),
                "domain": row["domain"],
                "source_split": row["source_split"],
                "model_split": row["model_split"],
                "session_id": row["session_id"],
                "role": row["role"],
                "status": status,
                "reason": reason,
                "x_shape": str(shapes.get("x", "")),
            })

    label_rows = [
        {"feature_set": key[0], "domain": key[1], "model_split": key[2], "label_head": key[3], "class_id": key[4], "count": count}
        for key, count in sorted(label_counts.items())
    ]
    window_rows: list[dict[str, object]] = []
    for manifest in args.window_manifests:
        counts = Counter((row["domain"], row["model_split"]) for row in read_csv(manifest))
        for (domain, model_split), count in sorted(counts.items()):
            window_rows.append({"manifest": str(manifest), "domain": domain, "model_split": model_split, "count": count})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "tensor_integrity.csv", list(integrity_rows[0].keys()), integrity_rows)
    if label_rows:
        write_csv(args.out_dir / "label_class_counts.csv", list(label_rows[0].keys()), label_rows)
    if window_rows:
        write_csv(args.out_dir / "window_counts.csv", list(window_rows[0].keys()), window_rows)
    errors = sum(row["status"] != "ok" for row in integrity_rows)
    print(f"Validated tensors: {len(integrity_rows)}; errors: {errors}")
    print(f"Output directory: {args.out_dir}")
    if errors:
        raise RuntimeError(f"Validation found {errors} tensor errors.")


if __name__ == "__main__":
    main()
