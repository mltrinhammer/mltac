"""Validate PinSoRo tensors, shared timebases, and window manifests."""

from __future__ import annotations
import argparse, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.acm_pipeline.io import read_csv, write_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate PinSoRo Stage 1 outputs.")
    p.add_argument("--manifests", type=Path, nargs="+", required=True)
    p.add_argument("--window-manifests", type=Path, nargs="*", default=[])
    p.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "outputs/pinsoro/validation"
    )
    return p.parse_args()


def resolve(text: str) -> Path:
    p = Path(text)
    return p if p.is_absolute() else PROJECT_ROOT / p


def main() -> None:
    args = parse_args()
    integrity = []
    label_counts = defaultdict(int)
    reference = {}
    cross = []
    for manifest in args.manifests:
        for row in read_csv(manifest):
            path = resolve(row["tensor_relative_path"])
            status = "ok"
            reasons = []
            shapes = {}
            key = (row["source_split"], row["session_id"], row["role"])
            try:
                with np.load(path, allow_pickle=True) as data:
                    arrays = {
                        name: np.asarray(data[name])
                        for name in (
                            "x",
                            "task_y",
                            "task_mask",
                            "social_y",
                            "social_mask",
                        )
                    }
                    shapes = {name: value.shape for name, value in arrays.items()}
                    length = len(arrays["x"])
                    if any(len(arrays[name]) != length for name in arrays):
                        reasons.append("internal_length_mismatch")
                    if not np.isfinite(arrays["x"]).all():
                        reasons.append("nonfinite_features")
                    if float(np.asarray(data["sample_rate_hz"])[0]) != 30.0:
                        reasons.append("not_30hz")
                    signature = (
                        length,
                        arrays["task_y"],
                        arrays["task_mask"],
                        arrays["social_y"],
                        arrays["social_mask"],
                    )
                    if key not in reference:
                        reference[key] = (row["feature_set"], signature)
                    else:
                        ref_feature, ref = reference[key]
                        mismatch = []
                        if length != ref[0]:
                            mismatch.append("length")
                        for idx, name in enumerate(
                            ("task_y", "task_mask", "social_y", "social_mask"), start=1
                        ):
                            if not np.array_equal(signature[idx], ref[idx]):
                                mismatch.append(name)
                        cross.append(
                            {
                                "source_split": key[0],
                                "session_id": key[1],
                                "role": key[2],
                                "reference_feature": ref_feature,
                                "feature_set": row["feature_set"],
                                "status": "error" if mismatch else "ok",
                                "reason": ";".join(mismatch),
                                "aligned_len": length,
                            }
                        )
                    for head in ("task", "social"):
                        labels = arrays[f"{head}_y"].astype(np.int64)
                        mask = arrays[f"{head}_mask"].astype(bool)
                        for value, count in Counter(labels[mask].tolist()).items():
                            label_counts[
                                (
                                    row["feature_set"],
                                    row["domain"],
                                    row["model_split"],
                                    head,
                                    int(value),
                                )
                            ] += count
            except Exception as exc:
                reasons.append(f"{type(exc).__name__}:{exc}")
            if reasons:
                status = "error"
            integrity.append(
                {
                    "manifest": str(manifest),
                    "feature_set": row.get("feature_set", ""),
                    "domain": row["domain"],
                    "source_split": row["source_split"],
                    "model_split": row["model_split"],
                    "session_id": row["session_id"],
                    "role": row["role"],
                    "status": status,
                    "reason": ";".join(reasons),
                    "x_shape": str(shapes.get("x", "")),
                }
            )
    label_rows = [
        {
            "feature_set": k[0],
            "domain": k[1],
            "model_split": k[2],
            "label_head": k[3],
            "class_id": k[4],
            "count": v,
        }
        for k, v in sorted(label_counts.items())
    ]
    window_rows = []
    for manifest in args.window_manifests:
        for (domain, split), count in sorted(
            Counter((r["domain"], r["model_split"]) for r in read_csv(manifest)).items()
        ):
            window_rows.append(
                {
                    "manifest": str(manifest),
                    "domain": domain,
                    "model_split": split,
                    "count": count,
                }
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "tensor_integrity.csv", list(integrity[0]), integrity)
    write_csv(args.out_dir / "cross_modality_alignment.csv", list(cross[0]), cross)
    if label_rows:
        write_csv(
            args.out_dir / "label_class_counts.csv", list(label_rows[0]), label_rows
        )
    if window_rows:
        write_csv(args.out_dir / "window_counts.csv", list(window_rows[0]), window_rows)
    errors = sum(r["status"] != "ok" for r in integrity) + sum(
        r["status"] != "ok" for r in cross
    )
    print(
        f"Validated tensors: {len(integrity)}; cross-modality comparisons: {len(cross)}; errors: {errors}"
    )
    print(f"Output directory: {args.out_dir}")
    if errors:
        raise RuntimeError(f"Validation found {errors} errors.")


if __name__ == "__main__":
    main()
