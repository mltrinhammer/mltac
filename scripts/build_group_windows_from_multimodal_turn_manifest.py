"""Convert dyadic multimodal turn manifests into group-window manifests.

This is mainly for applying the group mean-pooling regression architecture to
two-person NOXI/NOXI-J sessions.  The input rows already define aligned
windows/turns and contain novice/expert tensors per modality; this script keeps
those windows but changes the manifest shape to the N-participant group format.
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

from src.acm_pipeline.io import write_csv


ROLE_ORDER = ("novice", "expert")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=[])
    return parser.parse_args()


def role_spec(spec: dict[str, object], role: str) -> dict[str, object]:
    prefix = "novice" if role == "novice" else "expert"
    return {
        "feature_set": spec.get("feature_set", ""),
        "transform_method": spec.get("transform_method", ""),
        "transform_suffix": spec.get("transform_suffix", spec.get("transform_method", "")),
        "tensor_relative_path": spec[f"{prefix}_tensor_relative_path"],
        "aligned_len": int(spec[f"{prefix}_aligned_len"]),
        "n_features": int(spec["n_features_per_role"]),
    }


def convert_row(row: dict[str, str]) -> dict[str, object]:
    modality_order = tuple(json.loads(row["modality_order_json"]))
    modalities = json.loads(row["modalities_json"])
    group_modalities: dict[str, object] = {}
    for modality_name in modality_order:
        spec = modalities[modality_name]
        group_modalities[modality_name] = {
            "roles": {
                role: role_spec(spec, role)
                for role in ROLE_ORDER
            }
        }
    return {
        "dataset": row["dataset"],
        "session_id": row["session_id"],
        "model_split": row["model_split"],
        "combo_name": row["combo_name"],
        "window_idx": int(row["turn_idx"]),
        "start_frame": int(row["start_frame"]),
        "end_frame": int(row["end_frame"]),
        "window_len": int(row["turn_len"]),
        "session_aligned_len": int(row["session_aligned_len"]),
        "role_order_json": json.dumps(list(ROLE_ORDER)),
        "modality_order_json": json.dumps(list(modality_order)),
        "modalities_json": json.dumps(group_modalities, sort_keys=True),
    }


def main() -> None:
    args = parse_args()
    datasets = set(args.datasets or [])
    rows: list[dict[str, object]] = []
    with args.input_manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if datasets and row["dataset"] not in datasets:
                continue
            rows.append(convert_row(row))
    if not rows:
        raise RuntimeError("No rows were produced.")
    write_csv(args.output_manifest, list(rows[0].keys()), rows)
    print(f"Rows: {len(rows)}")
    print(f"Output manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
