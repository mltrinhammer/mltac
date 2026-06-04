from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.io import read_csv, write_csv


JOIN_FIELDS = (
    "dataset",
    "session_id",
    "model_split",
    "turn_idx",
    "speaker",
    "start_frame",
    "end_frame",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join unimodal paired turn manifests into one multimodal turn manifest."
    )
    parser.add_argument("--input-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--combo-name", default="")
    return parser.parse_args()


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in JOIN_FIELDS)


def infer_combo_name(feature_sets: list[str]) -> str:
    return "__".join(feature_sets)


def default_manifest_path(combo_name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "manifests" / f"model_processed_manifest_{combo_name}_multimodal_turns.csv"


def load_manifest_rows(path: Path) -> tuple[str, dict[tuple[str, ...], dict[str, str]]]:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"No rows in manifest: {path}")

    feature_sets = sorted({row.get("feature_set", "") for row in rows})
    if len(feature_sets) != 1 or not feature_sets[0]:
        raise RuntimeError(f"Expected one feature_set in {path}, got: {feature_sets}")
    feature_set = feature_sets[0]

    keyed_rows: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = row_key(row)
        if key in keyed_rows:
            raise RuntimeError(f"Duplicate turn key in {path}: {key}")
        keyed_rows[key] = row
    return feature_set, keyed_rows


def build_multimodal_rows(
    manifests: list[Path],
    combo_name: str,
) -> list[dict[str, object]]:
    feature_sets: list[str] = []
    manifest_rows: dict[str, dict[tuple[str, ...], dict[str, str]]] = {}

    for manifest_path in manifests:
        feature_set, keyed_rows = load_manifest_rows(manifest_path)
        if feature_set in manifest_rows:
            raise RuntimeError(f"Duplicate feature set in multimodal join: {feature_set}")
        feature_sets.append(feature_set)
        manifest_rows[feature_set] = keyed_rows

    shared_keys = set.intersection(*(set(rows.keys()) for rows in manifest_rows.values()))
    if not shared_keys:
        raise RuntimeError("No shared turn keys were found across the requested manifests.")

    feature_sets = list(feature_sets)
    multimodal_rows: list[dict[str, object]] = []
    for key in sorted(shared_keys):
        base_row = manifest_rows[feature_sets[0]][key]
        modalities: dict[str, dict[str, object]] = {}
        session_aligned_lens: list[int] = []
        for feature_set in feature_sets:
            row = manifest_rows[feature_set][key]
            for field in ("turn_len", "dataset", "session_id", "model_split", "speaker", "start_frame", "end_frame", "turn_idx"):
                if str(row[field]) != str(base_row[field]):
                    raise RuntimeError(
                        f"Join mismatch for feature set {feature_set} and key {key}: field {field!r} differs."
                    )
            session_aligned_lens.append(int(row["session_aligned_len"]))
            modalities[feature_set] = {
                "feature_set": row["feature_set"],
                "transform_method": row.get("transform_method", ""),
                "transform_suffix": row.get("transform_suffix", ""),
                "n_features_per_role": int(row["n_features_per_role"]),
                "novice_tensor_relative_path": row["novice_tensor_relative_path"],
                "expert_tensor_relative_path": row["expert_tensor_relative_path"],
                "novice_aligned_len": int(row["novice_aligned_len"]),
                "expert_aligned_len": int(row["expert_aligned_len"]),
                "session_aligned_len": int(row["session_aligned_len"]),
            }

        multimodal_rows.append(
            {
                "dataset": base_row["dataset"],
                "session_id": base_row["session_id"],
                "model_split": base_row["model_split"],
                "combo_name": combo_name,
                "feature_set_combo": "+".join(feature_sets),
                "turn_idx": base_row["turn_idx"],
                "speaker": base_row["speaker"],
                "start_frame": base_row["start_frame"],
                "end_frame": base_row["end_frame"],
                "turn_len": base_row["turn_len"],
                "session_aligned_len": min(session_aligned_lens),
                "modality_order_json": json.dumps(feature_sets),
                "modalities_json": json.dumps(modalities, sort_keys=True),
            }
        )

    return multimodal_rows


def main() -> None:
    args = parse_args()
    feature_sets = []
    for manifest_path in args.input_manifests:
        rows = read_csv(manifest_path)
        if not rows:
            raise RuntimeError(f"No rows in manifest: {manifest_path}")
        manifest_feature_sets = sorted({row.get("feature_set", "") for row in rows})
        if len(manifest_feature_sets) != 1 or not manifest_feature_sets[0]:
            raise RuntimeError(f"Expected one feature_set in {manifest_path}, got: {manifest_feature_sets}")
        feature_sets.append(manifest_feature_sets[0])

    combo_name = args.combo_name.strip() or infer_combo_name(feature_sets)
    output_manifest = args.output_manifest or default_manifest_path(combo_name)
    rows = build_multimodal_rows(args.input_manifests, combo_name=combo_name)
    write_csv(output_manifest, list(rows[0].keys()), rows)
    print(f"Combo name: {combo_name}")
    print(f"Modalities: {', '.join(feature_sets)}")
    print(f"Wrote multimodal rows: {len(rows)}")
    print(f"Output manifest: {output_manifest}")


if __name__ == "__main__":
    main()