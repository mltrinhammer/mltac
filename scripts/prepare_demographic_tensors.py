"""Prepare demographic feature tensors (age, gender, language) from annotation CSVs.

Each session/role gets a ``[T, 3]`` tensor where ``T = aligned_len`` (matching
other feature sets) and the three features are ``[age, gender_code, language_code]``.
All values are static per participant and broadcast identically to every frame.

Gender encoding: male/m -> 0.0, female/f -> 1.0
Language encoding: integer code per language (z-score normalized downstream)
Age: raw numeric value (z-score normalized downstream by Step 2)

The script reads a **reference** 25 Hz manifest (any existing feature set) to
obtain session metadata and aligned_len, and copies ``y`` / ``target_mask``
from the reference NPZ so the output schema is identical to other 25 Hz tensors.

Usage::

    python scripts/prepare_demographic_tensors.py \\
        --reference-manifest ACM/outputs/manifests/model_processed_manifest_audio_egemaps_25hz.csv \\
        --data-root /path/to/data \\
        --out-root ACM/processed/demographic_25hz \\
        --processed-manifest ACM/outputs/manifests/model_processed_manifest_demographic_25hz.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.io import read_csv, write_csv


FEATURE_SET_NAME = "demographic"
N_FEATURES = 3  # [age, gender, language]

GENDER_MAP: dict[str, float] = {
    "male": 0.0,
    "m": 0.0,
    "0": 0.0,
    "0.0": 0.0,
    "female": 1.0,
    "f": 1.0,
    "1": 1.0,
    "1.0": 1.0,
}

# Integer codes for languages found across NoXi / NoXi-J / MPIIG.
# Values are arbitrary — z-score normalization makes the spacing irrelevant.
LANGUAGE_MAP: dict[str, float] = {
    "arabic": 0.0,
    "english": 1.0,
    "french": 2.0,
    "german": 3.0,
    "indonesian": 4.0,
    "japanese": 5.0,
    "spanish": 6.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare demographic feature tensors (age, gender) from annotation CSVs.",
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        required=True,
        help="Any existing 25 Hz manifest to get session metadata and aligned_len.",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--processed-manifest", type=Path, default=None)
    parser.add_argument(
        "--valid-roles",
        nargs="*",
        default=None,
        help="Roles to process. Default: all roles found in reference manifest.",
    )
    return parser.parse_args()


def locate_annotation(
    data_root: Path,
    dataset: str,
    session_id: str,
    role: str,
    attribute: str,
) -> Path | None:
    """Find an annotation CSV for *attribute* in the session directory.

    Tries role-specific ``{role}.{attribute}.annotation.csv`` first, then
    falls back to session-level ``{attribute}.annotation.csv`` (e.g. language
    is shared across all participants in a dyad/group).

    Searches flat layout (sessions directly in dataset root) and hierarchical
    layout (split subdirectory between dataset and session).
    """
    candidates = [
        f"{role}.{attribute}.annotation.csv",
        f"{attribute}.annotation.csv",
    ]
    base = data_root / dataset
    if not base.exists():
        return None
    for filename in candidates:
        # Flat layout: dataset/session_id/
        flat_candidate = base / session_id / filename
        if flat_candidate.exists():
            return flat_candidate
        # Hierarchical layout: dataset/split/session_id/
        for split_dir in sorted(base.iterdir()):
            if not split_dir.is_dir():
                continue
            candidate = split_dir / session_id / filename
            if candidate.exists():
                return candidate
    return None


def read_annotation_third_column(path: Path) -> str:
    """Read the value from the 3rd column (0-indexed: 2) of a 1-row annotation CSV.

    The organizer annotation CSVs have no header — just a single data row with
    four semicolon-separated columns: ``start;end;value;confidence``.
    The delimiter is auto-detected for robustness.
    """
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)

        delimiter = ";"
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
            delimiter = dialect.delimiter
        except csv.Error:
            pass

        reader = csv.reader(handle, delimiter=delimiter)
        data_row = next(reader, None)
        if data_row is None or len(data_row) < 3:
            raise RuntimeError(f"Could not read 3rd column from {path}")
        return data_row[2].strip()


def encode_gender(value: str) -> float:
    """Encode a gender string to a numeric code."""
    v = value.strip().lower()
    if v in GENDER_MAP:
        return GENDER_MAP[v]
    try:
        return float(v)
    except ValueError:
        return float("nan")


def parse_age(value: str) -> float:
    """Parse an age value to float."""
    try:
        return float(value.strip())
    except (ValueError, TypeError):
        return float("nan")


def encode_language(value: str) -> float:
    """Encode a language string to a numeric code."""
    v = value.strip().lower()
    if v in LANGUAGE_MAP:
        return LANGUAGE_MAP[v]
    try:
        return float(v)
    except ValueError:
        return float("nan")


def main() -> None:
    args = parse_args()
    out_root = args.out_root or PROJECT_ROOT / "processed" / f"{FEATURE_SET_NAME}_25hz"
    processed_manifest = args.processed_manifest or (
        PROJECT_ROOT / "outputs" / "manifests" / f"model_processed_manifest_{FEATURE_SET_NAME}_25hz.csv"
    )

    ref_rows = read_csv(args.reference_manifest)
    if not ref_rows:
        raise RuntimeError(f"No rows in reference manifest: {args.reference_manifest}")

    valid_roles = set(args.valid_roles) if args.valid_roles else None

    processed_rows: list[dict[str, object]] = []
    stats: dict[str, int] = {
        "found_all_3": 0,
        "found_age_gender": 0,
        "found_partial": 0,
        "found_none": 0,
        "skipped_role": 0,
    }

    for ref_row in ref_rows:
        dataset = ref_row["dataset"]
        session_id = ref_row["session_id"]
        role = ref_row["role"]
        model_split = ref_row["model_split"]
        aligned_len = int(ref_row["aligned_len"])

        if valid_roles is not None and role not in valid_roles:
            stats["skipped_role"] += 1
            continue

        # Locate annotation CSVs.
        age_path = locate_annotation(args.data_root, dataset, session_id, role, "age")
        gender_path = locate_annotation(args.data_root, dataset, session_id, role, "gender")
        language_path = locate_annotation(args.data_root, dataset, session_id, role, "language")

        # Default to 0.0 for missing attributes.  After z-score normalization a
        # universally-missing attribute becomes a constant across all samples and
        # carries no information.  When only *some* sessions lack the attribute,
        # 0.0 maps to a non-mean z-score — acceptable for the small number of
        # partially-missing cases (the normalizer's mean is dominated by valid
        # observations).
        age_value = 0.0
        gender_value = 0.0
        language_value = 0.0
        has_age = False
        has_gender = False
        has_language = False

        if age_path is not None:
            raw_age = parse_age(read_annotation_third_column(age_path))
            if not np.isnan(raw_age):
                age_value = raw_age
                has_age = True

        if gender_path is not None:
            raw_gender = encode_gender(read_annotation_third_column(gender_path))
            if not np.isnan(raw_gender):
                gender_value = raw_gender
                has_gender = True

        if language_path is not None:
            raw_language = encode_language(read_annotation_third_column(language_path))
            if not np.isnan(raw_language):
                language_value = raw_language
                has_language = True

        n_found = sum([has_age, has_gender, has_language])
        if n_found == 3:
            stats["found_all_3"] += 1
        elif n_found == 2 and has_age and has_gender:
            stats["found_age_gender"] += 1
        elif n_found > 0:
            stats["found_partial"] += 1
        else:
            stats["found_none"] += 1

        # Create [T, 3] tensor: [age, gender, language] broadcast to all frames.
        x = np.zeros((aligned_len, N_FEATURES), dtype=np.float32)
        x[:, 0] = age_value
        x[:, 1] = gender_value
        x[:, 2] = language_value

        # Copy y and target_mask from the reference feature set's NPZ.
        ref_tensor_path = Path(ref_row["tensor_relative_path"])
        if not ref_tensor_path.is_absolute():
            ref_tensor_path = PROJECT_ROOT / ref_tensor_path
        with np.load(ref_tensor_path, allow_pickle=True) as data:
            y = np.asarray(data["y"], dtype=np.float32)[:aligned_len]
            target_mask = np.asarray(data["target_mask"], dtype=np.float32)[:aligned_len]

        # Save NPZ.
        out_dir = out_root / dataset / session_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{role}.{FEATURE_SET_NAME}.25hz.npz"
        np.savez_compressed(
            out_path,
            x=x,
            y=y,
            target_mask=target_mask,
            stream_names=np.asarray(["age", "gender", "language"]),
            stream_dims=np.asarray(["1", "1", "1"]),
            stream_source_rates=np.asarray(["0.000", "0.000", "0.000"]),
            stream_alignment_methods=np.asarray(["broadcast", "broadcast", "broadcast"]),
            sample_rate_hz=np.asarray([25.0], dtype=np.float32),
            feature_set=np.asarray([FEATURE_SET_NAME]),
        )

        processed_rows.append(
            {
                "dataset": dataset,
                "session_id": session_id,
                "role": role,
                "model_split": model_split,
                "feature_set": FEATURE_SET_NAME,
                "sample_rate_hz": "25.000",
                "tensor_relative_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "included_streams": "age;gender;language",
                "stream_dims": "1;1;1",
                "stream_source_rates": "0.000;0.000;0.000",
                "stream_alignment_methods": "broadcast;broadcast;broadcast",
                "n_features": str(N_FEATURES),
                "aligned_len": str(aligned_len),
                "target_n_values": ref_row.get("target_n_values", "0"),
                "target_valid_count": ref_row.get("target_valid_count", "0"),
                "target_nan_count": ref_row.get("target_nan_count", "0"),
                "dropped_target_tail_frames": "0",
            }
        )

    if not processed_rows:
        raise RuntimeError("No demographic tensor rows produced.")

    write_csv(processed_manifest, list(processed_rows[0].keys()), processed_rows)

    print(f"Reference manifest: {args.reference_manifest}")
    print(f"Data root: {args.data_root}")
    print(f"Wrote {len(processed_rows)} demographic tensor(s)  [age, gender, language]")
    print(f"  found_all_3={stats['found_all_3']}")
    print(f"  found_age_gender={stats['found_age_gender']}")
    print(f"  found_partial={stats['found_partial']}")
    print(f"  found_none={stats['found_none']}")
    print(f"  skipped_role={stats['skipped_role']}")
    print(f"Output manifest: {processed_manifest}")


if __name__ == "__main__":
    main()
