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
from src.acm_pipeline.io import local_cache_path, read_csv, read_stream_matrix, read_target, write_csv


DEFAULT_CACHE_ROOT = PROJECT_ROOT / "cache"
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "model_raw_manifest_train_with_split.csv"
DEFAULT_STREAMS = PROJECT_ROOT / "outputs" / "model_raw_manifest_streams_train.csv"

# This script is the common first preprocessing step. It does not normalize or
# reduce dimensionality; it only builds one aligned 25 Hz tensor set that later
# transform branches can reuse.

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one feature-set alignment run."""

    parser = argparse.ArgumentParser(description="Prepare one feature set as 25 Hz aligned NPZ tensors.")
    parser.add_argument("--feature-set", required=True, choices=sorted(FEATURE_SETS))
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--streams", type=Path, default=DEFAULT_STREAMS)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--processed-manifest", type=Path, default=None)
    parser.add_argument("--status-out", type=Path, default=None)
    return parser.parse_args()


def stream_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["dataset"], row["session_id"], row["role"]


def build_stream_lookup(stream_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, dict[str, str]]]:
    """Index stream metadata by dataset/session/role and stream name."""

    lookup: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for row in stream_rows:
        lookup.setdefault(stream_key(row), {})[row["stream_name"]] = row
    return lookup


def default_out_root(feature_set: str) -> Path:
    return PROJECT_ROOT / "processed" / f"{feature_set}_25hz"


def default_manifest_path(feature_set: str) -> Path:
    return PROJECT_ROOT / "outputs" / "manifests" / f"model_processed_manifest_{feature_set}_25hz.csv"


def process_row(
    row: dict[str, str],
    stream_lookup: dict[tuple[str, str, str], dict[str, dict[str, str]]],
    feature_set_name: str,
    cache_root: Path,
    out_root: Path,
    required_streams: tuple[str, ...],
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Build one aligned tensor row and one status row.

    For each requested stream, the script reads the local cached SSI header and
    binary, aligns it to the target annotation length, concatenates streams when
    needed, and writes one compressed NPZ file for later transform/model steps.
    """

    dataset = row["dataset"]
    session_id = row["session_id"]
    role = row["role"]
    key = (dataset, session_id, role)

    # A status row is produced for every candidate example, including skipped
    # rows. This makes UCloud cache/debug runs auditable without reading logs.
    status: dict[str, object] = {
        "dataset": dataset,
        "session_id": session_id,
        "role": role,
        "model_split": row.get("model_split", ""),
        "feature_set": feature_set_name,
        "required_streams": ";".join(required_streams),
        "status": "ok",
        "reason": "",
    }

    target_rel = row.get("target_relative_path", "")
    has_target = bool(target_rel)
    if has_target:
        target_path = local_cache_path(cache_root, dataset, target_rel)
        if not target_path.exists():
            status.update({"status": "skipped", "reason": f"missing_target:{target_rel}"})
            return None, status
        # Targets are kept as two arrays: a NaN-filled raw target becomes y
        # with NaNs replaced by zero, while target_mask records valid frames.
        y_raw = read_target(target_path)
        target_mask = np.isfinite(y_raw).astype(np.float32)
        y = np.nan_to_num(y_raw, nan=0.0).astype(np.float32)
        reference_len = len(y)
    else:
        # Test session without engagement labels. We read streams first and
        # derive the reference length from their frame counts afterwards.
        y_raw = None
        y = None
        target_mask = None
        reference_len = None

    # Look up all streams for this dataset/session/role and collect the subset
    # required by the requested feature-set registry entry.
    available = stream_lookup.get(key, {})
    x_parts: list[np.ndarray] = []
    included_streams: list[str] = []
    included_dims: list[str] = []
    included_rates: list[str] = []
    alignment_methods: list[str] = []
    missing_streams: list[str] = []
    stream_lengths: list[int] = []
    raw_stream_lengths: list[int] = []

    for stream_name in required_streams:
        # A missing stream means the example cannot be used for this full
        # feature-set experiment; the status CSV records the reason.
        stream = available.get(stream_name)
        if stream is None:
            missing_streams.append(stream_name)
            continue
        header_path = local_cache_path(cache_root, dataset, stream["local_relative_path"])
        binary_path = local_cache_path(cache_root, dataset, stream["binary_local_relative_path"])
        if not header_path.exists() or not binary_path.exists():
            missing_streams.append(stream_name)
            continue
        mat, sr, dim = read_stream_matrix(header_path, binary_path)
        raw_stream_lengths.append(len(mat))

        if reference_len is not None:
            # Each stream is read independently and aligned to the target
            # length before concatenation.
            aligned, method = align_to_target_grid(mat, sr, reference_len)
        else:
            # No target to align against — keep the stream at its native
            # length and record it for later min-length determination.
            aligned = mat.astype(np.float32, copy=False)
            method = "native_no_target"

        if len(aligned) == 0:
            missing_streams.append(stream_name)
            continue
        x_parts.append(aligned)
        included_streams.append(stream_name)
        included_dims.append(str(dim))
        included_rates.append(f"{sr:.3f}")
        alignment_methods.append(f"{stream_name}:{method}")
        stream_lengths.append(len(aligned))

    if missing_streams:
        status["missing_streams"] = ";".join(missing_streams)
        if len(x_parts) == 0 or set(missing_streams) & set(required_streams):
            status.update({"status": "skipped", "reason": "missing_required_streams"})
            return None, status

    # Concatenate streams after alignment. If streams differ slightly in
    # length, use the shared prefix so x, y, and target_mask are identical.
    if has_target:
        aligned_len = min([len(y), *stream_lengths])
        y_out = y[:aligned_len]
        mask_out = target_mask[:aligned_len]
        target_n_values = len(y_raw)
    else:
        # Test session: derive aligned_len from the minimum stream length and
        # fill dummy y / target_mask so the NPZ schema stays identical.
        aligned_len = min(stream_lengths)
        y_out = np.zeros(aligned_len, dtype=np.float32)
        mask_out = np.zeros(aligned_len, dtype=np.float32)
        target_n_values = 0

    x = np.concatenate([part[:aligned_len] for part in x_parts], axis=1).astype(np.float32, copy=False)

    out_dir = out_root / dataset / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{role}.{feature_set_name}.25hz.npz"
    # Store both model arrays and provenance metadata in the NPZ. The metadata
    # lets later scripts verify which streams/rates/alignment methods produced
    # a given tensor.
    np.savez_compressed(
        out_path,
        x=x,
        y=y_out,
        target_mask=mask_out,
        stream_names=np.asarray(included_streams),
        stream_dims=np.asarray(included_dims),
        stream_source_rates=np.asarray(included_rates),
        stream_alignment_methods=np.asarray(alignment_methods),
        sample_rate_hz=np.asarray([TARGET_RATE_HZ], dtype=np.float32),
        feature_set=np.asarray([feature_set_name]),
    )

    processed = {
        "dataset": dataset,
        "session_id": session_id,
        "role": role,
        "model_split": row.get("model_split", ""),
        "feature_set": feature_set_name,
        "sample_rate_hz": f"{TARGET_RATE_HZ:.3f}",
        "tensor_relative_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "included_streams": ";".join(included_streams),
        "stream_dims": ";".join(included_dims),
        "stream_source_rates": ";".join(included_rates),
        "stream_alignment_methods": ";".join(alignment_methods),
        "n_features": str(x.shape[1]),
        "aligned_len": str(aligned_len),
        "target_n_values": str(target_n_values),
        "target_valid_count": str(int(mask_out.sum())),
        "target_nan_count": str(int((1.0 - mask_out).sum())),
        "dropped_target_tail_frames": str(max(0, target_n_values - aligned_len)),
    }
    status.update(
        {
            "status": "ok",
            "reason": "",
            "included_streams": processed["included_streams"],
            "n_features": processed["n_features"],
            "aligned_len": processed["aligned_len"],
        }
    )
    return processed, status


def main() -> None:
    args = parse_args()
    feature_set = get_feature_set(args.feature_set)

    # Default outputs are named by feature set so UCloud runs can process each
    # modality independently and keep their manifests side by side.
    out_root = args.out_root or default_out_root(args.feature_set)
    processed_manifest = args.processed_manifest or default_manifest_path(args.feature_set)
    status_out = args.status_out or PROJECT_ROOT / "outputs" / "manifests" / f"feature_status_{args.feature_set}_25hz.csv"

    # The raw manifest supplies target paths and split labels; the stream
    # manifest supplies the local stream paths and declared stream metadata.
    manifest_rows = read_csv(args.manifest)
    stream_lookup = build_stream_lookup(read_csv(args.streams))

    processed_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for row in manifest_rows:
        # Only split-assigned expert/novice examples are valid modelling rows.
        if row.get("role") not in {"expert", "novice"}:
            continue
        if not row.get("model_split") or row.get("model_split") == "unassigned":
            continue
        processed, status = process_row(
            row=row,
            stream_lookup=stream_lookup,
            feature_set_name=args.feature_set,
            cache_root=args.cache_root,
            out_root=out_root,
            required_streams=feature_set.streams,
        )
        status_rows.append(status)
        if processed is not None:
            processed_rows.append(processed)

    processed_fields = [
        "dataset",
        "session_id",
        "role",
        "model_split",
        "feature_set",
        "sample_rate_hz",
        "tensor_relative_path",
        "included_streams",
        "stream_dims",
        "stream_source_rates",
        "stream_alignment_methods",
        "n_features",
        "aligned_len",
        "target_n_values",
        "target_valid_count",
        "target_nan_count",
        "dropped_target_tail_frames",
    ]
    status_fields = [
        "dataset",
        "session_id",
        "role",
        "model_split",
        "feature_set",
        "required_streams",
        "status",
        "reason",
        "missing_streams",
        "included_streams",
        "n_features",
        "aligned_len",
    ]
    # Write both the train-ready manifest and the status table. Downstream
    # transform/model scripts consume the processed manifest only.
    write_csv(processed_manifest, processed_fields, processed_rows)
    write_csv(status_out, status_fields, status_rows)
    print(f"Feature set: {args.feature_set}")
    print(f"Wrote processed rows: {len(processed_rows)}")
    print(f"Processed manifest: {processed_manifest}")
    print(f"Status CSV: {status_out}")


if __name__ == "__main__":
    main()
