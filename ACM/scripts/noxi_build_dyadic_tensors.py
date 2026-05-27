from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Dyadic fusion stage.
#
# Input rows are still role-level transformed tensors. This script pairs novice
# and expert within each session, aligns them by frame index, and writes one
# session-level dyadic tensor where each time step contains both people.
from src.acm_pipeline.io import read_csv, write_csv


ROLE_ORDER = ("novice", "expert")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for role-level -> dyadic tensor fusion."""

    parser = argparse.ArgumentParser(description="Build time-aligned dyadic tensors from a transformed role-level manifest.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--frame-tolerance", type=int, default=25)
    return parser.parse_args()


def tensor_path(row: dict[str, str]) -> Path:
    """Resolve tensor paths from a manifest row."""

    path = Path(row["tensor_relative_path"])
    return path if path.is_absolute() else PROJECT_ROOT / path


def infer_name(rows: list[dict[str, str]]) -> str:
    """Create a stable dyadic branch name from feature/transform metadata."""

    feature_sets = sorted({row.get("feature_set", "features") for row in rows})
    suffixes = sorted({row.get("transform_suffix", row.get("transform_method", "raw")) for row in rows})
    feature_set = feature_sets[0] if len(feature_sets) == 1 else "features"
    suffix = suffixes[0] if len(suffixes) == 1 else "mixed"
    return f"{feature_set}_{suffix}_dyadic"


def default_out_root(branch_name: str) -> Path:
    return PROJECT_ROOT / "processed" / "dyadic" / branch_name


def default_manifest_path(branch_name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "manifests" / f"model_processed_manifest_{branch_name}.csv"


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    """Group manifest rows by dataset/session and role."""

    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        if row.get("role") not in ROLE_ORDER:
            continue
        key = (row["dataset"], row["session_id"])
        grouped.setdefault(key, {})[row["role"]] = row
    return grouped


def load_role_tensor(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one role-level tensor and assert the model contract."""

    with np.load(tensor_path(row), allow_pickle=True) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float32)
        target_mask = np.asarray(data["target_mask"], dtype=np.float32)

    # Fail early if a transformed tensor violates the expected role-level
    # contract. Dyadic fusion should not hide upstream preprocessing problems.
    if x.ndim != 2:
        raise ValueError(f"Expected x [time, features], got shape {x.shape} for {row['tensor_relative_path']}")
    if y.ndim != 1:
        raise ValueError(f"Expected y [time], got shape {y.shape} for {row['tensor_relative_path']}")
    if target_mask.ndim != 1:
        raise ValueError(f"Expected target_mask [time], got shape {target_mask.shape} for {row['tensor_relative_path']}")
    if len(x) != len(y) or len(y) != len(target_mask):
        raise ValueError(f"x/y/mask lengths do not match for {row['tensor_relative_path']}")
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite feature values found in {row['tensor_relative_path']}")
    if not np.isfinite(y).all():
        raise ValueError(f"Non-finite target values found in {row['tensor_relative_path']}")
    if not np.isfinite(target_mask).all():
        raise ValueError(f"Non-finite target mask values found in {row['tensor_relative_path']}")
    return x, y, target_mask


def build_dyad(
    dataset: str,
    session_id: str,
    role_rows: dict[str, dict[str, str]],
    out_root: Path,
    branch_name: str,
    frame_tolerance: int,
) -> dict[str, object]:
    """Fuse novice and expert tensors for one session."""

    missing_roles = [role for role in ROLE_ORDER if role not in role_rows]
    if missing_roles:
        raise ValueError(f"Missing roles for {dataset}/{session_id}: {missing_roles}")

    novice_row = role_rows["novice"]
    expert_row = role_rows["expert"]
    if novice_row["model_split"] != expert_row["model_split"]:
        raise ValueError(f"Split mismatch for {dataset}/{session_id}: novice={novice_row['model_split']}, expert={expert_row['model_split']}")

    novice_x, novice_y, novice_mask = load_role_tensor(novice_row)
    expert_x, expert_y, expert_mask = load_role_tensor(expert_row)
    if novice_x.shape[1] != expert_x.shape[1]:
        raise ValueError(f"Feature dimension mismatch for {dataset}/{session_id}: novice={novice_x.shape[1]}, expert={expert_x.shape[1]}")

    length_diff = abs(len(novice_x) - len(expert_x))
    if length_diff > frame_tolerance:
        raise ValueError(f"Role frame-count mismatch exceeds tolerance for {dataset}/{session_id}: diff={length_diff}")

    # Alignment is by frame index on the already 25 Hz role-level tensors. Use
    # the shared prefix if one role has a tiny tail difference.
    aligned_len = min(len(novice_x), len(expert_x))
    frame_idx = np.arange(aligned_len, dtype=np.int64)

    x = np.concatenate([novice_x[:aligned_len], expert_x[:aligned_len]], axis=1).astype(np.float32, copy=False)
    y = np.stack([novice_y[:aligned_len], expert_y[:aligned_len]], axis=1).astype(np.float32, copy=False)
    target_mask = np.stack([novice_mask[:aligned_len], expert_mask[:aligned_len]], axis=1).astype(np.float32, copy=False)

    # Explicit shape assertions document the dyadic contract and catch accidental
    # regressions if upstream manifests change.
    feature_dim = novice_x.shape[1]
    assert x.shape[-1] == 2 * feature_dim
    assert y.shape[-1] == 2
    assert target_mask.shape[-1] == 2
    assert len(frame_idx) == aligned_len

    out_dir = out_root / dataset / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}.{branch_name}.npz"
    np.savez_compressed(
        out_path,
        x=x,
        y=y,
        target_mask=target_mask,
        frame_idx=frame_idx,
        role_order=np.asarray(ROLE_ORDER),
        source_novice_tensor_relative_path=np.asarray([novice_row["tensor_relative_path"]]),
        source_expert_tensor_relative_path=np.asarray([expert_row["tensor_relative_path"]]),
        feature_set=np.asarray([novice_row.get("feature_set", "")]),
        transform_method=np.asarray([novice_row.get("transform_method", "")]),
        transform_scope=np.asarray([novice_row.get("transform_scope", "shared")]),
    )

    metadata = {
        "novice": {k: v for k, v in novice_row.items() if k not in {"tensor_relative_path"}},
        "expert": {k: v for k, v in expert_row.items() if k not in {"tensor_relative_path"}},
    }

    return {
        "dataset": dataset,
        "session_id": session_id,
        "model_split": novice_row["model_split"],
        "feature_set": novice_row.get("feature_set", ""),
        "transform_method": novice_row.get("transform_method", ""),
        "transform_scope": novice_row.get("transform_scope", "shared"),
        "transform_suffix": novice_row.get("transform_suffix", ""),
        "tensor_relative_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_novice_tensor_relative_path": novice_row["tensor_relative_path"],
        "source_expert_tensor_relative_path": expert_row["tensor_relative_path"],
        "role_order": ";".join(ROLE_ORDER),
        "n_features_per_role": str(feature_dim),
        "n_features": str(x.shape[1]),
        "aligned_len": str(aligned_len),
        "role_frame_count_diff": str(length_diff),
        "novice_target_valid_count": str(int(target_mask[:, 0].sum())),
        "expert_target_valid_count": str(int(target_mask[:, 1].sum())),
        "metadata_json": json.dumps(metadata, sort_keys=True),
    }


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_manifest)
    if not rows:
        raise RuntimeError(f"No rows in manifest: {args.input_manifest}")

    branch_name = infer_name(rows)
    out_root = args.out_root or default_out_root(branch_name)
    output_manifest = args.output_manifest or default_manifest_path(branch_name)

    processed_rows = []
    for (dataset, session_id), role_rows in sorted(group_rows(rows).items()):
        processed_rows.append(
            build_dyad(
                dataset=dataset,
                session_id=session_id,
                role_rows=role_rows,
                out_root=out_root,
                branch_name=branch_name,
                frame_tolerance=args.frame_tolerance,
            )
        )

    if not processed_rows:
        raise RuntimeError("No dyadic rows were produced.")

    write_csv(output_manifest, list(processed_rows[0].keys()), processed_rows)
    print(f"Input manifest: {args.input_manifest}")
    print(f"Dyadic branch: {branch_name}")
    print(f"Wrote dyadic rows: {len(processed_rows)}")
    print(f"Output manifest: {output_manifest}")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()

