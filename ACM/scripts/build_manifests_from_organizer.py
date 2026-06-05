"""Build ACM raw manifests and cache symlinks from organizer-format data.

Scans the organizer data layout (as extracted on the HPC) and produces:
  1. Symlinks in ACM/cache/ pointing to real data split directories
  2. outputs/model_raw_manifest_train_with_split.csv  (session-level)
  3. outputs/model_raw_manifest_streams_train.csv      (stream-level)

These two CSVs feed directly into the existing ACM preprocessing pipeline:
  noxi_prepare_feature_tensors_25hz.py -> transform scripts -> training scripts

Usage:
    python scripts/build_manifests_from_organizer.py --data-root /home/mlut/mltac
    python scripts/build_manifests_from_organizer.py --data-root /home/mlut/mltac --datasets noxi noxij
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.io import write_csv

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

DATASETS = {
    "noxi": {
        "cache_dir": "noxi_a",
        "roles": ("expert", "novice"),
        "splits": {
            "train": "train_internal",
            "val": "val_internal",
            "test-base": "test_internal",
            "test-additional": "test_additional",
        },
    },
    "noxij": {
        "cache_dir": "noxi_b",
        "roles": ("expert", "novice"),
        "splits": {
            "train": "train_internal",
            "val": "val_internal",
            "test": "test_internal",
        },
    },
    "mpiigroupinteraction": {
        "cache_dir": "mpiii",
        "roles": "auto",
        "flat": True,
        "splits": {
            "test": "test",
        },
    },
}

TARGET_SUFFIX = "engagement.annotation.csv"

# All streams known from the organizer feature set.
KNOWN_STREAMS = (
    "audio.egemapsv2",
    "audio.w2vbert2_embeddings",
    "audio.xlm_roberta_embeddings",
    "clip",
    "dino",
    "openface2",
    "openface3",
    "openpose",
    "swin",
    "videomae",
)

# Reference stream used for auto-discovering participant roles from filenames.
_AUTO_DISCOVER_STREAM = KNOWN_STREAMS[0]  # "audio.egemapsv2"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_symlink(target: Path, link: Path) -> None:
    """Create a directory symlink, skipping if it already points correctly."""
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"{link} exists and is not a symlink")
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(target.resolve()), str(link), target_is_directory=True)


def discover_sessions(split_dir: Path) -> list[str]:
    """Return sorted session IDs found as subdirectories of a split dir."""
    if not split_dir.is_dir():
        return []
    return sorted(
        d.name
        for d in split_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def discover_roles(session_dir: Path) -> tuple[str, ...]:
    """Auto-discover participant role names from stream filenames.

    Scans for files matching ``*.{_AUTO_DISCOVER_STREAM}.stream~`` and extracts
    the prefix before the first dot as the role name.  Returns a sorted tuple of
    unique role names found (e.g. ``("subjectPos1", "subjectPos2", ...)``).
    """
    suffix = f".{_AUTO_DISCOVER_STREAM}.stream~"
    roles: set[str] = set()
    for child in session_dir.iterdir():
        if child.name.endswith(suffix):
            role = child.name[: -len(suffix)]
            if role:
                roles.add(role)
    if not roles:
        return ()
    return tuple(sorted(roles))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ACM manifests from organizer-format data directories."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Project root containing noxi/, noxij/ data directories.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=sorted(DATASETS.keys()),
        help="Which datasets to include (default: all).",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "cache",
        help="Where to create symlinks (default: ACM/cache/).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Where to write manifest CSVs (default: ACM/outputs/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root: Path = args.data_root.resolve()

    manifest_rows: list[dict[str, str]] = []
    stream_rows: list[dict[str, str]] = []

    for dataset_name in args.datasets:
        cfg = DATASETS[dataset_name]
        dataset_dir = data_root / dataset_name
        if not dataset_dir.is_dir():
            print(f"WARNING: {dataset_dir} not found, skipping {dataset_name}")
            continue

        cache_dataset = args.cache_root / cfg["cache_dir"]
        flat_layout = cfg.get("flat", False)
        if not flat_layout:
            cache_dataset.mkdir(parents=True, exist_ok=True)

        for split_dirname, model_split in cfg["splits"].items():
            if flat_layout:
                # Flat layout: sessions live directly in the dataset root,
                # not under a split subdirectory.  We symlink the dataset
                # root into the cache under the split name so downstream
                # path resolution (cache/mpiii/test/001/...) still works.
                split_dir = dataset_dir
            else:
                split_dir = dataset_dir / split_dirname
            if not split_dir.is_dir():
                print(f"  skip {dataset_name}/{split_dirname}: directory not found")
                continue

            # Symlink into the cache.
            link_path = cache_dataset / split_dirname
            if flat_layout:
                cache_dataset.mkdir(parents=True, exist_ok=True)
            create_symlink(split_dir, link_path)
            print(f"  link {link_path} -> {split_dir}")

            for session_id in discover_sessions(split_dir):
                session_dir = split_dir / session_id

                # Resolve roles: fixed tuple for NoXi/NoXiJ, auto-discovered for MPII.
                roles_cfg = cfg["roles"]
                if roles_cfg == "auto":
                    roles = discover_roles(session_dir)
                    if not roles:
                        print(f"  skip {dataset_name}/{split_dirname}/{session_id}: no roles discovered")
                        continue
                else:
                    roles = roles_cfg

                for role in roles:
                    # Use engagement annotation when available; test
                    # sessions may lack labels — an empty path signals
                    # "no supervision" to downstream preprocessing.
                    target_file = session_dir / f"{role}.{TARGET_SUFFIX}"
                    target_rel = (
                        f"{split_dirname}/{session_id}/{role}.{TARGET_SUFFIX}"
                        if target_file.exists()
                        else ""
                    )

                    manifest_rows.append(
                        {
                            "dataset": dataset_name,
                            "session_id": session_id,
                            "role": role,
                            "model_split": model_split,
                            "target_relative_path": target_rel,
                        }
                    )

                    # Discover available streams for this role.
                    for stream_name in KNOWN_STREAMS:
                        header = session_dir / f"{role}.{stream_name}.stream"
                        binary = session_dir / f"{role}.{stream_name}.stream~"
                        if not binary.exists():
                            continue
                        header_rel = f"{split_dirname}/{session_id}/{role}.{stream_name}.stream"
                        binary_rel = f"{split_dirname}/{session_id}/{role}.{stream_name}.stream~"
                        stream_rows.append(
                            {
                                "dataset": dataset_name,
                                "session_id": session_id,
                                "role": role,
                                "stream_name": stream_name,
                                "local_relative_path": header_rel,
                                "binary_local_relative_path": binary_rel,
                                "has_header": "yes" if header.exists() else "no",
                            }
                        )

    # Write manifests.
    manifest_fields = [
        "dataset",
        "session_id",
        "role",
        "model_split",
        "target_relative_path",
    ]
    stream_fields = [
        "dataset",
        "session_id",
        "role",
        "stream_name",
        "local_relative_path",
        "binary_local_relative_path",
        "has_header",
    ]

    manifest_path = args.out_dir / "model_raw_manifest_train_with_split.csv"
    streams_path = args.out_dir / "model_raw_manifest_streams_train.csv"

    write_csv(manifest_path, manifest_fields, manifest_rows)
    write_csv(streams_path, stream_fields, stream_rows)

    # Summary.
    datasets_found = {r["dataset"] for r in manifest_rows}
    splits_found = {r["model_split"] for r in manifest_rows}
    sessions_found = {(r["dataset"], r["session_id"]) for r in manifest_rows}
    streams_found = {r["stream_name"] for r in stream_rows}

    print(f"\n--- Summary ---")
    print(f"Datasets:   {sorted(datasets_found)}")
    print(f"Splits:     {sorted(splits_found)}")
    print(f"Sessions:   {len(sessions_found)}")
    print(f"Manifest rows (role-level): {len(manifest_rows)}")
    print(f"Stream rows:  {len(stream_rows)}")
    print(f"Streams found: {sorted(streams_found)}")
    print(f"\nManifest:  {manifest_path}")
    print(f"Streams:   {streams_path}")
    print(f"Cache:     {args.cache_root}")


if __name__ == "__main__":
    main()
