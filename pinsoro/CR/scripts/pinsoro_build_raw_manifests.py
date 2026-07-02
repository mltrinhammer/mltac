"""Extract PinSoRo split archives and build raw ACM manifests."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.io import write_csv

ARCHIVES = ("train-cc.zip", "train-cr.zip", "val.zip", "test.zip")
ROLES = ("purple", "yellow")
KNOWN_STREAMS = (
    "audio.egemapsv2", "audio.w2vbert2_embeddings", "audio.xlm_roberta_embeddings",
    "clip", "dino", "openface2", "openface3", "openpose", "swin", "videomae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PinSoRo archives and build raw ACM manifests.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "cache" / "pinsoro")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "pinsoro")
    parser.add_argument("--skip-extract", action="store_true")
    return parser.parse_args()


def split_metadata(split_name: str) -> tuple[str, str]:
    prefix, domain = split_name.split("-", maxsplit=1)
    return domain.upper(), {"train": "train_internal", "val": "val_internal", "test": "test_internal"}[prefix]


def archive_roots(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return sorted({name.split("/", maxsplit=1)[0] for name in archive.namelist() if "/" in name})


def extract_archive(path: Path, cache_root: Path) -> list[str]:
    roots = archive_roots(path)
    marker = cache_root / f".{path.stem}.extracted"
    if not marker.exists():
        cache_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(cache_root)
        marker.write_text(json.dumps({"archive": path.name, "roots": roots}, indent=2), encoding="utf-8")
    return roots


def build_rows(split_name: str, split_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    domain, model_split = split_metadata(split_name)
    raw_rows: list[dict[str, object]] = []
    stream_rows: list[dict[str, object]] = []
    sessions = sorted(path for path in split_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
    for session_dir in sessions:
        for role in ROLES:
            task_path = session_dir / f"{role}.task_engagement.annotation.csv"
            social_path = session_dir / f"{role}.social_engagement.annotation.csv"
            supervised = model_split != "test_internal" and not (domain == "CR" and role == "yellow")
            raw_rows.append({
                "dataset": "pinsoro", "domain": domain, "source_split": split_name,
                "model_split": model_split, "session_id": session_dir.name, "role": role,
                "supervised": "yes" if supervised else "no",
                "task_target_relative_path": str(task_path.relative_to(split_dir.parent)).replace("\\", "/") if task_path.exists() else "",
                "social_target_relative_path": str(social_path.relative_to(split_dir.parent)).replace("\\", "/") if social_path.exists() else "",
            })
            for stream_name in KNOWN_STREAMS:
                header = session_dir / f"{role}.{stream_name}.stream"
                binary = session_dir / f"{role}.{stream_name}.stream~"
                if binary.exists():
                    stream_rows.append({
                        "dataset": "pinsoro", "domain": domain, "source_split": split_name,
                        "model_split": model_split, "session_id": session_dir.name, "role": role,
                        "stream_name": stream_name,
                        "local_relative_path": str(header.relative_to(split_dir.parent)).replace("\\", "/"),
                        "binary_local_relative_path": str(binary.relative_to(split_dir.parent)).replace("\\", "/"),
                        "has_header": "yes" if header.exists() else "no",
                    })
    return raw_rows, stream_rows


def main() -> None:
    args = parse_args()
    data_root, cache_root, out_dir = args.data_root.resolve(), args.cache_root.resolve(), args.out_dir.resolve()
    roots: set[str] = set()
    for name in ARCHIVES:
        path = data_root / name
        if path.exists():
            roots.update(archive_roots(path) if args.skip_extract else extract_archive(path, cache_root))
        else:
            print(f"WARNING: missing archive {path}")
    raw_rows: list[dict[str, object]] = []
    stream_rows: list[dict[str, object]] = []
    for split_name in sorted(roots):
        split_dir = cache_root / split_name
        if split_dir.is_dir():
            split_raw, split_streams = build_rows(split_name, split_dir)
            raw_rows.extend(split_raw)
            stream_rows.extend(split_streams)
    if not raw_rows:
        raise RuntimeError("No PinSoRo role rows were discovered.")
    write_csv(out_dir / "raw_manifest.csv", list(raw_rows[0].keys()), raw_rows)
    write_csv(out_dir / "raw_stream_manifest.csv", list(stream_rows[0].keys()), stream_rows)
    print(f"Splits: {sorted(roots)}; role rows: {len(raw_rows)}; streams: {len(stream_rows)}")
    print(f"Cache: {cache_root}\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
