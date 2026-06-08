"""Build an uncompressed memory-mapped cache for PinSoRo training tensors."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WINDOWS_DIR = PROJECT_ROOT / "outputs/pinsoro/windows"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "processed/pinsoro_mmap"
TENSOR_ARRAY_KEYS = ("x", "task_y", "task_mask", "social_y", "social_mask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the PinSoRo uncompressed mmap training cache."
    )
    parser.add_argument("--manifests", nargs="+", type=Path)
    parser.add_argument("--features", nargs="+")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def source_paths(args: argparse.Namespace) -> list[Path]:
    manifests = args.manifests or sorted(
        DEFAULT_WINDOWS_DIR.glob("*_w300_s75_dyadic.csv")
    )
    if args.features:
        requested = set(args.features)
        manifests = [
            path
            for path in manifests
            if any(path.name.startswith(f"{name}_") for name in requested)
        ]
    if not manifests:
        raise FileNotFoundError("No PinSoRo dyadic window manifests found.")
    paths: set[Path] = set()
    for manifest in manifests:
        with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                for field in (
                    "purple_tensor_relative_path",
                    "yellow_tensor_relative_path",
                ):
                    path = Path(row[field])
                    paths.add(path if path.is_absolute() else PROJECT_ROOT / path)
    return sorted(paths)


def cache_dir(cache_root: Path, source: Path) -> Path:
    relative = source.resolve().relative_to(PROJECT_ROOT.resolve())
    return cache_root / relative.parent / relative.stem


def source_signature(source: Path) -> dict[str, int]:
    stat = source.stat()
    return {"source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns}


def cache_is_current(source: Path, target: Path) -> bool:
    marker = target / ".complete"
    if not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata == source_signature(source) and all(
        (target / f"{key}.npy").is_file() for key in TENSOR_ARRAY_KEYS
    )


def atomic_save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def build_one(source: Path, cache_root: Path, force: bool) -> tuple[str, int]:
    target = cache_dir(cache_root, source)
    if not force and cache_is_current(source, target):
        return "skipped", sum(
            (target / f"{key}.npy").stat().st_size for key in TENSOR_ARRAY_KEYS
        )
    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".complete"
    marker.unlink(missing_ok=True)
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in TENSOR_ARRAY_KEYS}
    for key, array in arrays.items():
        atomic_save(target / f"{key}.npy", array)
    metadata = source_signature(source)
    temporary_marker = marker.with_suffix(".tmp")
    temporary_marker.write_text(json.dumps(metadata), encoding="utf-8")
    temporary_marker.replace(marker)
    return "built", sum(
        (target / f"{key}.npy").stat().st_size for key in TENSOR_ARRAY_KEYS
    )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be at least 1.")
    sources = source_paths(args)
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} source tensors; first: {missing[0]}"
        )
    args.cache_root.mkdir(parents=True, exist_ok=True)
    is_full_default = args.manifests is None and args.features is None
    global_marker = args.cache_root / ".complete"
    if is_full_default:
        global_marker.unlink(missing_ok=True)
    started = time.perf_counter()
    built = skipped = total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(build_one, source, args.cache_root, args.force): source
            for source in sources
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            status, size = future.result()
            built += status == "built"
            skipped += status == "skipped"
            total_bytes += size
            if completed % 25 == 0 or completed == len(sources):
                elapsed = time.perf_counter() - started
                print(
                    f"completed={completed}/{len(sources)} built={built} skipped={skipped} "
                    f"cache_gib={total_bytes / 2**30:.2f} elapsed_minutes={elapsed / 60:.1f}",
                    flush=True,
                )
    if is_full_default:
        global_marker.write_text(
            json.dumps(
                {
                    "source_count": len(sources),
                    "cache_bytes": total_bytes,
                    "created_at_unix": int(time.time()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"Cache root: {args.cache_root}", flush=True)
    if not is_full_default:
        print(
            "Partial cache built; global .complete marker was not written.", flush=True
        )


if __name__ == "__main__":
    main()
