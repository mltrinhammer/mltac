from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np


DATASET_CACHE_DIR = {
    "noxi": "noxi_a",
    "noxij": "noxi_b",
    "mpiigroupinteraction": "mpiii",
}

# The cached rclone layout uses dataset-specific folder names (`noxi_a`,
# `noxi_b`), while manifests use logical names (`noxi`, `noxij`). Keep the
# mapping here so path construction is not repeated in every script.

def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows while normalizing BOM-polluted column names."""

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [{str(k).replace("\ufeff", "").strip(): str(v).strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    """Write a small metadata table, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dataset_root(cache_root: Path, dataset: str) -> Path:
    return cache_root / DATASET_CACHE_DIR[dataset]


def local_cache_path(cache_root: Path, dataset: str, relative_path: str) -> Path:
    """Resolve a manifest-relative path inside the local rclone cache layout."""

    return dataset_root(cache_root, dataset) / relative_path


def read_target(path: Path) -> np.ndarray:
    """Read engagement annotations, preserving blanks/unparseable values as NaN."""

    # Targets are stored as one value per line. Empty or malformed rows become
    # NaN here; downstream tensor preparation writes a parallel target_mask so
    # losses and metrics can ignore missing target frames.
    vals: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                vals.append(np.nan)
                continue
            try:
                vals.append(float(text))
            except ValueError:
                vals.append(np.nan)
    return np.asarray(vals, dtype=np.float32)


def parse_stream_header(path: Path) -> tuple[float, int]:
    """Extract sample rate and feature dimension from an SSI .stream header."""

    text = path.read_text(encoding="utf-8", errors="ignore")
    sr_match = re.search(r'sr="([^"]+)"', text)
    dim_match = re.search(r'dim="([^"]+)"', text)
    if sr_match is None or dim_match is None:
        raise ValueError(f"Could not parse sr/dim from {path}")
    return float(sr_match.group(1)), int(float(dim_match.group(1)))


def read_stream_matrix(header_path: Path, binary_path: Path | None = None) -> tuple[np.ndarray, float, int]:
    """Read an SSI stream pair as [frames, features] float32 data.

    The text header stores metadata; the sibling file with a trailing "~" stores
    raw float32 values in row-major frame order.
    """

    # SSI stores metadata in the XML-like header and the actual matrix in a
    # sibling raw binary file. The binary contains flat float32 values, so the
    # parsed feature dimension is needed to recover [frames, features].
    sr, dim = parse_stream_header(header_path)
    binary_path = binary_path or Path(str(header_path) + "~")
    raw = np.fromfile(binary_path, dtype=np.float32)
    n_frames = raw.size // dim
    if n_frames == 0:
        return np.empty((0, dim), dtype=np.float32), sr, dim
    return raw[: n_frames * dim].reshape(n_frames, dim), sr, dim
