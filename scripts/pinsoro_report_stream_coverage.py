"""Summarize PinSoRo stream availability by split and domain."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.io import read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "outputs" / "pinsoro" / "raw_stream_manifest.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "pinsoro" / "validation" / "stream_coverage.csv")
    args = parser.parse_args()
    counts = Counter((row["source_split"], row["domain"], row["stream_name"]) for row in read_csv(args.manifest))
    rows = [
        {"source_split": key[0], "domain": key[1], "stream_name": key[2], "role_stream_count": count}
        for key, count in sorted(counts.items())
    ]
    if not rows:
        raise RuntimeError(f"No stream rows in {args.manifest}")
    write_csv(args.output, list(rows[0].keys()), rows)
    print(f"Coverage rows: {len(rows)}; output: {args.output}")


if __name__ == "__main__":
    main()
