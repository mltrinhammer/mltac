"""Split PinSoRo window manifests into CC-only and CR-only CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.io import read_csv, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write one PinSoRo window manifest per requested domain."
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--domains", nargs="+", default=("CC", "CR"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_manifest)
    if not rows:
        raise RuntimeError(f"No rows in {args.input_manifest}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for domain in args.domains:
        selected = [row for row in rows if row["domain"] == domain]
        if not selected:
            raise RuntimeError(f"No rows for domain {domain} in {args.input_manifest}")
        output = args.out_dir / f"{args.prefix}_{domain.lower()}.csv"
        write_csv(output, fieldnames, selected)
        print(f"{domain}: rows={len(selected)} manifest={output}", flush=True)


if __name__ == "__main__":
    main()
