"""Print speech-turn length distribution from a turn manifest CSV.

Usage:
    python ACM/scripts/turn_length_distribution.py --manifest ACM/outputs/manifests/model_processed_manifest_audio_egemaps_raw_turns.csv
    python ACM/scripts/turn_length_distribution.py --manifest ACM/outputs/manifests/model_processed_manifest_audio_egemaps_raw_turns.csv --split train_internal
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_turns(manifest: Path, split: str | None = None) -> list[dict]:
    """Read turn manifest and deduplicate by (dataset, session_id, turn_idx)."""
    seen: set[tuple[str, str, str]] = set()
    turns: list[dict] = []
    with manifest.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if split and row.get("model_split") != split:
                continue
            key = (row["dataset"], row["session_id"], row["turn_idx"])
            if key in seen:
                continue
            seen.add(key)
            turns.append(row)
    return turns


def main() -> None:
    parser = argparse.ArgumentParser(description="Speech-turn length distribution.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=str, default=None,
                        help="Filter to a specific model_split (e.g. train_internal).")
    args = parser.parse_args()

    turns = read_turns(args.manifest, args.split)
    if not turns:
        print("No turns found.")
        return

    lengths_frames = np.array([int(t["turn_len"]) for t in turns])
    lengths_sec = lengths_frames / 25.0
    speakers = [t["speaker"] for t in turns]

    print(f"Manifest: {args.manifest.name}")
    if args.split:
        print(f"Split:    {args.split}")
    print(f"Turns:    {len(turns)}")
    print()

    # Overall statistics
    print("=== Overall (seconds) ===")
    print(f"  Mean:   {np.mean(lengths_sec):.2f}")
    print(f"  Median: {np.median(lengths_sec):.2f}")
    print(f"  Std:    {np.std(lengths_sec):.2f}")
    print(f"  Min:    {np.min(lengths_sec):.2f}")
    print(f"  Max:    {np.max(lengths_sec):.2f}")
    for p in [5, 25, 75, 95]:
        print(f"  P{p:02d}:    {np.percentile(lengths_sec, p):.2f}")
    print()

    # Per-speaker breakdown
    for spk in sorted(set(speakers)):
        mask = np.array([s == spk for s in speakers])
        spk_sec = lengths_sec[mask]
        print(f"=== {spk} ({len(spk_sec)} turns) ===")
        print(f"  Mean:   {np.mean(spk_sec):.2f}")
        print(f"  Median: {np.median(spk_sec):.2f}")
        print(f"  Std:    {np.std(spk_sec):.2f}")
        print()

    # Histogram (text-based)
    edges = [0, 1, 2, 3, 5, 10, 20, 30, 60, 120, float("inf")]
    labels = ["<1s", "1-2s", "2-3s", "3-5s", "5-10s", "10-20s", "20-30s", "30-60s", "60-120s", ">120s"]
    counts = np.zeros(len(labels), dtype=int)
    for sec in lengths_sec:
        for i in range(len(edges) - 1):
            if edges[i] <= sec < edges[i + 1]:
                counts[i] += 1
                break

    bar_max = 50
    max_count = max(counts) if max(counts) > 0 else 1
    print("=== Duration histogram ===")
    for label, count in zip(labels, counts):
        bar = "#" * int(count / max_count * bar_max)
        pct = 100.0 * count / len(turns)
        print(f"  {label:>7s} | {bar:<{bar_max}s} {count:5d} ({pct:5.1f}%)")
    print()


if __name__ == "__main__":
    main()
