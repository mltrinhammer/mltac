"""Convert long-form PinSoRo test predictions into organizer submission files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEAD_OUTPUT_NAMES = {"task": "task_engagement", "social": "social_engagement"}
CLASS_LABELS = {
    "task": ("goaloriented", "aimless", "adultseeking", "noplay"),
    "social": ("solitary", "onlooker", "parallel", "associative", "cooperative"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export organizer-format PinSoRo prediction files."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[tuple[str, str, str, str], list[tuple[int, int]]] = defaultdict(list)
    with args.predictions.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["domain"], row["session_id"], row["role"], row["head"])
            grouped[key].append((int(row["frame_idx"]), int(row["y_pred"])))
    written = 0
    for (domain, session_id, role, head), values in sorted(grouped.items()):
        if domain == "CR" and role == "yellow":
            continue
        values.sort()
        indices = [frame_idx for frame_idx, _ in values]
        if indices != list(range(len(indices))):
            raise RuntimeError(
                f"Predictions are not contiguous from frame 0 for "
                f"{domain}/{session_id}/{role}/{head}."
            )
        labels = CLASS_LABELS[head]
        session_dir = args.output_dir / f"pinsoro-{domain.lower()}" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / f"{role}.{HEAD_OUTPUT_NAMES[head]}.prediction.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            for _, prediction in values:
                if not 0 <= prediction < len(labels):
                    raise ValueError(f"Invalid {head} class id: {prediction}")
                handle.write(f"{labels[prediction]}\n")
        written += 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / ".complete").write_text(f"files={written}\n", encoding="utf-8")
    print(f"Wrote {written} PinSoRo prediction files to {args.output_dir}")


if __name__ == "__main__":
    main()
