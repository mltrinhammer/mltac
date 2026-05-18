from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


DATA_ROOT = Path("X:/")
PROJECT_ROOT = Path(r"C:/Users/anec/OneDrive - Syddansk Universitet/Projects/PinSoRo")
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def keep_row(value: str, scenario: str) -> bool:
    value_l = value.strip().lower()
    is_blank = value_l == ""
    is_nan = value_l == "nan"
    if scenario == "all":
        return True
    if scenario == "drop_blank":
        return not is_blank
    if scenario == "drop_nan":
        return not is_nan
    if scenario == "drop_blank_and_nan":
        return (not is_blank) and (not is_nan)
    raise ValueError(f"Unknown scenario: {scenario}")


def class_weights(counts: Counter[str]) -> dict[str, float]:
    labels = [label for label, count in counts.items() if count > 0]
    n = sum(counts[l] for l in labels)
    k = len(labels)
    if n == 0 or k == 0:
        return {}
    return {label: n / (k * counts[label]) for label in labels}


def read_rows(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="drop_blank_and_nan", choices=["all", "drop_blank", "drop_nan", "drop_blank_and_nan"])
    parser.add_argument("--color", default="purple", choices=["purple", "yellow"])
    parser.add_argument("--splits", nargs="+", default=["train-cc", "train-cr"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_rows = []
    for split in args.splits:
        split_dir = DATA_ROOT / split
        sessions = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
        for task in ("task_engagement", "social_engagement"):
            counts: Counter[str] = Counter()
            used_rows = 0
            for session in sessions:
                file_path = session / f"{args.color}.{task}.annotation.csv"
                if not file_path.exists():
                    continue
                rows = read_rows(file_path)
                kept = [r for r in rows if keep_row(r, args.scenario)]
                counts.update(kept)
                used_rows += len(kept)

            weights = class_weights(counts)
            for label, count in sorted(counts.items()):
                out_rows.append(
                    {
                        "scenario": args.scenario,
                        "split": split,
                        "color": args.color,
                        "task": task,
                        "label": label,
                        "count": str(count),
                        "weight": f"{weights.get(label, 0.0):.8f}",
                        "used_rows": str(used_rows),
                    }
                )

    out_path = OUTPUT_DIR / f"training_prep_{args.color}_{args.scenario}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario", "split", "color", "task", "label", "count", "weight", "used_rows"],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote training prep export: {out_path}")


if __name__ == "__main__":
    main()

