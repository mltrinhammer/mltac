"""Collect PinSoRo feature-ablation results from completed runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PinSoRo TCN result summaries.")
    parser.add_argument("--experiments-dir", type=Path, default=PROJECT_ROOT / "outputs" / "pinsoro" / "experiments")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "pinsoro" / "results_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for config_path in sorted(args.experiments_dir.glob("*/config.json")):
        run_dir = config_path.parent
        metrics_path = run_dir / "metrics_overall.csv"
        if not metrics_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with metrics_path.open("r", newline="", encoding="utf-8") as handle:
            metrics = {row["head"]: row for row in csv.DictReader(handle)}
        task = metrics.get("task", {})
        social = metrics.get("social", {})
        kappas = [float(row["kappa"]) for row in (task, social) if row and row["kappa"] not in {"", "nan"}]
        rows.append(
            {
                "run_name": run_dir.name,
                "feature_set": config["feature_set"],
                "model": config["model"],
                "seed": config["seed"],
                "mean_kappa": sum(kappas) / len(kappas) if kappas else float("nan"),
                "task_kappa": task.get("kappa", ""),
                "task_macro_f1": task.get("macro_f1", ""),
                "task_accuracy": task.get("accuracy", ""),
                "social_kappa": social.get("kappa", ""),
                "social_macro_f1": social.get("macro_f1", ""),
                "social_accuracy": social.get("accuracy", ""),
            }
        )
    if not rows:
        raise RuntimeError(f"No completed PinSoRo runs found under {args.experiments_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: float(row["mean_kappa"]), reverse=True))
    print(f"Wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
