"""Collect organizer-compatible PinSoRo feature-ablation results."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect PinSoRo TCN result summaries.")
    p.add_argument(
        "--experiments-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/pinsoro/experiments",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/pinsoro/results_summary.csv",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs/pinsoro/results_report.md",
    )
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def value(row: dict[str, str], key: str) -> float:
    text = row.get(key, "")
    return float(text) if text not in {"", "nan"} else float("nan")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                f"{row.get(c, ''):.4f}"
                if isinstance(row.get(c), float)
                else str(row.get(c, ""))
                for c in columns
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = []
    grouped = {"domain": [], "role": [], "domain_role": []}
    for config_path in sorted(args.experiments_dir.glob("*/config.json")):
        run_dir = config_path.parent
        domain_rows = read_rows(run_dir / "metrics_by_domain.csv")
        if not domain_rows:
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        domain = {(r["domain"], r["head"]): r for r in domain_rows}
        overall = {r["head"]: r for r in read_rows(run_dir / "metrics_overall.csv")}
        kappas = [
            value(domain.get((d, h), {}), "kappa")
            for d in ("CC", "CR")
            for h in ("task", "social")
        ]
        finite = [x for x in kappas if np.isfinite(x)]
        rows.append(
            {
                "run_name": run_dir.name,
                "feature_set": config["feature_set"],
                "model": config["model"],
                "seed": config["seed"],
                "organizer_score": float(np.mean(finite)) if finite else float("nan"),
                "cc_task_kappa": kappas[0],
                "cc_social_kappa": kappas[1],
                "cr_task_kappa": kappas[2],
                "cr_social_kappa": kappas[3],
                "task_macro_f1": value(overall.get("task", {}), "macro_f1"),
                "task_weighted_f1": value(overall.get("task", {}), "weighted_f1"),
                "social_macro_f1": value(overall.get("social", {}), "macro_f1"),
                "social_weighted_f1": value(overall.get("social", {}), "weighted_f1"),
            }
        )
        for name, filename in (
            ("domain", "metrics_by_domain.csv"),
            ("role", "metrics_by_role.csv"),
            ("domain_role", "metrics_by_domain_role.csv"),
        ):
            for metric in read_rows(run_dir / filename):
                grouped[name].append(
                    {
                        "run_name": run_dir.name,
                        "feature_set": config["feature_set"],
                        "model": config["model"],
                        "seed": config["seed"],
                        **metric,
                    }
                )
    if not rows:
        raise RuntimeError(
            f"No completed PinSoRo runs found under {args.experiments_dir}"
        )
    rows = sorted(rows, key=lambda r: float(r["organizer_score"]), reverse=True)
    write_rows(args.output, rows)
    for name, items in grouped.items():
        write_rows(args.output.with_name(f"results_by_{name}.csv"), items)
    best_models = {}
    best_features = {}
    for row in rows:
        best_models.setdefault(str(row["feature_set"]), row)
        best_features.setdefault(str(row["model"]), row)
    cols = [
        "feature_set",
        "model",
        "organizer_score",
        "cc_task_kappa",
        "cc_social_kappa",
        "cr_task_kappa",
        "cr_social_kappa",
    ]
    report = [
        "# PinSoRo Seed-13 Results",
        "",
        f"Completed runs: {len(rows)}",
        "",
        "Primary score: arithmetic mean of CC-task, CC-social, CR-task, and CR-social Cohen's kappa.",
        "",
        "## Overall Ranking",
        "",
        markdown_table(rows, cols),
        "",
        "## Best Architecture Per Feature",
        "",
        markdown_table(list(best_models.values()), cols),
        "",
        "## Best Feature Per Architecture",
        "",
        markdown_table(list(best_features.values()), cols),
        "",
        "Detailed breakdowns: `results_by_domain.csv`, `results_by_role.csv`, and `results_by_domain_role.csv`.",
        "",
    ]
    args.report.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {len(rows)} runs to {args.output}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
