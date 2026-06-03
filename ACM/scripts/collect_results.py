"""Collect experiment results and format a comparison table.

Scans ACM/outputs/experiments/ for completed runs, reads their metric CSVs,
and prints a markdown table comparable to the MultiMediate26 organizer baseline.

All metrics are on the validation set (test labels held by organizers).

Usage:
    python ACM/scripts/collect_results.py
    python ACM/scripts/collect_results.py --experiments-dir ACM/outputs/experiments
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model type display order (matches the handoff model ladder).
MODEL_ORDER = [
    "turns_simple_tcn",
    "turns_dyadic_shared",
    "turns_attention",
]

MODEL_LABELS = {
    "turns_simple_tcn": "Turn-Level Simple TCN",
    "turns_dyadic_shared": "Turn-Level Dyadic (shared head)",
    "turns_attention": "Turn-Level Attention TCN",
}

# Feature set display order (voice first, then text, then video — matching organizer).
FEATURE_ORDER = [
    "audio_egemaps",
    "audio_w2vbert2",
    "text_xlm_roberta",
    "visual_openface",
    "visual_openpose",
    "visual_clip",
    "visual_dino",
    "visual_swin",
    "visual_videomae",
]

FEATURE_LABELS = {
    "audio_egemaps": "eGeMAPS v2",
    "audio_w2vbert2": "w2vBERT2",
    "text_xlm_roberta": "XLM-RoBERTa",
    "visual_openface": "OpenFace 2+3",
    "visual_openpose": "OpenPose",
    "visual_clip": "CLIP",
    "visual_dino": "DINOv2",
    "visual_swin": "SwinTransformer",
    "visual_videomae": "VideoMAE",
}

# Organizer MLP baseline CCC (combined across test sets) for reference.
ORGANIZER_BASELINE = {
    "audio_egemaps": 0.4529,
    "audio_w2vbert2": 0.2222,
    "text_xlm_roberta": 0.0793,
    "visual_openface": 0.1433,
    "visual_openpose": 0.0505,
    "visual_clip": 0.1474,
    "visual_dino": 0.1285,
    "visual_swin": 0.1463,
    "visual_videomae": 0.0955,
}


def read_metric_csv(path: Path) -> dict[str, str]:
    """Read a single-row metric CSV and return its fields."""
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def read_role_metrics(path: Path) -> dict[str, float]:
    """Read metrics_by_role.csv and return CCC per role."""
    if not path.exists():
        return {}
    result = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            role = row.get("role", "")
            ccc = row.get("ccc", "")
            if role and ccc:
                try:
                    result[role] = float(ccc)
                except ValueError:
                    pass
    return result


def parse_run_name(run_name: str) -> tuple[str, str] | None:
    """Split a run name into (feature_set, model_type)."""
    # Match the longest suffix first so nested names like
    # "*_turns_partner_lag" are not misclassified as "*_partner_lag".
    for model in sorted(MODEL_ORDER, key=len, reverse=True):
        if run_name.endswith(f"_{model}"):
            feature_set = run_name[: -(len(model) + 1)]
            return feature_set, model
    return None


def scan_experiments(experiments_dir: Path) -> dict[tuple[str, str], dict]:
    """Scan experiment directories and collect metrics."""
    results: dict[tuple[str, str], dict] = {}
    if not experiments_dir.is_dir():
        return results

    for run_dir in sorted(experiments_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        best_model = run_dir / "model_best.pt"
        if not best_model.exists():
            continue

        parsed = parse_run_name(run_dir.name)
        if parsed is None:
            continue
        feature_set, model_type = parsed

        overall = read_metric_csv(run_dir / "metrics_overall.csv")
        role_metrics = read_role_metrics(run_dir / "metrics_by_role.csv")

        results[(feature_set, model_type)] = {
            "run_name": run_dir.name,
            "ccc": float(overall.get("ccc", "nan")),
            "mae": float(overall.get("mae", "nan")),
            "rmse": float(overall.get("rmse", "nan")),
            "pearson": float(overall.get("pearson", "nan")),
            "novice_ccc": role_metrics.get("novice", float("nan")),
            "expert_ccc": role_metrics.get("expert", float("nan")),
        }

    return results


def fmt(val: float, best: float | None = None, precision: int = 4) -> str:
    """Format a metric value, bolding the best in its column."""
    if val != val:  # NaN
        return "-"
    s = f"{val:.{precision}f}"
    if best is not None and abs(val - best) < 1e-8:
        return f"**{s}**"
    return s


def print_table_by_model(results: dict[tuple[str, str], dict]) -> None:
    """Print one table per model type: features as rows, metrics as columns."""
    print("\n## Results by Model Type (Validation CCC)\n")

    for model in MODEL_ORDER:
        model_results = {fs: r for (fs, mt), r in results.items() if mt == model}
        if not model_results:
            continue

        print(f"### {MODEL_LABELS.get(model, model)}\n")
        print("| Feature Set | Val CCC | Novice | Expert | MAE | Organizer MLP (test) |")
        print("|---|---|---|---|---|---|")

        # Find best CCC in this model type for bolding.
        cccs = [r["ccc"] for r in model_results.values() if r["ccc"] == r["ccc"]]
        best_ccc = max(cccs) if cccs else None

        for fs in FEATURE_ORDER:
            if fs not in model_results:
                continue
            r = model_results[fs]
            org = ORGANIZER_BASELINE.get(fs, float("nan"))
            print(
                f"| {FEATURE_LABELS.get(fs, fs)} "
                f"| {fmt(r['ccc'], best_ccc)} "
                f"| {fmt(r['novice_ccc'])} "
                f"| {fmt(r['expert_ccc'])} "
                f"| {fmt(r['mae'])} "
                f"| {fmt(org)} |"
            )
        print()


def print_table_by_feature(results: dict[tuple[str, str], dict]) -> None:
    """Print one summary table: features as rows, model types as columns."""
    print("\n## Summary: Val CCC by Feature Set x Model Type\n")

    # Only include models that have at least one result.
    active_models = [m for m in MODEL_ORDER if any(mt == m for (_, mt) in results)]
    if not active_models:
        print("No completed experiments found.\n")
        return

    header = "| Feature Set | " + " | ".join(MODEL_LABELS.get(m, m) for m in active_models) + " | Organizer MLP (test) |"
    sep = "|---" * (len(active_models) + 2) + "|"
    print(header)
    print(sep)

    for fs in FEATURE_ORDER:
        row_vals = []
        for m in active_models:
            r = results.get((fs, m))
            row_vals.append(r["ccc"] if r else float("nan"))

        valid = [v for v in row_vals if v == v]
        best = max(valid) if valid else None
        org = ORGANIZER_BASELINE.get(fs, float("nan"))

        cells = " | ".join(fmt(v, best) for v in row_vals)
        print(f"| {FEATURE_LABELS.get(fs, fs)} | {cells} | {fmt(org)} |")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and format ACM experiment results.")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "experiments",
    )
    args = parser.parse_args()

    results = scan_experiments(args.experiments_dir)
    if not results:
        print(f"No completed experiments found in {args.experiments_dir}")
        sys.exit(1)

    n = len(results)
    features = sorted({fs for fs, _ in results})
    models = sorted({mt for _, mt in results})
    print(f"Found {n} completed experiments across {len(features)} feature sets and {len(models)} model types.\n")

    print_table_by_feature(results)
    print_table_by_model(results)


if __name__ == "__main__":
    main()
