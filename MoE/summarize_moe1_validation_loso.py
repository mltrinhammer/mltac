"""Summarize MoE1 validation-session LOSO results and best/worst model paths."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent / "experiments" / "moe1_validation_loso_metadata_head_two_head_hmm"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize MoE1 LOSO fold results.")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--copy-best-worst", action="store_true", help="Copy best/worst fold model_best.pt files into a retained_models folder.")
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def model_paths(root: Path, domain: str, heldout: str) -> list[dict[str, object]]:
    fold = root / domain.lower() / f"heldout_{heldout}" / "experts"
    rows = []
    for feature in FEATURES:
        run = fold / f"{domain.lower()}_{feature}_dyadic_tcn_k11_seed13"
        rows.append({
            "domain": domain,
            "heldout_session": heldout,
            "feature": feature,
            "model_best": str(run / "model_best.pt"),
            "model_last": str(run / "model_last.pt"),
            "config": str(run / "config.json"),
            "training_log": str(run / "training_log.csv"),
            "exists": (run / "model_best.pt").exists(),
        })
    return rows


def copy_retained(root: Path, selected: list[dict[str, object]]) -> None:
    out = root / "retained_best_worst_model_weights"
    for row in selected:
        label = str(row["selection"])
        domain = str(row["domain"])
        heldout = str(row["heldout_session"])
        for m in model_paths(root, domain, heldout):
            src_model = Path(str(m["model_best"]))
            src_config = Path(str(m["config"]))
            src_log = Path(str(m["training_log"]))
            dst = out / f"{domain.lower()}_{label}_heldout_{heldout}" / str(m["feature"])
            dst.mkdir(parents=True, exist_ok=True)
            if src_model.exists():
                shutil.copy2(src_model, dst / "model_best.pt")
            if src_config.exists():
                shutil.copy2(src_config, dst / "config.json")
            if src_log.exists():
                shutil.copy2(src_log, dst / "training_log.csv")


def main() -> None:
    args = parse_args()
    root = args.root
    all_fold_rows = []
    model_manifest = []
    selected = []
    for domain in ("CR", "CC"):
        rows = read_rows(root / f"{domain.lower()}_fold_hmm_results.csv")
        all_fold_rows.extend(rows)
        candidates = [
            row for row in rows
            if row.get("mode") == "two_head_hmm" and row.get("head") == "mean"
        ]
        if not candidates:
            continue
        for row in candidates:
            row["kappa_float"] = float(row["kappa"])
        best = max(candidates, key=lambda r: r["kappa_float"])
        worst = min(candidates, key=lambda r: r["kappa_float"])
        for label, row in (("best", best), ("worst", worst)):
            selected_row = {
                "selection": label,
                "domain": domain,
                "heldout_session": row["heldout_session"],
                "mode": row["mode"],
                "param": row["param"],
                "mean_kappa": row["kappa"],
            }
            selected.append(selected_row)
            for m in model_paths(root, domain, row["heldout_session"]):
                model_manifest.append(selected_row | m)
    write_csv(root / "all_leave_one_out_scores.csv", all_fold_rows)
    write_csv(root / "best_worst_folds.csv", selected)
    write_csv(root / "best_worst_model_weight_manifest.csv", model_manifest)
    if args.copy_best_worst and selected:
        copy_retained(root, selected)
    md = ["# MoE1 Validation LOSO Summary", "", f"Root: `{root}`", "", "## Best/Worst Held-Out Folds", ""]
    if selected:
        md.append("| Selection | Domain | Held-out session | Mean kappa |")
        md.append("| --- | --- | --- | ---: |")
        for row in selected:
            md.append(f"| {row['selection']} | {row['domain']} | {row['heldout_session']} | {float(row['mean_kappa']):.4f} |")
    else:
        md.append("No completed `two_head_hmm` mean rows found yet.")
    md.extend(["", "Detailed files:", "", "- `all_leave_one_out_scores.csv`", "- `best_worst_folds.csv`", "- `best_worst_model_weight_manifest.csv`"])
    if args.copy_best_worst:
        md.append("- `retained_best_worst_model_weights/`")
    (root / "best_worst_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(root / "best_worst_summary.md")


if __name__ == "__main__":
    main()
