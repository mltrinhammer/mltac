#!/usr/bin/env python3
"""Verify cached CC ablation metrics bundled with this repo."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "runs"
MODALITY_SUMMARY = (
    ROOT
    / "artifacts"
    / "ablation_outputs"
    / "pinsoro_cc_submitted_checkpoint_modality_masks_3006"
    / "submitted_checkpoint_modality_mask_summary.csv"
)

PARTNER_ROWS = [
    ("CC task", "No partner", "task", RUNS / "pinsoro_cc_task_shared_none_shared_tcn_delta010_metadata_seed13", 0.330583),
    ("CC task", "Late linear partner", "task", RUNS / "pinsoro_cc_task_shared_linear_shared_tcn_delta010_metadata_seed13", 0.376920),
    ("CC task", "Late gated partner", "task", RUNS / "pinsoro_cc_task_submitted_late_gated_shared_tcn_delta010_metadata_seed13", 0.373723),
    ("CC social", "No partner", "social", RUNS / "pinsoro_cc_social_submitted_no_partner_head_adapters_delta010_metadata_seed13", 0.354842),
    ("CC social", "Late linear partner", "social", RUNS / "pinsoro_cc_headarch_head_adapters_delta010_metadata_seed13", 0.346729),
    ("CC social", "Late gated partner", "social", RUNS / "pinsoro_cc_both_headarch_head_adapters_logit_gated_scale0.1_delta010_metadata_both_seed13", 0.347857),
]

MODALITY_EXPECTED = {
    ("task", "atv"): 0.376920,
    ("task", "at"): 0.140607,
    ("task", "av"): 0.373855,
    ("task", "tv"): 0.382419,
    ("task", "a"): 0.089435,
    ("task", "t"): 0.160351,
    ("task", "v"): 0.380484,
    ("social", "atv"): 0.346729,
    ("social", "at"): 0.195056,
    ("social", "av"): 0.351916,
    ("social", "tv"): 0.314679,
    ("social", "a"): 0.164105,
    ("social", "t"): 0.100059,
    ("social", "v"): 0.290584,
}


def close(actual: float, expected: float, tol: float = 5e-6) -> bool:
    return abs(actual - expected) <= tol


def read_overall_kappa(run_dir: Path, head: str) -> float:
    metrics_path = run_dir / "metrics_overall.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") == "overall" and row.get("head") == head:
                return float(row["kappa"])
    raise RuntimeError(f"No overall/{head} row in {metrics_path}")


def verify_partner_table() -> None:
    print("Partner ablation table")
    for task, mode, head, run_dir, expected in PARTNER_ROWS:
        actual = read_overall_kappa(run_dir, head)
        status = "ok" if close(actual, expected) else "FAIL"
        print(f"  {status:4s} {task:9s} {mode:20s} actual={actual:.6f} expected={expected:.6f}")
        if status != "ok":
            raise SystemExit(1)


def verify_modality_table() -> None:
    if not MODALITY_SUMMARY.exists():
        raise FileNotFoundError(MODALITY_SUMMARY)
    seen: dict[tuple[str, str], float] = {}
    with MODALITY_SUMMARY.open(newline="") as handle:
        for row in csv.DictReader(handle):
            seen[(row["head"], row["modality_tag"])] = float(row["raw_kappa"])

    print("Submitted-checkpoint modality mask table")
    for key, expected in MODALITY_EXPECTED.items():
        actual = seen[key]
        status = "ok" if close(actual, expected) else "FAIL"
        print(f"  {status:4s} {key[0]:6s} {key[1]:3s} actual={actual:.6f} expected={expected:.6f}")
        if status != "ok":
            raise SystemExit(1)


def main() -> None:
    verify_partner_table()
    verify_modality_table()


if __name__ == "__main__":
    main()
