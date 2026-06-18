"""Analyze confidence and calibration from PinSoRo prediction score exports."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "model improvement/test 5 receptive field ablation/results/"
    "w2400_s1200_l5_k11_causal/deep_error_analysis/predictions"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "model improvement/test 5 receptive field ablation/results/"
    "w2400_s1200_l5_k11_causal/deep_error_analysis/probability_analysis"
)
CLASS_COUNTS = {"task": 4, "social": 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PinSoRo prediction scores.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ece-bins", type=int, default=10)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def empty_metrics(bins: int) -> dict[str, object]:
    return {
        "n": 0,
        "correct": 0,
        "confidence_sum": 0.0,
        "entropy_sum": 0.0,
        "nll_sum": 0.0,
        "brier_sum": 0.0,
        "bin_n": [0] * bins,
        "bin_correct": [0] * bins,
        "bin_confidence": [0.0] * bins,
    }


def update_metrics(
    metrics: dict[str, object], true: int, pred: int, probabilities: list[float],
    confidence: float, entropy: float, bins: int,
) -> None:
    correct = int(true == pred)
    metrics["n"] += 1
    metrics["correct"] += correct
    metrics["confidence_sum"] += confidence
    metrics["entropy_sum"] += entropy
    metrics["nll_sum"] += -math.log(max(probabilities[true], 1e-12))
    metrics["brier_sum"] += sum(
        (probability - int(class_id == true)) ** 2
        for class_id, probability in enumerate(probabilities)
    )
    bin_id = min(bins - 1, int(confidence * bins))
    metrics["bin_n"][bin_id] += 1
    metrics["bin_correct"][bin_id] += correct
    metrics["bin_confidence"][bin_id] += confidence


def finalize(key: tuple[str, ...], metrics: dict[str, object]) -> dict[str, object]:
    n = int(metrics["n"])
    ece = 0.0
    for bin_n, bin_correct, bin_confidence in zip(
        metrics["bin_n"], metrics["bin_correct"], metrics["bin_confidence"]
    ):
        if bin_n:
            ece += bin_n / n * abs(bin_correct / bin_n - bin_confidence / bin_n)
    return {
        **dict(zip(("fold", "domain", "role", "head"), key)),
        "n_frames": n,
        "accuracy": metrics["correct"] / n,
        "mean_confidence": metrics["confidence_sum"] / n,
        "mean_entropy": metrics["entropy_sum"] / n,
        "nll": metrics["nll_sum"] / n,
        "brier": metrics["brier_sum"] / n,
        "ece": ece,
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.run_root.rglob("val_prediction_scores.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No val_prediction_scores.csv.gz under {args.run_root}")

    metrics = defaultdict(lambda: empty_metrics(args.ece_bins))
    class_stats = defaultdict(lambda: {"n": 0, "confidence": 0.0, "true_prob": 0.0})
    cr_sessions = defaultdict(lambda: empty_metrics(args.ece_bins))
    for path in paths:
        fold = path.parent.name
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                head = row["head"]
                true = int(row["y_true"])
                pred = int(row["y_pred"])
                probabilities = [
                    float(row[f"prob_{class_id}"])
                    for class_id in range(CLASS_COUNTS[head])
                ]
                confidence = float(row["confidence"])
                entropy = float(row["entropy"])
                key = (fold, row["domain"], row["role"], head)
                update_metrics(
                    metrics[key], true, pred, probabilities, confidence, entropy,
                    args.ece_bins,
                )
                pooled_key = ("all_oof", row["domain"], row["role"], head)
                update_metrics(
                    metrics[pooled_key], true, pred, probabilities, confidence,
                    entropy, args.ece_bins,
                )
                class_key = (*pooled_key[1:], true)
                class_stats[class_key]["n"] += 1
                class_stats[class_key]["confidence"] += confidence
                class_stats[class_key]["true_prob"] += probabilities[true]
                if row["domain"] == "CR" and head == "social":
                    session_key = (fold, row["session_id"], row["role"], head)
                    update_metrics(
                        cr_sessions[session_key], true, pred, probabilities,
                        confidence, entropy, args.ece_bins,
                    )

    metric_rows = [finalize(key, value) for key, value in sorted(metrics.items())]
    session_rows = []
    for key, value in sorted(cr_sessions.items()):
        row = finalize(key, value)
        row["session_id"] = row.pop("domain")
        session_rows.append(row)
    class_rows = []
    for key, values in sorted(class_stats.items()):
        n = values["n"]
        class_rows.append(
            {
                **dict(zip(("domain", "role", "head", "true_class"), key)),
                "n_frames": n,
                "mean_confidence": values["confidence"] / n,
                "mean_probability_of_true_class": values["true_prob"] / n,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "calibration_by_fold_and_regime.csv", metric_rows)
    write_rows(args.output_dir / "class_probability_summary.csv", class_rows)
    write_rows(args.output_dir / "cr_social_session_confidence.csv", session_rows)
    print(f"Analyzed {len(paths)} score exports; wrote {args.output_dir}")


if __name__ == "__main__":
    main()
