"""Diagnose PinSoRo frame predictions and validation-to-test behavior shifts."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    PROJECT_ROOT
    / "outputs/pinsoro/experiments/pinsoro_visual_videomae_attention_seed13"
)
GROUP_FIELDS = ("domain", "role", "head")
TIMELINE_FIELDS = ("domain", "source_split", "session_id", "role", "head")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze PinSoRo validation errors and test prediction shift."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--transition-radii",
        type=int,
        nargs="*",
        default=[0, 15, 30, 75, 150],
        help="Frame radii used to measure accuracy near true transitions.",
    )
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_timelines(path: Path) -> dict[tuple[str, ...], dict[str, np.ndarray]]:
    timelines: dict[tuple[str, ...], dict[str, list[int]]] = defaultdict(
        lambda: {"frame_idx": [], "y_pred": [], "y_true": []}
    )
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[field] for field in TIMELINE_FIELDS)
            timelines[key]["frame_idx"].append(int(row["frame_idx"]))
            timelines[key]["y_pred"].append(int(row["y_pred"]))
            if row.get("y_true", "") != "":
                timelines[key]["y_true"].append(int(row["y_true"]))
    result = {}
    for key, values in timelines.items():
        order = np.argsort(np.asarray(values["frame_idx"]), kind="stable")
        result[key] = {
            name: np.asarray(items, dtype=np.int64 if name == "frame_idx" else np.int16)[order]
            for name, items in values.items()
            if items
        }
    return result


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    labels = np.union1d(y_true, y_pred)
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    true_idx = np.searchsorted(labels, y_true)
    pred_idx = np.searchsorted(labels, y_pred)
    np.add.at(confusion, (true_idx, pred_idx), 1)
    observed = float(np.trace(confusion) / len(y_true))
    expected = float(confusion.sum(axis=1) @ confusion.sum(axis=0) / len(y_true) ** 2)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 0.0


def run_lengths(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.asarray([], dtype=np.int64)
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    return np.diff(np.r_[0, boundaries, len(values)])


def distribution(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    counts = np.asarray([(values == label).sum() for label in labels], dtype=float)
    return counts / counts.sum() if counts.sum() else counts


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = (left + right) / 2.0

    def kl_divergence(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return (kl_divergence(left, midpoint) + kl_divergence(right, midpoint)) / 2.0


def fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, float):
        return "nan" if not math.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def analyze_supervised(
    timelines: dict[tuple[str, ...], dict[str, np.ndarray]],
    radii: list[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, np.ndarray]]] = defaultdict(list)
    session_rows = []
    for key, arrays in timelines.items():
        if "y_true" not in arrays:
            continue
        group = (key[0], key[3], key[4])
        grouped[group].append(arrays)
        true = arrays["y_true"]
        pred = arrays["y_pred"]
        session_rows.append(
            {
                **dict(zip(TIMELINE_FIELDS, key)),
                "n_frames": len(true),
                "kappa": cohen_kappa(true, pred),
                "accuracy": float(np.mean(true == pred)),
                "true_transition_rate": float(np.mean(true[1:] != true[:-1])),
                "pred_transition_rate": float(np.mean(pred[1:] != pred[:-1])),
                "pred_dominant_fraction": float(
                    max(Counter(pred.tolist()).values()) / len(pred)
                ),
                "pred_unique_classes": len(np.unique(pred)),
            }
        )

    summary_rows = []
    class_rows = []
    for group, items in sorted(grouped.items()):
        true = np.concatenate([item["y_true"] for item in items])
        pred = np.concatenate([item["y_pred"] for item in items])
        labels = np.union1d(true, pred)
        true_transitions = np.flatnonzero(true[1:] != true[:-1]) + 1
        pred_transitions = np.flatnonzero(pred[1:] != pred[:-1]) + 1
        true_rate = len(true_transitions) / max(1, len(true) - 1)
        pred_rate = len(pred_transitions) / max(1, len(pred) - 1)
        row: dict[str, object] = {
            **dict(zip(GROUP_FIELDS, group)),
            "n_frames": len(true),
            "kappa": cohen_kappa(true, pred),
            "accuracy": float(np.mean(true == pred)),
            "majority_accuracy": float(max(Counter(true.tolist()).values()) / len(true)),
            "true_transition_rate": true_rate,
            "pred_transition_rate": pred_rate,
            "transition_rate_ratio": pred_rate / true_rate if true_rate else float("nan"),
            "true_median_run_frames": float(median(run_lengths(true))),
            "pred_median_run_frames": float(median(run_lengths(pred))),
        }
        distance = np.full(len(true), len(true), dtype=np.int32)
        if len(true_transitions):
            transition_mask = np.zeros(len(true), dtype=bool)
            transition_mask[true_transitions] = True
            last = -len(true)
            for index in range(len(true)):
                if transition_mask[index]:
                    last = index
                distance[index] = index - last
            last = 2 * len(true)
            for index in range(len(true) - 1, -1, -1):
                if transition_mask[index]:
                    last = index
                distance[index] = min(distance[index], last - index)
        for radius in radii:
            near = distance <= radius
            row[f"accuracy_within_{radius}f"] = (
                float(np.mean(true[near] == pred[near])) if near.any() else float("nan")
            )
            row[f"n_within_{radius}f"] = int(near.sum())
        summary_rows.append(row)

        for label in labels:
            true_mask = true == label
            pred_mask = pred == label
            true_positive = int(np.sum(true_mask & pred_mask))
            class_rows.append(
                {
                    **dict(zip(GROUP_FIELDS, group)),
                    "class": int(label),
                    "true_count": int(true_mask.sum()),
                    "true_fraction": float(true_mask.mean()),
                    "pred_count": int(pred_mask.sum()),
                    "pred_fraction": float(pred_mask.mean()),
                    "recall": true_positive / true_mask.sum() if true_mask.any() else float("nan"),
                    "precision": true_positive / pred_mask.sum() if pred_mask.any() else float("nan"),
                }
            )
    return summary_rows, class_rows, session_rows


def prediction_shift_rows(
    val: dict[tuple[str, ...], dict[str, np.ndarray]],
    test: dict[tuple[str, ...], dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    grouped: dict[str, dict[tuple[str, str, str], list[np.ndarray]]] = {
        "validation": defaultdict(list),
        "test": defaultdict(list),
    }
    for split_name, timelines in (("validation", val), ("test", test)):
        for key, arrays in timelines.items():
            grouped[split_name][(key[0], key[3], key[4])].append(arrays["y_pred"])

    rows = []
    groups = sorted(set(grouped["validation"]) | set(grouped["test"]))
    for group in groups:
        val_items = grouped["validation"].get(group, [])
        test_items = grouped["test"].get(group, [])
        if not val_items or not test_items:
            continue
        val_values = np.concatenate(val_items)
        test_values = np.concatenate(test_items)
        labels = np.union1d(val_values, test_values)
        val_dist = distribution(val_values, labels)
        test_dist = distribution(test_values, labels)
        for label, val_fraction, test_fraction in zip(labels, val_dist, test_dist):
            rows.append(
                {
                    **dict(zip(GROUP_FIELDS, group)),
                    "class": int(label),
                    "val_pred_fraction": float(val_fraction),
                    "test_pred_fraction": float(test_fraction),
                    "test_minus_val": float(test_fraction - val_fraction),
                    "group_js_divergence": js_divergence(val_dist, test_dist),
                    "val_n_frames": len(val_values),
                    "test_n_frames": len(test_values),
                }
            )
    return rows


def make_report(
    run_dir: Path,
    summary_rows: list[dict[str, object]],
    class_rows: list[dict[str, object]],
    session_rows: list[dict[str, object]],
    shift_rows: list[dict[str, object]],
) -> str:
    lines = [
        "# PinSoRo Prediction Error Analysis",
        "",
        f"Run: `{run_dir}`",
        "",
        "Test labels are withheld, so test error attribution is impossible. Test analysis below uses prediction-distribution shift only.",
        "",
        "## Group Summary",
        "",
        "| Domain | Role | Head | Kappa | Accuracy | Majority Acc. | Pred/True Transition Rate | Median Run Pred/True |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['domain']} | {row['role']} | {row['head']} | "
            f"{fmt(row['kappa'])} | {fmt(row['accuracy'])} | "
            f"{fmt(row['majority_accuracy'])} | {fmt(row['transition_rate_ratio'])} | "
            f"{fmt(row['pred_median_run_frames'], 1)}/{fmt(row['true_median_run_frames'], 1)} |"
        )

    lines += ["", "## Largest Validation Class-Recall Failures", ""]
    recall_rows = sorted(
        (row for row in class_rows if math.isfinite(float(row["recall"]))),
        key=lambda row: float(row["recall"]),
    )[:12]
    lines += [
        "| Domain | Role | Head | Class | True Fraction | Pred Fraction | Recall | Precision |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in recall_rows:
        lines.append(
            f"| {row['domain']} | {row['role']} | {row['head']} | {row['class']} | "
            f"{fmt(row['true_fraction'])} | {fmt(row['pred_fraction'])} | "
            f"{fmt(row['recall'])} | {fmt(row['precision'])} |"
        )

    lines += ["", "## Largest Validation-to-Test Prediction Shifts", ""]
    shift_groups = {}
    for row in shift_rows:
        key = tuple(row[field] for field in GROUP_FIELDS)
        shift_groups[key] = float(row["group_js_divergence"])
    lines += [
        "| Domain | Role | Head | Jensen-Shannon Divergence |",
        "| --- | --- | --- | ---: |",
    ]
    for group, divergence in sorted(
        shift_groups.items(), key=lambda item: item[1], reverse=True
    ):
        lines.append(
            f"| {group[0]} | {group[1]} | {group[2]} | {fmt(divergence)} |"
        )

    worst_sessions = sorted(session_rows, key=lambda row: float(row["kappa"]))[:10]
    lines += [
        "",
        "## Worst Validation Timelines",
        "",
        "| Domain | Session | Role | Head | Frames | Kappa | Accuracy | Pred Dominant Fraction |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in worst_sessions:
        lines.append(
            f"| {row['domain']} | {row['session_id']} | {row['role']} | {row['head']} | "
            f"{row['n_frames']} | {fmt(row['kappa'])} | {fmt(row['accuracy'])} | "
            f"{fmt(row['pred_dominant_fraction'])} |"
        )
    lines += [
        "",
        "Detailed machine-readable outputs are beside this report.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir or run_dir / "error_analysis"
    val_path = run_dir / "val_predictions.csv"
    test_path = run_dir / "test_predictions.csv"
    if not val_path.is_file():
        raise FileNotFoundError("Run directory must contain val_predictions.csv")

    val = read_timelines(val_path)
    test = read_timelines(test_path) if test_path.is_file() else {}
    summary_rows, class_rows, session_rows = analyze_supervised(
        val, sorted(set(args.transition_radii))
    )
    shift_rows = prediction_shift_rows(val, test)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "validation_group_summary.csv", summary_rows)
    write_rows(output_dir / "validation_class_metrics.csv", class_rows)
    write_rows(output_dir / "validation_session_metrics.csv", session_rows)
    write_rows(output_dir / "validation_test_prediction_shift.csv", shift_rows)
    report = make_report(run_dir, summary_rows, class_rows, session_rows, shift_rows)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Wrote PinSoRo error analysis to {output_dir}")


if __name__ == "__main__":
    main()
