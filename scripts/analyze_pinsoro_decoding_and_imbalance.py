"""Benchmark label smoothing and diagnose imbalance from PinSoRo CV predictions."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "model improvement/test 1/results"
DEFAULT_OUTPUT = PROJECT_ROOT / "model improvement/prepared analyses"
KEY_FIELDS = ("domain", "source_split", "session_id", "role", "head")
GROUP_FIELDS = ("domain", "role", "head")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze smoothing, class imbalance, and CR-social errors."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[31, 61, 151, 301, 601, 901],
        help="Odd-sized majority-filter windows in frames.",
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


def read_predictions(path: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], dict[str, list[int]]] = defaultdict(
        lambda: {"frame_idx": [], "y_true": [], "y_pred": []}
    )
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[field] for field in KEY_FIELDS)
            grouped[key]["frame_idx"].append(int(row["frame_idx"]))
            grouped[key]["y_true"].append(int(row["y_true"]))
            grouped[key]["y_pred"].append(int(row["y_pred"]))

    timelines = []
    for key, values in grouped.items():
        order = np.argsort(np.asarray(values["frame_idx"]), kind="stable")
        timelines.append(
            {
                **dict(zip(KEY_FIELDS, key)),
                "frame_idx": np.asarray(values["frame_idx"], dtype=np.int64)[order],
                "y_true": np.asarray(values["y_true"], dtype=np.int16)[order],
                "y_pred": np.asarray(values["y_pred"], dtype=np.int16)[order],
            }
        )
    return timelines


def majority_filter(values: np.ndarray, window: int, causal: bool) -> np.ndarray:
    if window <= 1 or len(values) <= 1:
        return values.copy()
    labels = np.unique(values)
    positions = np.arange(len(values))
    if causal:
        left = np.maximum(0, positions - window + 1)
        right = positions + 1
    else:
        radius = window // 2
        left = np.maximum(0, positions - radius)
        right = np.minimum(len(values), positions + radius + 1)

    counts = np.empty((len(labels), len(values)), dtype=np.int32)
    for index, label in enumerate(labels):
        cumulative = np.r_[0, np.cumsum(values == label, dtype=np.int32)]
        counts[index] = cumulative[right] - cumulative[left]
    maxima = counts.max(axis=0)
    result = labels[np.argmax(counts, axis=0)]
    original_index = np.searchsorted(labels, values)
    keep_original = counts[original_index, positions] == maxima
    result[keep_original] = values[keep_original]
    return result.astype(values.dtype, copy=False)


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.union1d(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels)
    observed = float(np.trace(matrix) / matrix.sum())
    expected = float(matrix.sum(axis=1) @ matrix.sum(axis=0) / matrix.sum() ** 2)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 0.0


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(
        matrix,
        (np.searchsorted(labels, y_true), np.searchsorted(labels, y_pred)),
        1,
    )
    return matrix


def run_lengths(values: np.ndarray) -> np.ndarray:
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    return np.diff(np.r_[0, boundaries, len(values)])


def transition_distance(values: np.ndarray) -> np.ndarray:
    transitions = np.flatnonzero(values[1:] != values[:-1]) + 1
    distance = np.full(len(values), len(values), dtype=np.int32)
    if not len(transitions):
        return distance
    mask = np.zeros(len(values), dtype=bool)
    mask[transitions] = True
    last = -len(values)
    for index in range(len(values)):
        if mask[index]:
            last = index
        distance[index] = index - last
    last = 2 * len(values)
    for index in range(len(values) - 1, -1, -1):
        if mask[index]:
            last = index
        distance[index] = min(distance[index], last - index)
    return distance


def metric_row(
    fold: str,
    group: tuple[str, str, str],
    decoder: str,
    window: int,
    timelines: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    true = np.concatenate([item[0] for item in timelines])
    pred = np.concatenate([item[1] for item in timelines])
    labels = np.union1d(true, pred)
    matrix = confusion_matrix(true, pred, labels)
    true_counts = matrix.sum(axis=1)
    pred_counts = matrix.sum(axis=0)
    recalls = np.divide(
        np.diag(matrix), true_counts, out=np.zeros(len(labels), dtype=float), where=true_counts > 0
    )
    precisions = np.divide(
        np.diag(matrix), pred_counts, out=np.zeros(len(labels), dtype=float), where=pred_counts > 0
    )
    f1 = np.divide(
        2 * recalls * precisions,
        recalls + precisions,
        out=np.zeros(len(labels), dtype=float),
        where=(recalls + precisions) > 0,
    )
    true_transitions = sum(int(np.sum(a[1:] != a[:-1])) for a, _ in timelines)
    pred_transitions = sum(int(np.sum(b[1:] != b[:-1])) for _, b in timelines)
    true_opportunities = sum(max(0, len(a) - 1) for a, _ in timelines)
    true_rate = true_transitions / true_opportunities
    pred_rate = pred_transitions / true_opportunities
    return {
        "fold": fold,
        **dict(zip(GROUP_FIELDS, group)),
        "decoder": decoder,
        "window_frames": window,
        "n_frames": len(true),
        "kappa": cohen_kappa(true, pred),
        "accuracy": float(np.mean(true == pred)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1)),
        "true_transition_rate": true_rate,
        "pred_transition_rate": pred_rate,
        "transition_rate_ratio": pred_rate / true_rate if true_rate else float("nan"),
        "true_median_run_frames": float(
            median(np.concatenate([run_lengths(a) for a, _ in timelines]))
        ),
        "pred_median_run_frames": float(
            median(np.concatenate([run_lengths(b) for _, b in timelines]))
        ),
    }


def decoder_metrics(
    folds: dict[str, list[dict[str, object]]], windows: list[int]
) -> list[dict[str, object]]:
    rows = []
    decoder_specs = [("raw", 1, False)]
    decoder_specs += [("centered_majority", window, False) for window in windows]
    decoder_specs += [("causal_majority", window, True) for window in windows]
    for fold_name, fold_timelines in {**folds, "all_oof": sum(folds.values(), [])}.items():
        groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
        for timeline in fold_timelines:
            group = tuple(str(timeline[field]) for field in GROUP_FIELDS)
            groups[group].append(timeline)
        for group, timelines in sorted(groups.items()):
            for decoder, window, causal in decoder_specs:
                pairs = [
                    (
                        timeline["y_true"],
                        timeline["y_pred"]
                        if decoder == "raw"
                        else majority_filter(timeline["y_pred"], window, causal),
                    )
                    for timeline in timelines
                ]
                rows.append(metric_row(fold_name, group, decoder, window, pairs))
    return rows


def imbalance_rows(
    folds: dict[str, list[dict[str, object]]]
) -> list[dict[str, object]]:
    rows = []
    for fold_name, timelines in {**folds, "all_oof": sum(folds.values(), [])}.items():
        groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
        for timeline in timelines:
            groups[tuple(str(timeline[field]) for field in GROUP_FIELDS)].append(timeline)
        for group, items in sorted(groups.items()):
            true = np.concatenate([item["y_true"] for item in items])
            pred = np.concatenate([item["y_pred"] for item in items])
            labels = np.union1d(true, pred)
            for label in labels:
                true_mask = true == label
                pred_mask = pred == label
                tp = int(np.sum(true_mask & pred_mask))
                rows.append(
                    {
                        "fold": fold_name,
                        **dict(zip(GROUP_FIELDS, group)),
                        "class": int(label),
                        "true_count": int(true_mask.sum()),
                        "true_fraction": float(true_mask.mean()),
                        "inverse_frequency_weight": len(true) / (len(labels) * true_mask.sum())
                        if true_mask.any()
                        else float("nan"),
                        "pred_count": int(pred_mask.sum()),
                        "pred_fraction": float(pred_mask.mean()),
                        "recall": tp / true_mask.sum() if true_mask.any() else float("nan"),
                        "precision": tp / pred_mask.sum() if pred_mask.any() else float("nan"),
                    }
                )
    return rows


def cr_social_rows(
    folds: dict[str, list[dict[str, object]]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    timelines = [
        (fold, timeline)
        for fold, items in folds.items()
        for timeline in items
        if timeline["domain"] == "CR" and timeline["head"] == "social"
    ]
    session_rows = []
    context_buckets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    true_all = []
    pred_all = []
    for fold, timeline in timelines:
        true = timeline["y_true"]
        pred = timeline["y_pred"]
        true_all.append(true)
        pred_all.append(pred)
        session_rows.append(
            {
                "fold": fold,
                "session_id": timeline["session_id"],
                "role": timeline["role"],
                "n_frames": len(true),
                "kappa": cohen_kappa(true, pred),
                "accuracy": float(np.mean(true == pred)),
                "true_dominant_class": Counter(true.tolist()).most_common(1)[0][0],
                "true_dominant_fraction": Counter(true.tolist()).most_common(1)[0][1]
                / len(true),
                "pred_dominant_class": Counter(pred.tolist()).most_common(1)[0][0],
                "pred_dominant_fraction": Counter(pred.tolist()).most_common(1)[0][1]
                / len(pred),
                "transition_rate_ratio": float(np.sum(pred[1:] != pred[:-1]))
                / max(1, int(np.sum(true[1:] != true[:-1]))),
            }
        )
        distance = transition_distance(true)
        for name, mask in (
            ("0-30", distance <= 30),
            ("31-150", (distance > 30) & (distance <= 150)),
            ("151-600", (distance > 150) & (distance <= 600)),
            (">600", distance > 600),
        ):
            if mask.any():
                context_buckets[name].append((true[mask], pred[mask]))

    context_rows = []
    for bucket, pairs in context_buckets.items():
        true = np.concatenate([pair[0] for pair in pairs])
        pred = np.concatenate([pair[1] for pair in pairs])
        context_rows.append(
            {
                "distance_to_true_transition_frames": bucket,
                "n_frames": len(true),
                "accuracy": float(np.mean(true == pred)),
                "kappa": cohen_kappa(true, pred),
            }
        )

    true = np.concatenate(true_all)
    pred = np.concatenate(pred_all)
    labels = np.union1d(true, pred)
    matrix = confusion_matrix(true, pred, labels)
    confusion_rows = []
    for true_index, true_label in enumerate(labels):
        true_count = int(matrix[true_index].sum())
        for pred_index, pred_label in enumerate(labels):
            confusion_rows.append(
                {
                    "true_class": int(true_label),
                    "pred_class": int(pred_label),
                    "count": int(matrix[true_index, pred_index]),
                    "fraction_of_true_class": float(
                        matrix[true_index, pred_index] / true_count
                    )
                    if true_count
                    else float("nan"),
                }
            )
    return session_rows, context_rows, confusion_rows


def fmt(value: object) -> str:
    if isinstance(value, float):
        return "nan" if not math.isfinite(value) else f"{value:.4f}"
    return str(value)


def make_report(
    decoder_rows: list[dict[str, object]],
    imbalance: list[dict[str, object]],
    sessions: list[dict[str, object]],
    context: list[dict[str, object]],
) -> str:
    aggregate = [row for row in decoder_rows if row["fold"] == "all_oof"]
    groups = sorted({tuple(row[field] for field in GROUP_FIELDS) for row in aggregate})
    lines = [
        "# Prepared Decoding, Imbalance, and CR-Social Analysis",
        "",
        "These results use the completed out-of-fold validation predictions. Re-run the script on each candidate architecture before selecting postprocessing parameters.",
        "",
        "## Best Centered Majority Filter Per Group",
        "",
        "| Domain | Role | Head | Raw Kappa | Best Window | Best Kappa | Transition Ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for group in groups:
        matches = [
            row
            for row in aggregate
            if tuple(row[field] for field in GROUP_FIELDS) == group
        ]
        raw = next(row for row in matches if row["decoder"] == "raw")
        best = max(
            (row for row in matches if row["decoder"] == "centered_majority"),
            key=lambda row: float(row["kappa"]),
        )
        lines.append(
            f"| {group[0]} | {group[1]} | {group[2]} | {fmt(raw['kappa'])} | "
            f"{best['window_frames']} | {fmt(best['kappa'])} | "
            f"{fmt(best['transition_rate_ratio'])} |"
        )

    aggregate_imbalance = [row for row in imbalance if row["fold"] == "all_oof"]
    worst_recall = sorted(
        aggregate_imbalance, key=lambda row: float(row["recall"])
    )[:10]
    lines += [
        "",
        "## Lowest-Recall Classes",
        "",
        "| Domain | Role | Head | Class | True Fraction | Pred Fraction | Recall |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in worst_recall:
        lines.append(
            f"| {row['domain']} | {row['role']} | {row['head']} | {row['class']} | "
            f"{fmt(row['true_fraction'])} | {fmt(row['pred_fraction'])} | "
            f"{fmt(row['recall'])} |"
        )

    lines += [
        "",
        "## CR Social: Distance From True Transitions",
        "",
        "| Distance (frames) | Frames | Accuracy | Kappa |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in context:
        lines.append(
            f"| {row['distance_to_true_transition_frames']} | {row['n_frames']} | "
            f"{fmt(row['accuracy'])} | {fmt(row['kappa'])} |"
        )
    worst_sessions = sorted(sessions, key=lambda row: float(row["kappa"]))[:8]
    lines += [
        "",
        "## Worst CR Social Sessions",
        "",
        "| Fold | Session | Role | Frames | Kappa | Accuracy | Pred Dominant Fraction |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in worst_sessions:
        lines.append(
            f"| {row['fold']} | {row['session_id']} | {row['role']} | "
            f"{row['n_frames']} | {fmt(row['kappa'])} | {fmt(row['accuracy'])} | "
            f"{fmt(row['pred_dominant_fraction'])} |"
        )
    lines += [
        "",
        "Centered filtering is suitable for offline submission. Causal results quantify the cost of a real-time constraint. Class-weight experiments still require retraining; the imbalance outputs identify which classes and groups to prioritize.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    windows = sorted(set(args.windows))
    if any(window < 1 or window % 2 == 0 for window in windows):
        raise ValueError("All majority-filter windows must be positive odd integers.")
    prediction_paths = sorted(args.run_root.rglob("val_predictions.csv"))
    if not prediction_paths:
        raise FileNotFoundError(f"No val_predictions.csv files under {args.run_root}")

    folds = {path.parent.name: read_predictions(path) for path in prediction_paths}
    decoder_rows = decoder_metrics(folds, windows)
    imbalance = imbalance_rows(folds)
    sessions, context, confusion = cr_social_rows(folds)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "decoder_benchmark.csv", decoder_rows)
    write_rows(output / "class_imbalance_diagnostics.csv", imbalance)
    write_rows(output / "cr_social_session_diagnostics.csv", sessions)
    write_rows(output / "cr_social_transition_context.csv", context)
    write_rows(output / "cr_social_confusion.csv", confusion)
    (output / "report.md").write_text(
        make_report(decoder_rows, imbalance, sessions, context), encoding="utf-8"
    )
    print(f"Analyzed {len(prediction_paths)} folds; wrote outputs to {output}")


if __name__ == "__main__":
    main()
