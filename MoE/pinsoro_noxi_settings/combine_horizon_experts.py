"""Combine two PinSoRo early-fusion horizon experts.

This is intentionally small and specific: it aligns two run directories by
frame-level prediction-score keys, combines logits with fixed or grid-searched
weights, and writes metrics plus combined prediction scores for HMM decoding.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np


CLASS_COUNTS = {"task": 4, "social": 5}
HEADS = ("task", "social")
MAX_CLASSES = max(CLASS_COUNTS.values())
Key = tuple[str, str, str, str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine short/long PinSoRo horizon experts.")
    parser.add_argument("--short-run", type=Path, required=True)
    parser.add_argument("--long-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixed", "val_grid", "train_grid"), default="fixed")
    parser.add_argument("--fixed-short-weight", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.05)
    return parser.parse_args()


def open_text(path: Path):
    return gzip.open(path, "rt", newline="", encoding="utf-8") if path.suffix == ".gz" else path.open(newline="", encoding="utf-8")


def read_scores(path: Path) -> dict[Key, dict[str, object]]:
    rows: dict[Key, dict[str, object]] = {}
    with open_text(path) as handle:
        for row in csv.DictReader(handle):
            head = row["head"]
            n_classes = CLASS_COUNTS[head]
            key = (
                row["domain"],
                row["source_split"],
                row["session_id"],
                row["role"],
                head,
                int(row["frame_idx"]),
            )
            logits = np.full(MAX_CLASSES, -1.0e9, dtype=np.float64)
            logits[:n_classes] = [float(row[f"logit_{idx}"]) for idx in range(n_classes)]
            rows[key] = {"y_true": int(row["y_true"]), "logits": logits}
    return rows


def kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    n = confusion.sum()
    if n == 0:
        return float("nan")
    accuracy = np.trace(confusion) / n
    expected = confusion.sum(axis=1) @ confusion.sum(axis=0) / (n * n)
    return float((accuracy - expected) / (1.0 - expected)) if expected < 1.0 else float("nan")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for klass in range(n_classes):
        tp = int(((y_true == klass) & (y_pred == klass)).sum())
        fp = int(((y_true != klass) & (y_pred == klass)).sum())
        fn = int(((y_true == klass) & (y_pred != klass)).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        if math.isnan(precision) or math.isnan(recall) or precision + recall == 0:
            continue
        f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else float("nan")


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    total = len(y_true)
    if total == 0:
        return float("nan")
    value = 0.0
    for klass in range(n_classes):
        support = int((y_true == klass).sum())
        if support == 0:
            continue
        tp = int(((y_true == klass) & (y_pred == klass)).sum())
        fp = int(((y_true != klass) & (y_pred == klass)).sum())
        fn = int(((y_true == klass) & (y_pred != klass)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        value += support * f1
    return float(value / total)


def metric_rows(keys: list[Key], labels: np.ndarray, logits: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for head in HEADS:
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        n_classes = CLASS_COUNTS[head]
        pred = np.argmax(logits[idx, :n_classes], axis=1)
        y = labels[idx]
        rows.append(
            {
                "group": "overall",
                "head": head,
                "n_frames": int(len(idx)),
                "kappa": kappa(y, pred, n_classes),
                "macro_f1": macro_f1(y, pred, n_classes),
                "weighted_f1": weighted_f1(y, pred, n_classes),
                "accuracy": float((y == pred).mean()) if len(idx) else float("nan"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_for(keys: list[Key], labels: np.ndarray, logits: np.ndarray, short_weight: float) -> float:
    _ = short_weight
    kappas = []
    for row in metric_rows(keys, labels, logits):
        value = float(row["kappa"])
        if np.isfinite(value):
            kappas.append(value)
    return float(np.mean(kappas)) if kappas else -float("inf")


def choose_weights(keys: list[Key], labels: np.ndarray, short_logits: np.ndarray, long_logits: np.ndarray, mode: str, fixed: float, step: float) -> dict[str, float]:
    if mode == "fixed":
        return {"task": fixed, "social": fixed}
    n = round(1.0 / step)
    weights = [idx / n for idx in range(n + 1)]
    selected = {}
    for head in HEADS:
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        best_weight = 0.5
        best_score = -float("inf")
        for weight in weights:
            combined = weight * short_logits[idx] + (1.0 - weight) * long_logits[idx]
            pred = np.argmax(combined[:, : CLASS_COUNTS[head]], axis=1)
            value = kappa(labels[idx], pred, CLASS_COUNTS[head])
            if np.isfinite(value) and value > best_score:
                best_score = value
                best_weight = float(weight)
        selected[head] = best_weight
    return selected


def write_prediction_scores(path: Path, keys: list[Key], labels: np.ndarray, logits: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        fieldnames = ["domain", "source_split", "session_id", "role", "head", "frame_idx", "y_true"] + [
            f"logit_{idx}" for idx in range(MAX_CLASSES)
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, y_true, row_logits in zip(keys, labels, logits):
            domain, source_split, session_id, role, head, frame_idx = key
            out = {
                "domain": domain,
                "source_split": source_split,
                "session_id": session_id,
                "role": role,
                "head": head,
                "frame_idx": frame_idx,
                "y_true": int(y_true),
            }
            for idx in range(MAX_CLASSES):
                out[f"logit_{idx}"] = float(row_logits[idx])
            writer.writerow(out)


def write_predictions(path: Path, keys: list[Key], labels: np.ndarray, logits: np.ndarray) -> None:
    rows = []
    for key, y_true, row_logits in zip(keys, labels, logits):
        domain, source_split, session_id, role, head, frame_idx = key
        n_classes = CLASS_COUNTS[head]
        rows.append(
            {
                "domain": domain,
                "source_split": source_split,
                "session_id": session_id,
                "role": role,
                "head": head,
                "frame_idx": frame_idx,
                "y_true": int(y_true),
                "y_pred": int(np.argmax(row_logits[:n_classes])),
            }
        )
    write_csv(path, rows)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.fixed_short_weight <= 1.0:
        raise ValueError("--fixed-short-weight must be in [0, 1].")
    short = read_scores(args.short_run / "val_prediction_scores.csv.gz")
    long = read_scores(args.long_run / "val_prediction_scores.csv.gz")
    long_config = json.loads((args.long_run / "config.json").read_text(encoding="utf-8"))
    keys = sorted(set(short) & set(long))
    if not keys:
        raise RuntimeError("No aligned validation scores between short and long runs.")
    labels = np.asarray([short[key]["y_true"] for key in keys], dtype=np.int64)
    long_labels = np.asarray([long[key]["y_true"] for key in keys], dtype=np.int64)
    if not np.array_equal(labels, long_labels):
        raise RuntimeError("Aligned labels disagree between short and long runs.")
    short_logits = np.stack([short[key]["logits"] for key in keys])
    long_logits = np.stack([long[key]["logits"] for key in keys])
    if args.mode == "train_grid":
        train_short = read_scores(args.short_run / "diagnostics" / "train_internal" / "val_prediction_scores.csv.gz")
        train_long = read_scores(args.long_run / "diagnostics" / "train_internal" / "val_prediction_scores.csv.gz")
        train_keys = sorted(set(train_short) & set(train_long))
        if not train_keys:
            raise RuntimeError("No aligned train/internal scores. Re-export runs with --include-train first.")
        train_labels = np.asarray([train_short[key]["y_true"] for key in train_keys], dtype=np.int64)
        train_long_labels = np.asarray([train_long[key]["y_true"] for key in train_keys], dtype=np.int64)
        if not np.array_equal(train_labels, train_long_labels):
            raise RuntimeError("Aligned train labels disagree between short and long runs.")
        train_short_logits = np.stack([train_short[key]["logits"] for key in train_keys])
        train_long_logits = np.stack([train_long[key]["logits"] for key in train_keys])
        weights = choose_weights(train_keys, train_labels, train_short_logits, train_long_logits, "val_grid", args.fixed_short_weight, args.step)
    else:
        weights = choose_weights(keys, labels, short_logits, long_logits, args.mode, args.fixed_short_weight, args.step)
    combined = np.empty_like(short_logits)
    for head in HEADS:
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        weight = weights[head]
        combined[idx] = weight * short_logits[idx] + (1.0 - weight) * long_logits[idx]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics_overall.csv", metric_rows(keys, labels, combined))
    write_prediction_scores(args.output_dir / "val_prediction_scores.csv.gz", keys, labels, combined)
    write_predictions(args.output_dir / "val_predictions.csv", keys, labels, combined)
    config = {
        "architecture": "two_horizon_logit_combiner",
        "mode": args.mode,
        "weights_short": weights,
        "short_run": str(args.short_run),
        "long_run": str(args.long_run),
        "n_aligned_rows": len(keys),
        "manifest": long_config["manifest"],
        "train_split": long_config["train_split"],
        "val_split": long_config["val_split"],
        "domain_scope": long_config["domain_scope"],
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
