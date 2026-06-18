"""Temporal smoothing ablations for selected PinSoRo MoE 1 logits.

Selected base model: metadata-head modality experts + logit-space two_head combiner.
This script applies post-hoc smoothing to combined validation logits/probabilities
or hard predictions, then reports kappa and label-flip rates.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "MoE" / "experiments"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "moe1_temporal_smoothing_ablation"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
CLASS_COUNTS = {"task": 4, "social": 5}
HEADS = ("task", "social")
MAX_CLASSES = max(CLASS_COUNTS.values())
Key = tuple[str, str, str, str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal smoothing ablations for MoE 1.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cc-expert-root", type=Path)
    parser.add_argument("--cr-expert-root", type=Path)
    parser.add_argument("--cc-combiner-root", type=Path)
    parser.add_argument("--cr-combiner-root", type=Path)
    parser.add_argument("--windows", nargs="+", type=int, default=[3, 5, 9, 15, 31, 61, 121])
    parser.add_argument("--ema-alphas", nargs="+", type=float, default=[0.15, 0.25, 0.4, 0.6])
    parser.add_argument("--hysteresis-margins", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    return parser.parse_args()


def expert_root(domain: str, args: argparse.Namespace | None = None) -> Path:
    if args is not None:
        custom = args.cc_expert_root if domain == "CC" else args.cr_expert_root
        if custom is not None:
            return custom
    return EXPERIMENT_ROOT / f"moe1_{domain.lower()}_metadata_head_experts"


def score_path(domain: str, feature: str, args: argparse.Namespace | None = None) -> Path:
    run = expert_root(domain, args) / f"{domain.lower()}_{feature}_dyadic_tcn_k11_seed13"
    return run / "val_prediction_scores.csv.gz"


def combiner_summary_path(domain: str, args: argparse.Namespace | None = None) -> Path:
    if args is not None:
        custom = args.cc_combiner_root if domain == "CC" else args.cr_combiner_root
        if custom is not None:
            return custom / "two_head" / "summary.json"
    return EXPERIMENT_ROOT / f"moe1_{domain.lower()}_metadata_head_combiners" / "two_head" / "summary.json"


def read_scores(path: Path) -> dict[Key, dict[str, object]]:
    rows: dict[Key, dict[str, object]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
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


def read_domain_data(domain: str, args: argparse.Namespace | None = None) -> tuple[list[Key], np.ndarray, dict[str, np.ndarray]]:
    by_feature = {feature: read_scores(score_path(domain, feature, args)) for feature in FEATURES}
    keys = sorted(set.intersection(*(set(rows) for rows in by_feature.values())))
    labels = np.asarray([by_feature[FEATURES[0]][key]["y_true"] for key in keys], dtype=np.int64)
    logits = {
        feature: np.stack([by_feature[feature][key]["logits"] for key in keys])
        for feature in FEATURES
    }
    return keys, labels, logits


def read_weights(domain: str, args: argparse.Namespace | None = None) -> dict[str, np.ndarray]:
    summary = json.loads(combiner_summary_path(domain, args).read_text(encoding="utf-8"))
    return {head: np.asarray(weights, dtype=np.float64) for head, weights in summary["weights_by_group"].items()}


def combine_two_head(keys: list[Key], logits: dict[str, np.ndarray], weights: dict[str, np.ndarray]) -> np.ndarray:
    combined = np.empty((len(keys), MAX_CLASSES), dtype=np.float64)
    for head in HEADS:
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        head_weights = weights[head]
        combined[idx] = sum(float(weight) * logits[feature][idx] for weight, feature in zip(head_weights, FEATURES))
    return combined


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def centered_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    radius = window // 2
    out = np.empty_like(values, dtype=np.float64)
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        out[idx] = values[start:end].mean(axis=0)
    return out


def causal_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    state = values[0].astype(np.float64)
    out[0] = state
    for idx in range(1, len(values)):
        state = alpha * values[idx] + (1.0 - alpha) * state
        out[idx] = state
    return out


def majority_filter(pred: np.ndarray, window: int, n_classes: int) -> np.ndarray:
    if window <= 1:
        return pred.copy()
    radius = window // 2
    out = np.empty_like(pred)
    for idx in range(len(pred)):
        start = max(0, idx - radius)
        end = min(len(pred), idx + radius + 1)
        counts = np.bincount(pred[start:end], minlength=n_classes)
        out[idx] = int(counts.argmax())
    return out


def hysteresis(logits: np.ndarray, margin: float, n_classes: int) -> np.ndarray:
    pred = logits[:, :n_classes].argmax(axis=1)
    out = np.empty_like(pred)
    current = int(pred[0])
    out[0] = current
    for idx in range(1, len(pred)):
        candidate = int(pred[idx])
        if candidate != current and logits[idx, candidate] >= logits[idx, current] + margin:
            current = candidate
        out[idx] = current
    return out


def groups_for_sequences(keys: list[Key]) -> dict[tuple[str, str, str], list[int]]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        _domain, _source_split, session_id, role, head, _frame_idx = key
        groups[(session_id, role, head)].append(idx)
    return {group: sorted(idxs, key=lambda idx: keys[idx][5]) for group, idxs in groups.items()}


def kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    n = confusion.sum()
    if n == 0:
        return float("nan")
    accuracy = np.trace(confusion) / n
    expected = confusion.sum(axis=1) @ confusion.sum(axis=0) / (n * n)
    return float((accuracy - expected) / (1.0 - expected)) if expected < 1.0 else float("nan")


def transition_stats(keys: list[Key], labels: np.ndarray, pred: np.ndarray, head: str) -> dict[str, object]:
    true_flips = 0
    pred_flips = 0
    transitions = 0
    sequences = 0
    for (_session, _role, group_head), idxs in groups_for_sequences(keys).items():
        if group_head != head or len(idxs) < 2:
            continue
        sequences += 1
        y_true = labels[idxs]
        y_pred = pred[idxs]
        true_flips += int((y_true[1:] != y_true[:-1]).sum())
        pred_flips += int((y_pred[1:] != y_pred[:-1]).sum())
        transitions += len(idxs) - 1
    return {
        "n_sequences": sequences,
        "n_transitions": transitions,
        "true_flips": true_flips,
        "pred_flips": pred_flips,
        "true_flip_rate": true_flips / transitions if transitions else float("nan"),
        "pred_flip_rate": pred_flips / transitions if transitions else float("nan"),
        "flip_rate_ratio": pred_flips / true_flips if true_flips else float("inf") if pred_flips else 0.0,
        "excess_flips": pred_flips - true_flips,
    }



def class_metric_rows(keys: list[Key], labels: np.ndarray, pred: np.ndarray, domain: str, mode: str, param: str) -> list[dict[str, object]]:
    rows = []
    for head in HEADS:
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        y_true = labels[idx]
        y_pred = pred[idx]
        for klass in range(CLASS_COUNTS[head]):
            tp = int(((y_true == klass) & (y_pred == klass)).sum())
            fp = int(((y_true != klass) & (y_pred == klass)).sum())
            fn = int(((y_true == klass) & (y_pred != klass)).sum())
            support = int((y_true == klass).sum())
            precision = tp / (tp + fp) if tp + fp else float("nan")
            recall = tp / (tp + fn) if tp + fn else float("nan")
            f1 = 2 * precision * recall / (precision + recall) if precision + recall and not np.isnan(precision) and not np.isnan(recall) else float("nan")
            rows.append({
                "domain": domain,
                "mode": mode,
                "param": param,
                "head": head,
                "class": klass,
                "support": support,
                "predicted": int((y_pred == klass).sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
            })
    return rows

def evaluate(keys: list[Key], labels: np.ndarray, pred: np.ndarray, domain: str, mode: str, param: str) -> list[dict[str, object]]:
    rows = []
    for head in HEADS:
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        rows.append({
            "domain": domain,
            "mode": mode,
            "param": param,
            "head": head,
            "n_frames": len(idx),
            "accuracy": float((labels[idx] == pred[idx]).mean()),
            "kappa": kappa(labels[idx], pred[idx], CLASS_COUNTS[head]),
            **transition_stats(keys, labels, pred, head),
        })
    return rows


def apply_by_sequence(
    keys: list[Key],
    base_logits: np.ndarray,
    base_pred: np.ndarray,
    mode: str,
    value: float | int,
) -> np.ndarray:
    out = np.empty(len(keys), dtype=np.int64)
    for (_session, _role, head), idxs in groups_for_sequences(keys).items():
        n_classes = CLASS_COUNTS[head]
        seq_logits = base_logits[idxs]
        if mode == "logit_mean":
            out[idxs] = centered_mean(seq_logits, int(value))[:, :n_classes].argmax(axis=1)
        elif mode == "prob_mean":
            out[idxs] = centered_mean(softmax(seq_logits), int(value))[:, :n_classes].argmax(axis=1)
        elif mode == "label_majority":
            out[idxs] = majority_filter(base_pred[idxs], int(value), n_classes)
        elif mode == "logit_ema":
            out[idxs] = causal_ema(seq_logits, float(value))[:, :n_classes].argmax(axis=1)
        elif mode == "hysteresis":
            out[idxs] = hysteresis(seq_logits, float(value), n_classes)
        else:
            raise ValueError(mode)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_domain(domain: str, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    keys, labels, logits = read_domain_data(domain, args)
    combined_logits = combine_two_head(keys, logits, read_weights(domain, args))
    base_pred = combined_logits.argmax(axis=1)
    rows = evaluate(keys, labels, base_pred, domain, "baseline", "none")
    class_rows = class_metric_rows(keys, labels, base_pred, domain, "baseline", "none")

    for window in args.windows:
        if window <= 1 or window % 2 == 0:
            continue
        for mode in ("logit_mean", "prob_mean", "label_majority"):
            pred = apply_by_sequence(keys, combined_logits, base_pred, mode, window)
            rows.extend(evaluate(keys, labels, pred, domain, mode, str(window)))
            class_rows.extend(class_metric_rows(keys, labels, pred, domain, mode, str(window)))

    for alpha in args.ema_alphas:
        pred = apply_by_sequence(keys, combined_logits, base_pred, "logit_ema", alpha)
        rows.extend(evaluate(keys, labels, pred, domain, "logit_ema", str(alpha)))
        class_rows.extend(class_metric_rows(keys, labels, pred, domain, "logit_ema", str(alpha)))

    for margin in args.hysteresis_margins:
        pred = apply_by_sequence(keys, combined_logits, base_pred, "hysteresis", margin)
        rows.extend(evaluate(keys, labels, pred, domain, "hysteresis", str(margin)))
        class_rows.extend(class_metric_rows(keys, labels, pred, domain, "hysteresis", str(margin)))
    return rows, class_rows


def add_mean_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["domain"]), str(row["mode"]), str(row["param"]))].append(row)
    mean_rows = []
    for (domain, mode, param), group in sorted(grouped.items()):
        kappas = [float(row["kappa"]) for row in group]
        accuracies = [float(row["accuracy"]) for row in group]
        true_flips = sum(int(row["true_flips"]) for row in group)
        pred_flips = sum(int(row["pred_flips"]) for row in group)
        transitions = sum(int(row["n_transitions"]) for row in group)
        mean_rows.append({
            "domain": domain,
            "mode": mode,
            "param": param,
            "head": "mean",
            "n_frames": sum(int(row["n_frames"]) for row in group),
            "accuracy": float(np.mean(accuracies)),
            "kappa": float(np.mean(kappas)),
            "n_sequences": sum(int(row["n_sequences"]) for row in group),
            "n_transitions": transitions,
            "true_flips": true_flips,
            "pred_flips": pred_flips,
            "true_flip_rate": true_flips / transitions if transitions else float("nan"),
            "pred_flip_rate": pred_flips / transitions if transitions else float("nan"),
            "flip_rate_ratio": pred_flips / true_flips if true_flips else float("inf") if pred_flips else 0.0,
            "excess_flips": pred_flips - true_flips,
        })
    return rows + mean_rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    all_class_rows = []
    for domain in ("CC", "CR"):
        raw_domain_rows, class_rows = run_domain(domain, args)
        domain_rows = add_mean_rows(raw_domain_rows)
        write_csv(args.output_root / f"{domain.lower()}_smoothing_results.csv", domain_rows)
        write_csv(args.output_root / f"{domain.lower()}_class_metrics.csv", class_rows)
        rows.extend(domain_rows)
        all_class_rows.extend(class_rows)
    write_csv(args.output_root / "class_metrics.csv", all_class_rows)

    mean_by_setting: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["head"] == "mean":
            mean_by_setting[(str(row["mode"]), str(row["param"]), str(row["head"]))].append(row)
    combined = []
    for (mode, param, _head), group in sorted(mean_by_setting.items()):
        if len(group) != 2:
            continue
        combined.append({
            "mode": mode,
            "param": param,
            "cc_mean_kappa": next(float(row["kappa"]) for row in group if row["domain"] == "CC"),
            "cr_mean_kappa": next(float(row["kappa"]) for row in group if row["domain"] == "CR"),
            "combined_mean_kappa": float(np.mean([float(row["kappa"]) for row in group])),
            "cc_pred_flips": next(int(row["pred_flips"]) for row in group if row["domain"] == "CC"),
            "cr_pred_flips": next(int(row["pred_flips"]) for row in group if row["domain"] == "CR"),
            "total_pred_flips": sum(int(row["pred_flips"]) for row in group),
            "total_true_flips": sum(int(row["true_flips"]) for row in group),
        })
    combined.sort(key=lambda row: float(row["combined_mean_kappa"]), reverse=True)
    write_csv(args.output_root / "combined_smoothing_results.csv", combined)
    print(json.dumps(combined[:20], indent=2), flush=True)


if __name__ == "__main__":
    main()
