"""Class-prior and class-bias calibration ablations for selected MoE 1.

Selected base model: metadata-head modality experts + logit-space two_head combiner.
The script uses frozen train-internal and validation expert logits. It evaluates:
- prior correction using train priors and several target priors
- class-bias calibration fit on train-internal logits
- optional hysteresis after calibration
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "MoE" / "experiments"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "moe1_class_calibration_ablation"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
CLASS_COUNTS = {"task": 4, "social": 5}
HEADS = ("task", "social")
MAX_CLASSES = max(CLASS_COUNTS.values())
Key = tuple[str, str, str, str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run class calibration ablations for MoE 1.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bias-lr", type=float, default=0.1)
    parser.add_argument("--bias-epochs", type=int, default=800)
    parser.add_argument("--bias-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-fit-rows", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--hysteresis-margin", type=float, default=1.0)
    return parser.parse_args()


def expert_root(domain: str) -> Path:
    return EXPERIMENT_ROOT / f"moe1_{domain.lower()}_metadata_head_experts"


def score_path(domain: str, feature: str, split: str) -> Path:
    run = expert_root(domain) / f"{domain.lower()}_{feature}_dyadic_tcn_k11_seed13"
    if split == "train":
        return run / "diagnostics" / "train_internal" / "val_prediction_scores.csv.gz"
    if split == "val":
        return run / "val_prediction_scores.csv.gz"
    raise ValueError(split)


def combiner_summary_path(domain: str) -> Path:
    return EXPERIMENT_ROOT / f"moe1_{domain.lower()}_metadata_head_combiners" / "two_head" / "summary.json"


def read_scores(path: Path) -> dict[Key, dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: dict[Key, dict[str, object]] = {}
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


def read_domain_data(domain: str, split: str) -> tuple[list[Key], np.ndarray, dict[str, np.ndarray]]:
    by_feature = {feature: read_scores(score_path(domain, feature, split)) for feature in FEATURES}
    keys = sorted(set.intersection(*(set(rows) for rows in by_feature.values())))
    labels = np.asarray([by_feature[FEATURES[0]][key]["y_true"] for key in keys], dtype=np.int64)
    logits = {
        feature: np.stack([by_feature[feature][key]["logits"] for key in keys])
        for feature in FEATURES
    }
    return keys, labels, logits


def read_weights(domain: str) -> dict[str, np.ndarray]:
    summary = json.loads(combiner_summary_path(domain).read_text(encoding="utf-8"))
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


def kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    n = confusion.sum()
    if n == 0:
        return float("nan")
    accuracy = np.trace(confusion) / n
    expected = confusion.sum(axis=1) @ confusion.sum(axis=0) / (n * n)
    return float((accuracy - expected) / (1.0 - expected)) if expected < 1.0 else float("nan")


def groups_for_sequences(keys: list[Key]) -> dict[tuple[str, str, str], list[int]]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        _domain, _source_split, session_id, role, head, _frame_idx = key
        groups[(session_id, role, head)].append(idx)
    return {group: sorted(idxs, key=lambda idx: keys[idx][5]) for group, idxs in groups.items()}


def hysteresis_by_sequence(keys: list[Key], logits: np.ndarray, margin: float) -> np.ndarray:
    out = np.empty(len(keys), dtype=np.int64)
    for (_session, _role, head), idxs in groups_for_sequences(keys).items():
        n_classes = CLASS_COUNTS[head]
        seq_logits = logits[idxs, :n_classes]
        pred = seq_logits.argmax(axis=1)
        current = int(pred[0])
        out[idxs[0]] = current
        for pos, idx in enumerate(idxs[1:], start=1):
            candidate = int(pred[pos])
            if candidate != current and seq_logits[pos, candidate] >= seq_logits[pos, current] + margin:
                current = candidate
            out[idx] = current
    return out


def transition_stats(keys: list[Key], labels: np.ndarray, pred: np.ndarray, head: str) -> dict[str, object]:
    true_flips = 0
    pred_flips = 0
    transitions = 0
    for (_session, _role, group_head), idxs in groups_for_sequences(keys).items():
        if group_head != head or len(idxs) < 2:
            continue
        y_true = labels[idxs]
        y_pred = pred[idxs]
        true_flips += int((y_true[1:] != y_true[:-1]).sum())
        pred_flips += int((y_pred[1:] != y_pred[:-1]).sum())
        transitions += len(idxs) - 1
    return {
        "n_transitions": transitions,
        "true_flips": true_flips,
        "pred_flips": pred_flips,
        "flip_rate_ratio": pred_flips / true_flips if true_flips else float("inf") if pred_flips else 0.0,
    }


def class_rows(keys: list[Key], labels: np.ndarray, pred: np.ndarray, domain: str, mode: str) -> list[dict[str, object]]:
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
            f1 = 2 * precision * recall / (precision + recall) if precision + recall and not math.isnan(precision) and not math.isnan(recall) else float("nan")
            rows.append({
                "domain": domain,
                "mode": mode,
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


def metric_rows(keys: list[Key], labels: np.ndarray, pred: np.ndarray, domain: str, mode: str) -> list[dict[str, object]]:
    rows = []
    kappas = []
    for head in HEADS:
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        score = kappa(labels[idx], pred[idx], CLASS_COUNTS[head])
        kappas.append(score)
        rows.append({
            "domain": domain,
            "mode": mode,
            "head": head,
            "n_frames": len(idx),
            "accuracy": float((labels[idx] == pred[idx]).mean()),
            "kappa": score,
            **transition_stats(keys, labels, pred, head),
        })
    rows.append({
        "domain": domain,
        "mode": mode,
        "head": "mean",
        "n_frames": len(keys),
        "accuracy": float(np.mean([row["accuracy"] for row in rows])),
        "kappa": float(np.nanmean(kappas)),
        "n_transitions": sum(int(row["n_transitions"]) for row in rows),
        "true_flips": sum(int(row["true_flips"]) for row in rows),
        "pred_flips": sum(int(row["pred_flips"]) for row in rows),
        "flip_rate_ratio": sum(int(row["pred_flips"]) for row in rows) / sum(int(row["true_flips"]) for row in rows),
    })
    return rows


def class_prior(labels: np.ndarray, n_classes: int, smoothing: float = 1.0) -> np.ndarray:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64) + smoothing
    return counts / counts.sum()


def apply_prior_correction(
    logits: np.ndarray,
    keys: list[Key],
    train_priors: dict[str, np.ndarray],
    target_priors: dict[str, np.ndarray],
) -> np.ndarray:
    corrected = logits.copy()
    for head in HEADS:
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        n_classes = CLASS_COUNTS[head]
        offset = np.log(target_priors[head]) - np.log(train_priors[head])
        corrected[idx, :n_classes] += offset
    return corrected


def fit_bias(
    logits: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    balanced: bool,
    lr: float,
    epochs: int,
    weight_decay: float,
) -> np.ndarray:
    bias = np.zeros(n_classes, dtype=np.float64)
    if balanced:
        counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
        weights = np.where(counts > 0, len(labels) / (n_classes * counts), 0.0)
        sample_weights = weights[labels]
        sample_weights = sample_weights / sample_weights.mean()
    else:
        sample_weights = np.ones(len(labels), dtype=np.float64)
    for _epoch in range(epochs):
        probs = softmax(logits[:, :n_classes] + bias)
        target = np.zeros_like(probs)
        target[np.arange(len(labels)), labels] = 1.0
        grad = ((probs - target) * sample_weights[:, None]).mean(axis=0)
        if weight_decay:
            grad += weight_decay * bias
        bias -= lr * grad
        bias -= bias.mean()
    return bias


def fit_biases_by_head(
    keys: list[Key],
    labels: np.ndarray,
    logits: np.ndarray,
    args: argparse.Namespace,
    balanced: bool,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    biases = {}
    for head in HEADS:
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        if args.max_fit_rows > 0 and len(idx) > args.max_fit_rows:
            idx = np.sort(rng.choice(idx, size=args.max_fit_rows, replace=False))
        biases[head] = fit_bias(
            logits[idx],
            labels[idx],
            CLASS_COUNTS[head],
            balanced,
            args.bias_lr,
            args.bias_epochs,
            args.bias_weight_decay,
        )
    return biases


def apply_biases(logits: np.ndarray, keys: list[Key], biases: dict[str, np.ndarray]) -> np.ndarray:
    corrected = logits.copy()
    for head, bias in biases.items():
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        corrected[idx, : CLASS_COUNTS[head]] += bias
    return corrected


def priors_by_head(keys: list[Key], labels: np.ndarray, target: str = "observed") -> dict[str, np.ndarray]:
    priors = {}
    for head in HEADS:
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        n_classes = CLASS_COUNTS[head]
        if target == "uniform":
            priors[head] = np.ones(n_classes, dtype=np.float64) / n_classes
        else:
            priors[head] = class_prior(labels[idx], n_classes)
    return priors


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


def evaluate_logits(keys: list[Key], labels: np.ndarray, logits: np.ndarray, domain: str, mode: str, rows: list[dict[str, object]], cls_rows: list[dict[str, object]]) -> None:
    pred = logits.argmax(axis=1)
    rows.extend(metric_rows(keys, labels, pred, domain, mode))
    cls_rows.extend(class_rows(keys, labels, pred, domain, mode))


def evaluate_logits_hysteresis(keys: list[Key], labels: np.ndarray, logits: np.ndarray, domain: str, mode: str, margin: float, rows: list[dict[str, object]], cls_rows: list[dict[str, object]]) -> None:
    pred = hysteresis_by_sequence(keys, logits, margin)
    rows.extend(metric_rows(keys, labels, pred, domain, f"{mode}+hysteresis_{margin:g}"))
    cls_rows.extend(class_rows(keys, labels, pred, domain, f"{mode}+hysteresis_{margin:g}"))


def run_domain(domain: str, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    weights = read_weights(domain)
    train_keys, train_labels, train_feature_logits = read_domain_data(domain, "train")
    val_keys, val_labels, val_feature_logits = read_domain_data(domain, "val")
    train_logits = combine_two_head(train_keys, train_feature_logits, weights)
    val_logits = combine_two_head(val_keys, val_feature_logits, weights)

    rows: list[dict[str, object]] = []
    cls_rows: list[dict[str, object]] = []
    evaluate_logits(val_keys, val_labels, val_logits, domain, "baseline", rows, cls_rows)
    evaluate_logits_hysteresis(val_keys, val_labels, val_logits, domain, "baseline", args.hysteresis_margin, rows, cls_rows)

    train_priors = priors_by_head(train_keys, train_labels)
    val_priors = priors_by_head(val_keys, val_labels)
    uniform_priors = priors_by_head(train_keys, train_labels, target="uniform")
    sqrt_uniform_priors = {head: np.sqrt(train_priors[head] * uniform_priors[head]) for head in HEADS}
    sqrt_uniform_priors = {head: prior / prior.sum() for head, prior in sqrt_uniform_priors.items()}
    target_sets = {
        "prior_uniform": uniform_priors,
        "prior_sqrt_uniform": sqrt_uniform_priors,
        "prior_val_oracle": val_priors,
    }
    for mode, target_priors in target_sets.items():
        corrected = apply_prior_correction(val_logits, val_keys, train_priors, target_priors)
        evaluate_logits(val_keys, val_labels, corrected, domain, mode, rows, cls_rows)
        evaluate_logits_hysteresis(val_keys, val_labels, corrected, domain, mode, args.hysteresis_margin, rows, cls_rows)

    for balanced in (False, True):
        mode = "bias_balanced_ce" if balanced else "bias_ce"
        biases = fit_biases_by_head(train_keys, train_labels, train_logits, args, balanced)
        corrected = apply_biases(val_logits, val_keys, biases)
        evaluate_logits(val_keys, val_labels, corrected, domain, mode, rows, cls_rows)
        evaluate_logits_hysteresis(val_keys, val_labels, corrected, domain, mode, args.hysteresis_margin, rows, cls_rows)
        for head, bias in biases.items():
            rows.append({
                "domain": domain,
                "mode": mode,
                "head": f"{head}_bias",
                "bias": json.dumps(bias.tolist()),
            })

    summary = {
        "domain": domain,
        "train_priors": {head: train_priors[head].tolist() for head in HEADS},
        "val_priors_oracle": {head: val_priors[head].tolist() for head in HEADS},
    }
    return rows, cls_rows, summary


def combined_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("head") == "mean":
            grouped[str(row["mode"])].append(row)
    output = []
    for mode, group in sorted(grouped.items()):
        if len(group) != 2:
            continue
        cc = next(row for row in group if row["domain"] == "CC")
        cr = next(row for row in group if row["domain"] == "CR")
        output.append({
            "mode": mode,
            "cc_mean_kappa": cc["kappa"],
            "cr_mean_kappa": cr["kappa"],
            "combined_mean_kappa": float(np.mean([float(cc["kappa"]), float(cr["kappa"])])),
            "cc_pred_flips": cc["pred_flips"],
            "cr_pred_flips": cr["pred_flips"],
        })
    output.sort(key=lambda row: float(row["combined_mean_kappa"]), reverse=True)
    return output


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    all_cls_rows: list[dict[str, object]] = []
    summaries = []
    for domain in ("CC", "CR"):
        rows, cls_rows, summary = run_domain(domain, args)
        write_csv(args.output_root / f"{domain.lower()}_metrics.csv", rows)
        write_csv(args.output_root / f"{domain.lower()}_class_metrics.csv", cls_rows)
        all_rows.extend(rows)
        all_cls_rows.extend(cls_rows)
        summaries.append(summary)
    write_csv(args.output_root / "metrics.csv", all_rows)
    write_csv(args.output_root / "class_metrics.csv", all_cls_rows)
    combined = combined_rows(all_rows)
    write_csv(args.output_root / "combined_results.csv", combined)
    (args.output_root / "summary.json").write_text(json.dumps({"domains": summaries, "top_results": combined[:20]}, indent=2), encoding="utf-8")
    print(json.dumps(combined[:20], indent=2), flush=True)


if __name__ == "__main__":
    main()
