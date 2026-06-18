"""Error analysis for the selected PinSoRo MoE 1 model.

Selected model: metadata-head modality experts + logit-space two_head combiner.
The script reconstructs validation predictions from exported expert logits and
writes granular CSV reports under ACM/MoE/experiments/.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "MoE" / "experiments"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "moe1_error_analysis_metadata_head_two_head"
DEFAULT_METADATA = PROJECT_ROOT / "MoE" / "moe_data" / "outputs" / "participant_metadata.csv"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
CLASS_COUNTS = {"task": 4, "social": 5}
HEADS = ("task", "social")
MAX_CLASSES = max(CLASS_COUNTS.values())
Key = tuple[str, str, str, str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze selected MoE 1 validation errors.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--top-errors", type=int, default=1000)
    return parser.parse_args()


def expert_root(domain: str) -> Path:
    return EXPERIMENT_ROOT / f"moe1_{domain.lower()}_metadata_head_experts"


def combiner_summary_path(domain: str) -> Path:
    return EXPERIMENT_ROOT / f"moe1_{domain.lower()}_metadata_head_combiners" / "two_head" / "summary.json"


def score_path(domain: str, feature: str) -> Path:
    run = expert_root(domain) / f"{domain.lower()}_{feature}_dyadic_tcn_k11_seed13"
    return run / "val_prediction_scores.csv.gz"


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


def read_domain_data(domain: str) -> tuple[list[Key], np.ndarray, dict[str, np.ndarray]]:
    by_feature = {feature: read_scores(score_path(domain, feature)) for feature in FEATURES}
    keys = sorted(set.intersection(*(set(rows) for rows in by_feature.values())))
    if not keys:
        raise RuntimeError(f"No aligned validation rows for {domain}.")
    labels = np.asarray([by_feature[FEATURES[0]][key]["y_true"] for key in keys], dtype=np.int64)
    logits = {
        feature: np.stack([by_feature[feature][key]["logits"] for key in keys])
        for feature in FEATURES
    }
    return keys, labels, logits


def read_weights(domain: str) -> dict[str, np.ndarray]:
    summary = json.loads(combiner_summary_path(domain).read_text(encoding="utf-8"))
    return {head: np.asarray(weights, dtype=np.float64) for head, weights in summary["weights_by_group"].items()}


def read_metadata(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    table: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            table[(row["source_split"], row["session_id"], row["role"])] = {
                "age": row.get("age", ""),
                "gender": row.get("gender", ""),
            }
    return table


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def combine_two_head(keys: list[Key], logits: dict[str, np.ndarray], weights: dict[str, np.ndarray]) -> np.ndarray:
    combined = np.empty((len(keys), MAX_CLASSES), dtype=np.float64)
    for head in HEADS:
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        head_weights = weights[head]
        combined[idx] = sum(float(weight) * logits[feature][idx] for weight, feature in zip(head_weights, FEATURES))
    return combined


def kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    n = confusion.sum()
    if n == 0:
        return float("nan")
    accuracy = np.trace(confusion) / n
    expected = confusion.sum(axis=1) @ confusion.sum(axis=0) / (n * n)
    return float((accuracy - expected) / (1.0 - expected)) if expected < 1.0 else float("nan")


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, object]:
    if len(y_true) == 0:
        return {"n_frames": 0, "accuracy": float("nan"), "kappa": float("nan")}
    return {
        "n_frames": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "kappa": kappa(y_true, y_pred, n_classes),
    }


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, klass: int) -> tuple[float, float, float, int, int, int]:
    tp = int(((y_true == klass) & (y_pred == klass)).sum())
    fp = int(((y_true != klass) & (y_pred == klass)).sum())
    fn = int(((y_true == klass) & (y_pred != klass)).sum())
    support = int((y_true == klass).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall and not math.isnan(precision) and not math.isnan(recall) else float("nan")
    return precision, recall, f1, support, tp, fp


def age_bin(age_text: str) -> str:
    if age_text == "":
        return "missing"
    age = float(age_text)
    if age <= 5:
        return "<=5"
    if age <= 7:
        return "6-7"
    return ">=8"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_rows(
    keys: list[Key],
    labels: np.ndarray,
    pred: np.ndarray,
    group_name: str,
    group_fn: Callable[[Key], str],
) -> list[dict[str, object]]:
    groups = sorted({group_fn(key) for key in keys})
    rows = []
    for group in groups:
        idx = np.asarray([i for i, key in enumerate(keys) if group_fn(key) == group], dtype=np.int64)
        head_values = {keys[i][4] for i in idx}
        if len(head_values) == 1:
            n_classes = CLASS_COUNTS[next(iter(head_values))]
        else:
            n_classes = MAX_CLASSES
        rows.append({group_name: group, **metric_dict(labels[idx], pred[idx], n_classes)})
    return rows


def confusion_rows(keys: list[Key], labels: np.ndarray, pred: np.ndarray, domain: str, head: str) -> list[dict[str, object]]:
    idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
    n_classes = CLASS_COUNTS[head]
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (labels[idx], pred[idx]), 1)
    rows = []
    for true_class in range(n_classes):
        row: dict[str, object] = {"domain": domain, "head": head, "true_class": true_class}
        for pred_class in range(n_classes):
            row[f"pred_{pred_class}"] = int(matrix[true_class, pred_class])
        rows.append(row)
    return rows



def transition_rows(keys: list[Key], labels: np.ndarray, pred: np.ndarray, domain: str) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        _domain, _source_split, session_id, role, head, _frame_idx = key
        groups[(session_id, role, head)].append(idx)

    rows = []
    aggregate: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {
        "n_sequences": 0,
        "n_transitions": 0,
        "true_flips": 0,
        "pred_flips": 0,
        "excess_flips": 0,
        "abs_flip_gap": 0,
    })
    for (session_id, role, head), idxs in sorted(groups.items()):
        idxs = sorted(idxs, key=lambda idx: keys[idx][5])
        if len(idxs) < 2:
            continue
        y_true = labels[idxs]
        y_pred = pred[idxs]
        true_flips = int((y_true[1:] != y_true[:-1]).sum())
        pred_flips = int((y_pred[1:] != y_pred[:-1]).sum())
        n_transitions = len(idxs) - 1
        row = {
            "domain": domain,
            "session_id": session_id,
            "role": role,
            "head": head,
            "n_frames": len(idxs),
            "n_transitions": n_transitions,
            "true_flips": true_flips,
            "pred_flips": pred_flips,
            "true_flip_rate": true_flips / n_transitions,
            "pred_flip_rate": pred_flips / n_transitions,
            "flip_rate_ratio": pred_flips / true_flips if true_flips else float("inf") if pred_flips else 0.0,
            "excess_flips": pred_flips - true_flips,
        }
        rows.append(row)
        agg = aggregate[(domain, head)]
        agg["n_sequences"] += 1
        agg["n_transitions"] += n_transitions
        agg["true_flips"] += true_flips
        agg["pred_flips"] += pred_flips
        agg["excess_flips"] += pred_flips - true_flips
        agg["abs_flip_gap"] += abs(pred_flips - true_flips)

    summary_rows = []
    for (agg_domain, head), values in sorted(aggregate.items()):
        true_flips = values["true_flips"]
        pred_flips = values["pred_flips"]
        transitions = values["n_transitions"]
        summary_rows.append({
            "domain": agg_domain,
            "head": head,
            "n_sequences": int(values["n_sequences"]),
            "n_transitions": int(transitions),
            "true_flips": int(true_flips),
            "pred_flips": int(pred_flips),
            "true_flip_rate": true_flips / transitions if transitions else float("nan"),
            "pred_flip_rate": pred_flips / transitions if transitions else float("nan"),
            "flip_rate_ratio": pred_flips / true_flips if true_flips else float("inf") if pred_flips else 0.0,
            "excess_flips": int(values["excess_flips"]),
            "mean_abs_flip_gap_per_sequence": values["abs_flip_gap"] / values["n_sequences"] if values["n_sequences"] else float("nan"),
        })
    return rows + [{"__summary_marker__": "summary_follows"}] + summary_rows

def analyze_domain(domain: str, metadata: dict[tuple[str, str, str], dict[str, str]], output_root: Path, top_errors: int) -> dict[str, object]:
    keys, labels, logits = read_domain_data(domain)
    weights = read_weights(domain)
    combined_logits = combine_two_head(keys, logits, weights)
    pred = combined_logits.argmax(axis=1)
    probs = softmax(combined_logits)
    confidence = probs[np.arange(len(pred)), pred]
    expert_pred = {feature: logits[feature].argmax(axis=1) for feature in FEATURES}

    rows = []
    rows.extend(group_rows(keys, labels, pred, "domain_head_role", lambda key: f"{key[0]}_{key[4]}_{key[3]}"))
    write_csv(output_root / "metrics_by_domain_head_role.csv", rows)

    class_rows = []
    for head in HEADS:
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        for klass in range(CLASS_COUNTS[head]):
            precision, recall, f1, support, tp, fp = precision_recall_f1(labels[idx], pred[idx], klass)
            class_rows.append({
                "domain": domain,
                "head": head,
                "class": klass,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positive": tp,
                "false_positive": fp,
            })
    write_csv(output_root / "metrics_by_domain_head_class.csv", class_rows)

    for head in HEADS:
        write_csv(output_root / f"confusion_{domain}_{head}.csv", confusion_rows(keys, labels, pred, domain, head))

    expert_rows = []
    for head in HEADS:
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        n_classes = CLASS_COUNTS[head]
        expert_rows.append({"domain": domain, "head": head, "model": "combined_two_head", **metric_dict(labels[idx], pred[idx], n_classes)})
        for feature in FEATURES:
            expert_rows.append({"domain": domain, "head": head, "model": feature, **metric_dict(labels[idx], expert_pred[feature][idx], n_classes)})
    write_csv(output_root / "expert_vs_combiner_by_head.csv", expert_rows)

    session_rows = []
    for session in sorted({key[2] for key in keys}):
        for head in HEADS:
            idx = np.asarray([i for i, key in enumerate(keys) if key[2] == session and key[4] == head], dtype=np.int64)
            if len(idx):
                session_rows.append({"domain": domain, "session_id": session, "head": head, **metric_dict(labels[idx], pred[idx], CLASS_COUNTS[head])})
    write_csv(output_root / "session_metrics.csv", session_rows)

    metadata_rows = []
    group_to_indices: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        meta = metadata.get((key[1], key[2], key[3]), {"age": "", "gender": ""})
        group_to_indices[(key[4], age_bin(meta["age"]), meta["gender"] or "missing")].append(idx)
    for (head, age_group, gender), idxs in sorted(group_to_indices.items()):
        idx = np.asarray(idxs, dtype=np.int64)
        metadata_rows.append({
            "domain": domain,
            "head": head,
            "age_bin": age_group,
            "gender": gender,
            **metric_dict(labels[idx], pred[idx], CLASS_COUNTS[head]),
        })
    write_csv(output_root / "metadata_bin_metrics.csv", metadata_rows)

    transition_output = transition_rows(keys, labels, pred, domain)
    marker_idx = next(i for i, row in enumerate(transition_output) if row.get("__summary_marker__") == "summary_follows")
    write_csv(output_root / "transition_metrics_by_sequence.csv", transition_output[:marker_idx])
    write_csv(output_root / "transition_metrics_summary.csv", transition_output[marker_idx + 1 :])

    error_rows = []
    wrong_idx = np.asarray([i for i in range(len(keys)) if labels[i] != pred[i]], dtype=np.int64)
    wrong_idx = wrong_idx[np.argsort(-confidence[wrong_idx])][:top_errors]
    for i in wrong_idx:
        key = keys[i]
        meta = metadata.get((key[1], key[2], key[3]), {"age": "", "gender": ""})
        row = {
            "domain": key[0],
            "source_split": key[1],
            "session_id": key[2],
            "role": key[3],
            "head": key[4],
            "frame_idx": key[5],
            "y_true": int(labels[i]),
            "y_pred": int(pred[i]),
            "confidence": float(confidence[i]),
            "age": meta["age"],
            "gender": meta["gender"],
        }
        for feature in FEATURES:
            row[f"{feature}_pred"] = int(expert_pred[feature][i])
        error_rows.append(row)
    write_csv(output_root / "high_confidence_errors.csv", error_rows)

    disagreement_rows = []
    disagreement_idx = []
    for i in range(len(keys)):
        preds = [int(expert_pred[feature][i]) for feature in FEATURES]
        if len(set(preds)) > 1 and labels[i] != pred[i]:
            disagreement_idx.append(i)
    disagreement_idx = sorted(disagreement_idx, key=lambda i: (-len({int(expert_pred[f][i]) for f in FEATURES}), -confidence[i]))[:top_errors]
    for i in disagreement_idx:
        key = keys[i]
        meta = metadata.get((key[1], key[2], key[3]), {"age": "", "gender": ""})
        row = {
            "domain": key[0],
            "source_split": key[1],
            "session_id": key[2],
            "role": key[3],
            "head": key[4],
            "frame_idx": key[5],
            "y_true": int(labels[i]),
            "combined_pred": int(pred[i]),
            "combined_confidence": float(confidence[i]),
            "age": meta["age"],
            "gender": meta["gender"],
        }
        for feature in FEATURES:
            row[f"{feature}_pred"] = int(expert_pred[feature][i])
        disagreement_rows.append(row)
    write_csv(output_root / "expert_disagreement_errors.csv", disagreement_rows)

    head_scores = []
    for head in HEADS:
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        head_scores.append(metric_dict(labels[idx], pred[idx], CLASS_COUNTS[head])["kappa"])
    return {
        "domain": domain,
        "n_rows": len(keys),
        "weights_by_head": {head: weights[head].tolist() for head in HEADS},
        "mean_head_kappa": float(np.nanmean(head_scores)),
        "task_kappa": float(head_scores[0]),
        "social_kappa": float(head_scores[1]),
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata(args.metadata)
    summaries = [analyze_domain(domain, metadata, output_root / domain.lower(), args.top_errors) for domain in ("CC", "CR")]

    combined_summary = {
        "selected_model": "metadata-head experts + logit two_head combiner",
        "domains": summaries,
        "combined_mean_head_kappa": float(np.mean([summary["mean_head_kappa"] for summary in summaries])),
        "output_root": str(output_root),
    }
    (output_root / "summary.json").write_text(json.dumps(combined_summary, indent=2), encoding="utf-8")

    overview_rows = []
    for summary in summaries:
        overview_rows.append({
            "domain": summary["domain"],
            "n_rows": summary["n_rows"],
            "task_kappa": summary["task_kappa"],
            "social_kappa": summary["social_kappa"],
            "mean_head_kappa": summary["mean_head_kappa"],
        })
    write_csv(output_root / "overview.csv", overview_rows)
    print(json.dumps(combined_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
