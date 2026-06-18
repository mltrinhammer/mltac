"""Constrained CR-social class-2 logit-bias search on MoE 1 two_head logits."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "MoE" / "experiments"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
ROOT = EXPERIMENT_ROOT / "moe1_cr_metadata_head_experts"
COMBINER = EXPERIMENT_ROOT / "moe1_cr_metadata_head_combiners" / "two_head" / "summary.json"
OUT = EXPERIMENT_ROOT / "moe1_cr_social_class2_bias_ablation"
N = 5


def read_scores(feature: str, split: str):
    run = ROOT / f"cr_{feature}_dyadic_tcn_k11_seed13"
    path = run / ("diagnostics/train_internal/val_prediction_scores.csv.gz" if split == "train" else "val_prediction_scores.csv.gz")
    rows = {}
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["head"] != "social":
                continue
            key = (row["domain"], row["source_split"], row["session_id"], row["role"], int(row["frame_idx"]))
            rows[key] = {
                "y": int(row["y_true"]),
                "logits": np.asarray([float(row[f"logit_{idx}"]) for idx in range(N)], dtype=np.float64),
            }
    return rows


def load(split: str):
    by = {feature: read_scores(feature, split) for feature in FEATURES}
    keys = sorted(set.intersection(*(set(v) for v in by.values())))
    y = np.asarray([by[FEATURES[0]][key]["y"] for key in keys], dtype=np.int64)
    logits = {feature: np.stack([by[feature][key]["logits"] for key in keys]) for feature in FEATURES}
    weights = np.asarray(json.loads(COMBINER.read_text())["weights_by_group"]["social"], dtype=np.float64)
    combined = sum(float(weight) * logits[feature] for weight, feature in zip(weights, FEATURES))
    return y, combined


def kappa(y, pred):
    cm = np.zeros((N, N), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    total = cm.sum()
    acc = np.trace(cm) / total
    exp = cm.sum(axis=0) @ cm.sum(axis=1) / (total * total)
    return float((acc - exp) / (1 - exp)) if exp < 1 else float("nan")


def class_stats(y, pred, klass=2):
    tp = int(((y == klass) & (pred == klass)).sum())
    fp = int(((y != klass) & (pred == klass)).sum())
    fn = int(((y == klass) & (pred != klass)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall and not np.isnan(precision) and not np.isnan(recall) else float("nan")
    return precision, recall, f1, tp, fp, fn


def evaluate(y, logits, bias2):
    adjusted = logits.copy()
    adjusted[:, 2] += bias2
    pred = adjusted.argmax(axis=1)
    p, r, f1, tp, fp, fn = class_stats(y, pred, 2)
    return {
        "kappa": kappa(y, pred),
        "accuracy": float((y == pred).mean()),
        "class2_precision": p,
        "class2_recall": r,
        "class2_f1": f1,
        "class2_predicted": int((pred == 2).sum()),
        "class2_tp": tp,
        "class2_fp": fp,
        "class2_fn": fn,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_y, train_logits = load("train")
    val_y, val_logits = load("val")
    rows = []
    for bias2 in np.round(np.arange(-2.0, 4.0001, 0.05), 4):
        train = evaluate(train_y, train_logits, float(bias2))
        val = evaluate(val_y, val_logits, float(bias2))
        row = {"bias2": float(bias2)}
        row.update({f"train_{k}": v for k, v in train.items()})
        row.update({f"val_{k}": v for k, v in val.items()})
        rows.append(row)
    write_csv(OUT / "class2_bias_grid.csv", rows)
    for criterion in ("train_kappa", "train_class2_f1", "train_accuracy"):
        best = max(rows, key=lambda row: float(row[criterion]))
        print(criterion, json.dumps(best, indent=2), flush=True)


if __name__ == "__main__":
    main()
