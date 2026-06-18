"""Fit CR two_head combiner with weighted CR-social CE variants."""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "MoE" / "experiments"
ROOT = EXPERIMENT_ROOT / "moe1_cr_metadata_head_experts"
OUT = EXPERIMENT_ROOT / "moe1_cr_social_weighted_two_head_combiner"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
CLASS_COUNTS = {"task": 4, "social": 5}
MAX_CLASSES = 5
Key = tuple[str, str, str, str, str, int]


def score_path(feature: str, split: str) -> Path:
    run = ROOT / f"cr_{feature}_dyadic_tcn_k11_seed13"
    if split == "train":
        return run / "diagnostics" / "train_internal" / "val_prediction_scores.csv.gz"
    return run / "val_prediction_scores.csv.gz"


def read_scores(feature: str, split: str):
    rows = {}
    with gzip.open(score_path(feature, split), "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            head = row["head"]
            n = CLASS_COUNTS[head]
            key = (row["domain"], row["source_split"], row["session_id"], row["role"], head, int(row["frame_idx"]))
            logits = np.full(MAX_CLASSES, -1e9, dtype=np.float64)
            logits[:n] = [float(row[f"logit_{idx}"]) for idx in range(n)]
            rows[key] = {"y": int(row["y_true"]), "logits": logits}
    return rows


def load(split: str):
    by = {feature: read_scores(feature, split) for feature in FEATURES}
    keys = sorted(set.intersection(*(set(v) for v in by.values())))
    y = np.asarray([by[FEATURES[0]][key]["y"] for key in keys], dtype=np.int64)
    logits = {feature: np.stack([by[feature][key]["logits"] for key in keys]) for feature in FEATURES}
    return keys, y, logits


def simplex(step=0.05):
    n = round(1 / step)
    for a in range(n + 1):
        for b in range(n + 1 - a):
            yield np.asarray([a, b, n - a - b], dtype=np.float64) / n


def combine(logits, weights):
    return sum(float(w) * logits[f] for w, f in zip(weights, FEATURES))


def weighted_ce(logits, y, n_classes, class_weights=None):
    z = logits[:, :n_classes]
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    losses = -logp[np.arange(len(y)), y]
    if class_weights is None:
        return float(losses.mean())
    w = class_weights[y]
    return float((losses * w).sum() / w.sum())


def fit(train_keys, train_y, train_logits, social_class_weights):
    out = {}
    losses = {}
    grid = list(simplex())
    for head in ("task", "social"):
        idx = np.asarray([i for i, key in enumerate(train_keys) if key[4] == head], dtype=np.int64)
        best_w = grid[0]
        best_loss = float("inf")
        n = CLASS_COUNTS[head]
        cw = social_class_weights if head == "social" else None
        group_logits = {f: v[idx] for f, v in train_logits.items()}
        group_y = train_y[idx]
        for weights in grid:
            loss = weighted_ce(combine(group_logits, weights), group_y, n, cw)
            if loss < best_loss:
                best_loss = loss
                best_w = weights
        out[head] = best_w
        losses[head] = best_loss
    return out, losses


def kappa(y, p, n):
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    total = cm.sum(); acc = np.trace(cm) / total; exp = cm.sum(0) @ cm.sum(1) / (total * total)
    return float((acc - exp) / (1 - exp)) if exp < 1 else float("nan")


def class2_stats(y, p):
    tp = int(((y == 2) & (p == 2)).sum()); fp = int(((y != 2) & (p == 2)).sum()); fn = int(((y == 2) & (p != 2)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan"); rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec and not math.isnan(prec) and not math.isnan(rec) else float("nan")
    return prec, rec, f1, tp, fp, fn


def evaluate(keys, y, logits, weights_by_head):
    pred = np.empty(len(keys), dtype=np.int64)
    rows = []
    kappas = []
    for head in ("task", "social"):
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        pred[idx] = combine({f: v[idx] for f, v in logits.items()}, weights_by_head[head])[:, :CLASS_COUNTS[head]].argmax(axis=1)
        kk = kappa(y[idx], pred[idx], CLASS_COUNTS[head]); kappas.append(kk)
        row = {"head": head, "kappa": kk, "accuracy": float((y[idx] == pred[idx]).mean()), "n": int(len(idx))}
        if head == "social":
            prec, rec, f1, tp, fp, fn = class2_stats(y[idx], pred[idx])
            row |= {"class2_precision": prec, "class2_recall": rec, "class2_f1": f1, "class2_tp": tp, "class2_fp": fp, "class2_fn": fn, "class2_predicted": int((pred[idx] == 2).sum())}
        rows.append(row)
    rows.append({"head": "mean", "kappa": float(np.mean(kappas)), "accuracy": float(np.mean([r["accuracy"] for r in rows])), "n": int(len(keys))})
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    train_keys, train_y, train_logits = load("train")
    val_keys, val_y, val_logits = load("val")
    all_rows = []
    summaries = []
    for name, cw in {
        "baseline_ce": np.ones(5),
        "class2x2_ce": np.asarray([1, 1, 2, 1, 1], dtype=float),
        "class2x4_ce": np.asarray([1, 1, 4, 1, 1], dtype=float),
        "class2x8_ce": np.asarray([1, 1, 8, 1, 1], dtype=float),
        "high_ge2_x2_ce": np.asarray([1, 1, 2, 2, 1], dtype=float),
        "high_ge2_x4_ce": np.asarray([1, 1, 4, 4, 1], dtype=float),
    }.items():
        weights, losses = fit(train_keys, train_y, train_logits, cw)
        rows = evaluate(val_keys, val_y, val_logits, weights)
        for row in rows:
            row = {"mode": name, **row}
            all_rows.append(row)
        summaries.append({"mode": name, "weights": {h: w.tolist() for h, w in weights.items()}, "losses": losses})
        print(name, summaries[-1], rows, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "val_metrics.csv", all_rows)
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
