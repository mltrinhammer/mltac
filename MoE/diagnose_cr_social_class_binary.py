"""Diagnose whether one CR social class is separable from frozen MoE expert logits.

This script trains simple binary classifiers on train-internal CR social rows:
target class vs not target class. It evaluates on validation and writes threshold,
session, and score-distribution diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "MoE" / "experiments"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
SOCIAL_CLASSES = 5
DEFAULT_EXPERT_ROOT = EXPERIMENT_ROOT / "moe1_cr_metadata_head_experts"
DEFAULT_OUTPUT = None
Key = tuple[str, str, str, str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary target-class diagnostic from frozen CR social expert logits.")
    parser.add_argument("--target-class", type=int, default=2)
    parser.add_argument("--expert-root", type=Path, default=DEFAULT_EXPERT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-train-negatives", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if not 0 <= args.target_class < SOCIAL_CLASSES:
        raise ValueError("target-class must be in [0, 4].")
    if args.output_root is None:
        args.output_root = EXPERIMENT_ROOT / f"moe1_cr_social_class{args.target_class}_binary_diagnostic"
    return args


def score_path(root: Path, feature: str, split: str) -> Path:
    run = root / f"cr_{feature}_dyadic_tcn_k11_seed13"
    if split == "train":
        return run / "diagnostics" / "train_internal" / "val_prediction_scores.csv.gz"
    if split == "val":
        return run / "val_prediction_scores.csv.gz"
    raise ValueError(split)


def read_feature_scores(path: Path) -> dict[Key, dict[str, object]]:
    rows: dict[Key, dict[str, object]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["head"] != "social":
                continue
            key = (
                row["domain"],
                row["source_split"],
                row["session_id"],
                row["role"],
                row["head"],
                int(row["frame_idx"]),
            )
            logits = np.asarray([float(row[f"logit_{idx}"]) for idx in range(SOCIAL_CLASSES)], dtype=np.float64)
            probs = np.asarray([float(row[f"prob_{idx}"]) for idx in range(SOCIAL_CLASSES)], dtype=np.float64)
            rows[key] = {"y_true": int(row["y_true"]), "logits": logits, "probs": probs}
    return rows


def load_split(root: Path, split: str, feature_kind: str) -> tuple[list[Key], np.ndarray, np.ndarray]:
    by_feature = {feature: read_feature_scores(score_path(root, feature, split)) for feature in FEATURES}
    keys = sorted(set.intersection(*(set(rows) for rows in by_feature.values())))
    y_multi = np.asarray([by_feature[FEATURES[0]][key]["y_true"] for key in keys], dtype=np.int64)
    parts = []
    for feature in FEATURES:
        if feature_kind == "logits":
            parts.append(np.stack([by_feature[feature][key]["logits"] for key in keys]))
        elif feature_kind == "probs":
            parts.append(np.stack([by_feature[feature][key]["probs"] for key in keys]))
        elif feature_kind == "logits_probs":
            parts.append(np.stack([by_feature[feature][key]["logits"] for key in keys]))
            parts.append(np.stack([by_feature[feature][key]["probs"] for key in keys]))
        else:
            raise ValueError(feature_kind)
    x = np.concatenate(parts, axis=1)
    return keys, y_multi, x


def subsample_train(x: np.ndarray, y: np.ndarray, max_negatives: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    if max_negatives > 0 and len(neg) > max_negatives:
        neg = rng.choice(neg, size=max_negatives, replace=False)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return x[idx], y[idx]


def threshold_rows(y_true: np.ndarray, score: np.ndarray) -> list[dict[str, object]]:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    rows = []
    # thresholds length = len(precision)-1. Include operating points from thresholds only.
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        pred = score >= t
        tp = int(((y_true == 1) & pred).sum())
        fp = int(((y_true == 0) & pred).sum())
        fn = int(((y_true == 1) & ~pred).sum())
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({
            "threshold": float(t),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "predicted_positive": int(pred.sum()),
        })
    rows.sort(key=lambda row: (float(row["f1"]), float(row["recall"])), reverse=True)
    return rows


def session_rows(keys: list[Key], y_true: np.ndarray, score: np.ndarray, threshold: float) -> list[dict[str, object]]:
    rows = []
    pred = score >= threshold
    sessions = sorted({key[2] for key in keys})
    for session in sessions:
        idx = np.asarray([i for i, key in enumerate(keys) if key[2] == session], dtype=np.int64)
        if len(idx) == 0:
            continue
        tp = int(((y_true[idx] == 1) & pred[idx]).sum())
        fp = int(((y_true[idx] == 0) & pred[idx]).sum())
        fn = int(((y_true[idx] == 1) & ~pred[idx]).sum())
        support = int(y_true[idx].sum())
        predicted = int(pred[idx].sum())
        rows.append({
            "session_id": session,
            "n_frames": len(idx),
            "target_support": support,
            "predicted_target": predicted,
            "precision": tp / predicted if predicted else float("nan"),
            "recall": tp / support if support else float("nan"),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_score_target": float(score[idx][y_true[idx] == 1].mean()) if support else float("nan"),
            "mean_score_not_target": float(score[idx][y_true[idx] == 0].mean()) if support < len(idx) else float("nan"),
        })
    return rows


def distribution_rows(y_true: np.ndarray, score: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for label, mask in (("target", y_true == 1), ("not_target", y_true == 0)):
        values = score[mask]
        rows.append({
            "group": label,
            "n": len(values),
            "mean": float(values.mean()) if len(values) else float("nan"),
            "std": float(values.std()) if len(values) else float("nan"),
            "p50": float(np.quantile(values, 0.50)) if len(values) else float("nan"),
            "p90": float(np.quantile(values, 0.90)) if len(values) else float("nan"),
            "p95": float(np.quantile(values, 0.95)) if len(values) else float("nan"),
            "p99": float(np.quantile(values, 0.99)) if len(values) else float("nan"),
        })
    return rows


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


def run_one(args: argparse.Namespace, feature_kind: str, class_weight: str) -> dict[str, object]:
    train_keys, train_multi, train_x = load_split(args.expert_root, "train", feature_kind)
    val_keys, val_multi, val_x = load_split(args.expert_root, "val", feature_kind)
    train_y = (train_multi == args.target_class).astype(np.int64)
    val_y = (val_multi == args.target_class).astype(np.int64)
    fit_x, fit_y = subsample_train(train_x, train_y, args.max_train_negatives, args.seed)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight=class_weight,
            max_iter=2000,
            solver="lbfgs",
            random_state=args.seed,
        ),
    )
    model.fit(fit_x, fit_y)
    val_score = model.predict_proba(val_x)[:, 1]
    ap = average_precision_score(val_y, val_score)
    try:
        roc = roc_auc_score(val_y, val_score)
    except ValueError:
        roc = float("nan")
    thresholds = threshold_rows(val_y, val_score)
    best = thresholds[0] if thresholds else {}
    mode = f"{feature_kind}_{class_weight or 'none'}"
    out = args.output_root / mode
    write_csv(out / "thresholds_by_f1.csv", thresholds[:200])
    if best:
        write_csv(out / "session_metrics_at_best_f1.csv", session_rows(val_keys, val_y, val_score, float(best["threshold"])))
    write_csv(out / "score_distributions.csv", distribution_rows(val_y, val_score))
    return {
        "mode": mode,
        "feature_kind": feature_kind,
        "class_weight": class_weight or "none",
        "train_rows": len(train_y),
        "target_class": args.target_class,
        "train_positive": int(train_y.sum()),
        "fit_rows": len(fit_y),
        "fit_positive": int(fit_y.sum()),
        "val_rows": len(val_y),
        "val_positive": int(val_y.sum()),
        "average_precision": float(ap),
        "roc_auc": float(roc),
        "best_f1": best.get("f1", float("nan")),
        "best_precision": best.get("precision", float("nan")),
        "best_recall": best.get("recall", float("nan")),
        "best_threshold": best.get("threshold", float("nan")),
        "best_predicted_positive": best.get("predicted_positive", 0),
        "best_tp": best.get("tp", 0),
        "best_fp": best.get("fp", 0),
        "best_fn": best.get("fn", 0),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for feature_kind in ("logits", "probs", "logits_probs"):
        for class_weight in (None, "balanced"):
            rows.append(run_one(args, feature_kind, class_weight))
    rows.sort(key=lambda row: float(row["average_precision"]), reverse=True)
    write_csv(args.output_root / "summary.csv", rows)
    (args.output_root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
