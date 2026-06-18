"""Leave-one-class3-session-out diagnostic for CR social class 3.

This is a lightweight session-transfer test on frozen expert outputs. It is not
a replacement for deep retraining; it asks whether class 3 is represented
consistently enough that a simple detector trained on other class-3 sessions can
recover it in the held-out session.
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
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "moe1_cr_social_class3_loso_diagnostic"
Key = tuple[str, str, str, str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOSO CR social class-3 diagnostic from frozen expert scores.")
    parser.add_argument("--expert-root", type=Path, default=DEFAULT_EXPERT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-kind", choices=("logits", "probs", "logits_probs"), default="logits")
    parser.add_argument("--max-train-negatives", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


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
            rows[key] = {
                "y_true": int(row["y_true"]),
                "logits": np.asarray([float(row[f"logit_{idx}"]) for idx in range(SOCIAL_CLASSES)], dtype=np.float64),
                "probs": np.asarray([float(row[f"prob_{idx}"]) for idx in range(SOCIAL_CLASSES)], dtype=np.float64),
            }
    return rows


def load_split(root: Path, split: str, feature_kind: str) -> tuple[list[Key], np.ndarray, np.ndarray]:
    by_feature = {feature: read_feature_scores(score_path(root, feature, split)) for feature in FEATURES}
    keys = sorted(set.intersection(*(set(rows) for rows in by_feature.values())))
    if not keys:
        raise RuntimeError(f"No aligned rows for {split}")
    y = np.asarray([by_feature[FEATURES[0]][key]["y_true"] for key in keys], dtype=np.int64)
    parts = []
    for feature in FEATURES:
        if feature_kind == "logits":
            parts.append(np.stack([by_feature[feature][key]["logits"] for key in keys]))
        elif feature_kind == "probs":
            parts.append(np.stack([by_feature[feature][key]["probs"] for key in keys]))
        else:
            parts.append(np.stack([by_feature[feature][key]["logits"] for key in keys]))
            parts.append(np.stack([by_feature[feature][key]["probs"] for key in keys]))
    return keys, y, np.concatenate(parts, axis=1)


def load_all(root: Path, feature_kind: str) -> tuple[list[Key], np.ndarray, np.ndarray]:
    all_keys: list[Key] = []
    all_y: list[np.ndarray] = []
    all_x: list[np.ndarray] = []
    for split in ("train", "val"):
        keys, y, x = load_split(root, split, feature_kind)
        all_keys.extend(keys)
        all_y.append(y)
        all_x.append(x)
    return all_keys, np.concatenate(all_y), np.concatenate(all_x, axis=0)


def threshold_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, object]:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    best: dict[str, object] = {
        "best_threshold": float("nan"),
        "best_precision": float("nan"),
        "best_recall": float("nan"),
        "best_f1": float("nan"),
        "best_tp": 0,
        "best_fp": 0,
        "best_fn": 0,
        "best_predicted_positive": 0,
    }
    best_f1 = -1.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best_f1:
            pred = score >= t
            best_f1 = float(f1)
            best = {
                "best_threshold": float(t),
                "best_precision": float(p),
                "best_recall": float(r),
                "best_f1": float(f1),
                "best_tp": int(((y_true == 1) & pred).sum()),
                "best_fp": int(((y_true == 0) & pred).sum()),
                "best_fn": int(((y_true == 1) & ~pred).sum()),
                "best_predicted_positive": int(pred.sum()),
            }
    return best


def sample_fit_rows(
    keys: list[Key],
    y: np.ndarray,
    train_mask: np.ndarray,
    max_negatives: int,
    seed: int,
) -> np.ndarray:
    pos = np.flatnonzero(train_mask & (y == 1))
    neg = np.flatnonzero(train_mask & (y == 0))
    rng = np.random.default_rng(seed)
    if max_negatives > 0 and len(neg) > max_negatives:
        neg = rng.choice(neg, size=max_negatives, replace=False)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return idx


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


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    keys, y_multi, x = load_all(args.expert_root, args.feature_kind)
    y = (y_multi == 3).astype(np.int64)
    sessions = np.asarray([key[2] for key in keys])
    splits = np.asarray(["train" if key[1].startswith("train") else "val" for key in keys])
    class3_sessions = sorted(session for session in set(sessions[y == 1]) if int(y[sessions == session].sum()) > 0)
    rows: list[dict[str, object]] = []
    for heldout in class3_sessions:
        test_mask = sessions == heldout
        train_mask = ~test_mask
        fit_idx = sample_fit_rows(keys, y, train_mask, args.max_train_negatives, args.seed)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=args.seed),
        )
        model.fit(x[fit_idx], y[fit_idx])
        score = model.predict_proba(x[test_mask])[:, 1]
        test_y = y[test_mask]
        ap = average_precision_score(test_y, score)
        try:
            roc = roc_auc_score(test_y, score)
        except ValueError:
            roc = float("nan")
        metrics = threshold_metrics(test_y, score)
        rows.append({
            "heldout_session": heldout,
            "heldout_split": sorted(set(splits[test_mask]))[0],
            "heldout_rows": int(test_mask.sum()),
            "heldout_positive": int(test_y.sum()),
            "train_rows_available": int(train_mask.sum()),
            "train_positive_available": int(y[train_mask].sum()),
            "fit_rows": int(len(fit_idx)),
            "fit_positive": int(y[fit_idx].sum()),
            "average_precision": float(ap),
            "roc_auc": float(roc),
            "positive_score_mean": float(score[test_y == 1].mean()) if test_y.sum() else float("nan"),
            "negative_score_mean": float(score[test_y == 0].mean()) if test_y.sum() < len(test_y) else float("nan"),
            **metrics,
        })
    rows.sort(key=lambda row: (str(row["heldout_split"]), str(row["heldout_session"])))
    write_csv(args.output_root / "loso_summary.csv", rows)
    (args.output_root / "loso_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
