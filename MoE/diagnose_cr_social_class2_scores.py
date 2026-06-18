"""CR social class-2 score and session diagnostics for MoE 1 two_head."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "MoE" / "experiments"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
ROOT = EXPERIMENT_ROOT / "moe1_cr_metadata_head_experts"
COMBINER = EXPERIMENT_ROOT / "moe1_cr_metadata_head_combiners" / "two_head" / "summary.json"
OUT = EXPERIMENT_ROOT / "moe1_cr_social_class2_score_diagnostic"
N_CLASSES = 5
Key = tuple[str, str, str, str, int]


def score_path(feature: str, split: str) -> Path:
    run = ROOT / f"cr_{feature}_dyadic_tcn_k11_seed13"
    if split == "train":
        return run / "diagnostics" / "train_internal" / "val_prediction_scores.csv.gz"
    return run / "val_prediction_scores.csv.gz"


def read_feature(feature: str, split: str):
    rows = {}
    with gzip.open(score_path(feature, split), "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["head"] != "social":
                continue
            key = (row["domain"], row["source_split"], row["session_id"], row["role"], int(row["frame_idx"]))
            rows[key] = {
                "y": int(row["y_true"]),
                "logits": np.asarray([float(row[f"logit_{idx}"]) for idx in range(N_CLASSES)], dtype=np.float64),
                "probs": np.asarray([float(row[f"prob_{idx}"]) for idx in range(N_CLASSES)], dtype=np.float64),
            }
    return rows


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def load_split(split: str):
    by_feature = {feature: read_feature(feature, split) for feature in FEATURES}
    keys = sorted(set.intersection(*(set(rows) for rows in by_feature.values())))
    y = np.asarray([by_feature[FEATURES[0]][key]["y"] for key in keys], dtype=np.int64)
    logits = {feature: np.stack([by_feature[feature][key]["logits"] for key in keys]) for feature in FEATURES}
    weights = np.asarray(json.loads(COMBINER.read_text())["weights_by_group"]["social"], dtype=np.float64)
    combined = sum(float(weight) * logits[feature] for weight, feature in zip(weights, FEATURES))
    probs = softmax(combined)
    pred = combined.argmax(axis=1)
    return keys, y, combined, probs, pred


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: np.ndarray, prefix: str) -> dict[str, object]:
    if len(values) == 0:
        return {f"{prefix}_{name}": float("nan") for name in ("n", "mean", "p10", "p50", "p90", "p99")}
    return {
        f"{prefix}_n": int(len(values)),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_p10": float(np.quantile(values, 0.10)),
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
        f"{prefix}_p99": float(np.quantile(values, 0.99)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dist_rows = []
    for split in ("train", "val"):
        keys, y, logits, probs, pred = load_split(split)
        margin2 = logits[:, 2] - np.max(np.delete(logits, 2, axis=1), axis=1)
        for klass in range(N_CLASSES):
            mask = y == klass
            row = {"split": split, "true_class": klass, "support": int(mask.sum()), "predicted_as_2": int((pred[mask] == 2).sum())}
            row.update(summarize(probs[mask, 2], "prob2"))
            row.update(summarize(margin2[mask], "margin2"))
            dist_rows.append(row)
        if split == "val":
            session_rows = []
            for session in sorted({key[2] for key in keys}):
                idx = np.asarray([i for i, key in enumerate(keys) if key[2] == session], dtype=np.int64)
                yy = y[idx]
                pp = pred[idx]
                cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
                np.add.at(cm, (yy, pp), 1)
                true2 = yy == 2
                pred_counts = Counter(int(v) for v in pp)
                row = {
                    "session_id": session,
                    "n_frames": int(len(idx)),
                    "true_0": int((yy == 0).sum()),
                    "true_1": int((yy == 1).sum()),
                    "true_2": int(true2.sum()),
                    "true_3": int((yy == 3).sum()),
                    "pred_0": pred_counts.get(0, 0),
                    "pred_1": pred_counts.get(1, 0),
                    "pred_2": pred_counts.get(2, 0),
                    "pred_3": pred_counts.get(3, 0),
                    "class2_recall": float((pp[true2] == 2).mean()) if true2.any() else float("nan"),
                    "class2_to_0": int(((yy == 2) & (pp == 0)).sum()),
                    "class2_to_1": int(((yy == 2) & (pp == 1)).sum()),
                    "class2_to_2": int(((yy == 2) & (pp == 2)).sum()),
                    "mean_prob2_true2": float(probs[idx][true2, 2].mean()) if true2.any() else float("nan"),
                    "mean_margin2_true2": float(margin2[idx][true2].mean()) if true2.any() else float("nan"),
                }
                session_rows.append(row)
            write_csv(OUT / "val_session_class2_metrics.csv", session_rows)
    write_csv(OUT / "train_val_class2_score_distributions.csv", dist_rows)
    print(OUT)


if __name__ == "__main__":
    main()
