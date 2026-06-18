"""Context diagnostics for the CR social class-3 collapse.

The goal is descriptive, not another training ablation: compare where class 3
appears, what surrounds it, and whether expert/combiner scores ever rank it
near the top.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "MoE" / "experiments"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
SOCIAL_CLASSES = 5
DEFAULT_EXPERT_ROOT = EXPERIMENT_ROOT / "moe1_cr_metadata_head_experts"
DEFAULT_COMBINER = EXPERIMENT_ROOT / "moe1_cr_metadata_head_combiners" / "two_head" / "summary.json"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "moe1_cr_social_class3_context_diagnostic"
Key = tuple[str, str, str, str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose CR social class-3 context and margins.")
    parser.add_argument("--expert-root", type=Path, default=DEFAULT_EXPERT_ROOT)
    parser.add_argument("--combiner-summary", type=Path, default=DEFAULT_COMBINER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
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
            logits = np.asarray([float(row[f"logit_{idx}"]) for idx in range(SOCIAL_CLASSES)], dtype=np.float64)
            probs = np.asarray([float(row[f"prob_{idx}"]) for idx in range(SOCIAL_CLASSES)], dtype=np.float64)
            rows[key] = {
                "y_true": int(row["y_true"]),
                "expert_pred": int(row["y_pred"]),
                "logits": logits,
                "probs": probs,
            }
    return rows


def load_split(root: Path, split: str) -> tuple[list[Key], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    by_feature = {feature: read_feature_scores(score_path(root, feature, split)) for feature in FEATURES}
    keys = sorted(set.intersection(*(set(rows) for rows in by_feature.values())))
    if not keys:
        raise RuntimeError(f"No aligned {split} rows in {root}")
    labels = np.asarray([by_feature[FEATURES[0]][key]["y_true"] for key in keys], dtype=np.int64)
    logits = {feature: np.stack([by_feature[feature][key]["logits"] for key in keys]) for feature in FEATURES}
    probs = {feature: np.stack([by_feature[feature][key]["probs"] for key in keys]) for feature in FEATURES}
    return keys, labels, logits, probs


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def load_social_weights(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["weights_by_group"]["social"], dtype=np.float64)


def combine_logits(logits: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    return sum(float(weight) * logits[feature] for weight, feature in zip(weights, FEATURES))


def groups_for_sequences(keys: list[Key]) -> dict[tuple[str, str], list[int]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        _domain, _source_split, session_id, role, _head, _frame_idx = key
        groups[(session_id, role)].append(idx)
    return groups


def rank_of_class(values: np.ndarray, class_idx: int) -> np.ndarray:
    order = np.argsort(-values, axis=1)
    return np.argmax(order == class_idx, axis=1) + 1


def true_class_margin(values: np.ndarray, class_idx: int) -> np.ndarray:
    others = np.delete(values, class_idx, axis=1)
    return values[:, class_idx] - others.max(axis=1)


def summarize_values(values: np.ndarray, prefix: str) -> dict[str, object]:
    if len(values) == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
        f"{prefix}_max": float(np.max(values)),
    }


def session_summary_rows(
    split: str,
    keys: list[Key],
    labels: np.ndarray,
    combined_logits: np.ndarray,
    combined_probs: np.ndarray,
    expert_logits: dict[str, np.ndarray],
    expert_probs: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    combined_pred = combined_logits.argmax(axis=1)
    combined_rank3 = rank_of_class(combined_logits, 3)
    combined_margin3 = true_class_margin(combined_logits, 3)
    combined_prob3 = combined_probs[:, 3]
    for (session_id, role), idxs in sorted(groups_for_sequences(keys).items()):
        idx = np.asarray(idxs, dtype=np.int64)
        y = labels[idx]
        pred = combined_pred[idx]
        true3 = y == 3
        not3 = ~true3
        class_counts = Counter(int(v) for v in y)
        pred_counts = Counter(int(v) for v in pred)
        row: dict[str, object] = {
            "split": split,
            "session_id": session_id,
            "role": role,
            "n_frames": int(len(idx)),
            "true_class0": class_counts.get(0, 0),
            "true_class1": class_counts.get(1, 0),
            "true_class2": class_counts.get(2, 0),
            "true_class3": class_counts.get(3, 0),
            "true_class4": class_counts.get(4, 0),
            "pred_class0": pred_counts.get(0, 0),
            "pred_class1": pred_counts.get(1, 0),
            "pred_class2": pred_counts.get(2, 0),
            "pred_class3": pred_counts.get(3, 0),
            "pred_class4": pred_counts.get(4, 0),
            "class3_recall": float((pred[true3] == 3).mean()) if true3.any() else float("nan"),
            "true3_combined_rank3_mean": float(combined_rank3[idx][true3].mean()) if true3.any() else float("nan"),
            "true3_combined_rank3_le2_frac": float((combined_rank3[idx][true3] <= 2).mean()) if true3.any() else float("nan"),
        }
        row.update(summarize_values(combined_prob3[idx][true3], "true3_combined_prob3"))
        row.update(summarize_values(combined_prob3[idx][not3], "not3_combined_prob3"))
        row.update(summarize_values(combined_margin3[idx][true3], "true3_combined_logit_margin3"))
        row.update(summarize_values(combined_margin3[idx][not3], "not3_combined_logit_margin3"))
        for feature in FEATURES:
            feature_rank3 = rank_of_class(expert_logits[feature][idx], 3)
            feature_margin3 = true_class_margin(expert_logits[feature][idx], 3)
            row[f"true3_{feature}_rank3_mean"] = float(feature_rank3[true3].mean()) if true3.any() else float("nan")
            row[f"true3_{feature}_rank3_le2_frac"] = float((feature_rank3[true3] <= 2).mean()) if true3.any() else float("nan")
            row.update(summarize_values(expert_probs[feature][idx, 3][true3], f"true3_{feature}_prob3"))
            row.update(summarize_values(feature_margin3[true3], f"true3_{feature}_logit_margin3"))
        rows.append(row)
    return rows


def run_segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    if len(labels) == 0:
        return []
    segments: list[tuple[int, int, int]] = []
    start = 0
    current = int(labels[0])
    for idx in range(1, len(labels)):
        label = int(labels[idx])
        if label != current:
            segments.append((start, idx, current))
            start = idx
            current = label
    segments.append((start, len(labels), current))
    return segments


def segment_rows(
    split: str,
    keys: list[Key],
    labels: np.ndarray,
    combined_logits: np.ndarray,
    combined_probs: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pred = combined_logits.argmax(axis=1)
    rank3 = rank_of_class(combined_logits, 3)
    margin3 = true_class_margin(combined_logits, 3)
    prob3 = combined_probs[:, 3]
    for (session_id, role), idxs in sorted(groups_for_sequences(keys).items()):
        idx = np.asarray(idxs, dtype=np.int64)
        local_labels = labels[idx]
        frame_idxs = np.asarray([keys[i][5] for i in idx], dtype=np.int64)
        segments = run_segments(local_labels)
        for seg_pos, (start, end, label) in enumerate(segments):
            if label != 3:
                continue
            global_idx = idx[start:end]
            prev_label = int(segments[seg_pos - 1][2]) if seg_pos > 0 else None
            next_label = int(segments[seg_pos + 1][2]) if seg_pos + 1 < len(segments) else None
            seg_pred = pred[global_idx]
            pred_counts = Counter(int(v) for v in seg_pred)
            rows.append({
                "split": split,
                "session_id": session_id,
                "role": role,
                "segment_index": seg_pos,
                "start_frame": int(frame_idxs[start]),
                "end_frame_inclusive": int(frame_idxs[end - 1]),
                "length": int(end - start),
                "prev_label": "" if prev_label is None else prev_label,
                "next_label": "" if next_label is None else next_label,
                "prev_next": f"{prev_label}->{next_label}",
                "pred_class0": pred_counts.get(0, 0),
                "pred_class1": pred_counts.get(1, 0),
                "pred_class2": pred_counts.get(2, 0),
                "pred_class3": pred_counts.get(3, 0),
                "pred_class4": pred_counts.get(4, 0),
                "combined_prob3_mean": float(prob3[global_idx].mean()),
                "combined_prob3_max": float(prob3[global_idx].max()),
                "combined_rank3_mean": float(rank3[global_idx].mean()),
                "combined_rank3_le2_frac": float((rank3[global_idx] <= 2).mean()),
                "combined_logit_margin3_mean": float(margin3[global_idx].mean()),
                "combined_logit_margin3_max": float(margin3[global_idx].max()),
            })
    return rows


def transition_rows(split: str, keys: list[Key], labels: np.ndarray) -> list[dict[str, object]]:
    counts: Counter[tuple[int, int]] = Counter()
    for idxs in groups_for_sequences(keys).values():
        local = labels[np.asarray(idxs, dtype=np.int64)]
        segments = run_segments(local)
        for (_s0, _e0, before), (_s1, _e1, after) in zip(segments, segments[1:]):
            counts[(before, after)] += 1
    return [
        {"split": split, "from_label": before, "to_label": after, "count": count}
        for (before, after), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


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
    weights = load_social_weights(args.combiner_summary)
    all_session_rows: list[dict[str, object]] = []
    all_segment_rows: list[dict[str, object]] = []
    all_transition_rows: list[dict[str, object]] = []
    for split in ("train", "val"):
        keys, labels, logits, probs = load_split(args.expert_root, split)
        combined_logits = combine_logits(logits, weights)
        combined_probs = softmax_np(combined_logits)
        all_session_rows.extend(session_summary_rows(split, keys, labels, combined_logits, combined_probs, logits, probs))
        all_segment_rows.extend(segment_rows(split, keys, labels, combined_logits, combined_probs))
        all_transition_rows.extend(transition_rows(split, keys, labels))
    write_csv(args.output_root / "session_context_summary.csv", all_session_rows)
    write_csv(args.output_root / "class3_segments.csv", all_segment_rows)
    write_csv(args.output_root / "label_run_transitions.csv", all_transition_rows)
    print(f"Wrote {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
