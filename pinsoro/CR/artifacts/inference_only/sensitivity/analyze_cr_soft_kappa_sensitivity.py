#!/usr/bin/env python3
"""CC-style soft-kappa HMM sensitivity for PinSoRo CR validation targets."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path("/work/ACM/ACM-clean")
HMM_DIR = PROJECT / "MoE/pinsoro_noxi_settings"
if str(HMM_DIR) not in sys.path:
    sys.path.insert(0, str(HMM_DIR))

from apply_person_interaction_hmm_active_heads import (  # noqa: E402
    apply_hmm,
    filter_domain_scores,
    filter_heads,
    filter_roles,
    log_softmax_by_head,
    read_scores,
    read_train_labels,
    transition_matrices,
)


SOFT_LABEL_ROOT = Path(
    "/work/ACM/mltac-main/ACM/MoE/moe_data_soft_labels/processed/domain_norm/audio_w2vbert2/val-cr"
)
TRAIN_MANIFESTS = [
    Path("/work/ACM/mltac-main/ACM/MoE/moe_data/outputs/windows_w2400_s1200/audio_w2vbert2_w2400_s1200_dyadic.csv"),
    Path("/work/ACM/mltac-main/ACM/MoE/moe_data/outputs/windows_w2400_s1200/text_xlm_roberta_w2400_s1200_dyadic.csv"),
    Path("/work/ACM/mltac-main/ACM/MoE/moe_data/outputs/windows_w2400_s1200/visual_videomae_w2400_s1200_dyadic.csv"),
]
OUT_DIR = Path("/work/ACM/submission_tracking/cr_soft_kappa_sensitivity")

RUNS = {
    "cr_task": {
        "head": "task",
        "n_classes": 4,
        "score_file": Path(
            "/work/ACM/ACM-clean/MoE/experiments/pinsoro_head_specialists_temporal_delta010_metadata/"
            "pinsoro_cr_audio_text_visual_concat_shared_encoder_linear_none_task_age_gender_role_metadata_head_delta010_seed13/"
            "val_prediction_scores_all_roles.csv.gz"
        ),
    },
    "cr_social": {
        "head": "social",
        "n_classes": 5,
        "score_file": Path(
            "/work/ACM/ACM-clean/MoE/experiments/pinsoro_cr_social_clean_arch_delta010_metadata/"
            "pinsoro_cr_social_dyadic_shared_tcn_pre_encoder_delta010_metadata_seed13/"
            "val_prediction_scores_all_roles.csv.gz"
        ),
    },
}


def load_soft_targets(head: str) -> dict[tuple[str, str, int], tuple[np.ndarray, float]]:
    targets: dict[tuple[str, str, int], tuple[np.ndarray, float]] = {}
    for path in sorted(SOFT_LABEL_ROOT.glob("*/*.npz")):
        session_id = path.parent.name
        role = path.name.split(".", 1)[0]
        with np.load(path) as data:
            soft = np.asarray(data[f"{head}_soft_y"], dtype=np.float64)
            mask = np.asarray(data[f"{head}_soft_mask"], dtype=bool)
            weight = np.asarray(data[f"{head}_weight"], dtype=np.float64)
        for frame_idx in np.flatnonzero(mask):
            targets[(session_id, role, int(frame_idx))] = (soft[frame_idx], float(weight[frame_idx]))
    return targets


def soft_kappa(targets: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    if len(pred) == 0:
        return float("nan")
    if weights is None:
        weights = np.ones(len(pred), dtype=np.float64)
    weighted_targets = targets * weights[:, None]
    n_classes = targets.shape[1]
    confusion = np.zeros((n_classes, n_classes), dtype=np.float64)
    for cls in range(n_classes):
        if np.any(pred == cls):
            confusion[:, cls] = weighted_targets[pred == cls].sum(axis=0)
    total = confusion.sum()
    if total <= 0:
        return float("nan")
    observed = np.trace(confusion) / total
    true_marginal = confusion.sum(axis=1)
    pred_marginal = confusion.sum(axis=0)
    expected = float(np.dot(true_marginal, pred_marginal) / (total * total))
    if np.isclose(1.0 - expected, 0.0):
        return 0.0
    return float((observed - expected) / (1.0 - expected))


def subset_targets(keys, labels: np.ndarray, soft_targets, n_classes: int, subset: str):
    rows = []
    target_rows = []
    weights = []
    source_counts = {"canonical": 0, "numbered_blank": 0, "missing": 0}
    for idx, key in enumerate(keys):
        _domain, _source_split, session_id, role, _head, frame_idx = key
        if labels[idx] >= 0:
            if subset == "numbered_blank":
                continue
            dist = np.zeros(n_classes, dtype=np.float64)
            dist[int(labels[idx])] = 1.0
            rows.append(idx)
            target_rows.append(dist)
            weights.append(1.0)
            source_counts["canonical"] += 1
            continue

        target = soft_targets.get((session_id, role, int(frame_idx)))
        if target is None:
            source_counts["missing"] += 1
            continue
        if subset == "canonical":
            continue
        dist, confidence = target
        rows.append(idx)
        target_rows.append(dist[:n_classes])
        weights.append(confidence)
        source_counts["numbered_blank"] += 1

    if not rows:
        return np.asarray([], dtype=np.int64), np.empty((0, n_classes)), np.asarray([]), source_counts
    return (
        np.asarray(rows, dtype=np.int64),
        np.vstack(target_rows).astype(np.float64),
        np.asarray(weights, dtype=np.float64),
        source_counts,
    )


def flip_rate(keys, pred: np.ndarray, idx: np.ndarray) -> float:
    groups: dict[tuple[str, str, str], list[int]] = {}
    for pos in idx:
        key = keys[int(pos)]
        groups.setdefault((key[2], key[3], key[4]), []).append(int(pos))
    flips = transitions = 0
    for positions in groups.values():
        positions.sort(key=lambda i: keys[i][5])
        if len(positions) < 2:
            continue
        values = pred[np.asarray(positions, dtype=np.int64)]
        flips += int((values[1:] != values[:-1]).sum())
        transitions += len(values) - 1
    return float(flips / transitions) if transitions else float("nan")


def evaluate_subset(name: str, subset: str, keys, labels, pred, soft_targets, n_classes: int, mode: str, param: str):
    idx, targets, weights, counts = subset_targets(keys, labels, soft_targets, n_classes, subset)
    return {
        "run": name,
        "target_subset": subset,
        "mode": mode,
        "param": param,
        "frames": int(len(idx)),
        "canonical_frames": int(counts["canonical"]),
        "numbered_blank_frames": int(counts["numbered_blank"]),
        "soft_kappa": soft_kappa(targets, pred[idx]),
        "confidence_weighted_soft_kappa": soft_kappa(targets, pred[idx], weights),
        "pred_flip_rate": flip_rate(keys, pred, idx),
    }


def evaluate_run(name: str, spec: dict[str, object]) -> list[dict[str, object]]:
    head = str(spec["head"])
    n_classes = int(spec["n_classes"])
    soft_targets = load_soft_targets(head)

    train_keys, train_labels = read_train_labels(TRAIN_MANIFESTS, "CR", "train_internal")
    train_keys, train_labels, _ = filter_heads(train_keys, train_labels, None, (head,))
    train_keys, train_labels, _ = filter_roles(train_keys, train_labels, None, ("purple",))

    keys, labels, logits = read_scores(Path(spec["score_file"]), require_labels=False)
    keys, labels, logits = filter_domain_scores(keys, labels, logits, "CR")
    keys, labels, logits = filter_heads(keys, labels, logits, (head,))
    keys, labels, logits = filter_roles(keys, labels, logits, ("purple",))
    log_probs = log_softmax_by_head(keys, logits)

    rows: list[dict[str, object]] = []
    raw_pred = logits[:, :n_classes].argmax(axis=1)
    for subset in ("canonical", "numbered_blank", "augmented"):
        rows.append(evaluate_subset(name, subset, keys, labels, raw_pred, soft_targets, n_classes, "raw", "none"))

    for mix in (0.5, 0.75, 1.0):
        matrices = transition_matrices(train_keys, train_labels, alpha=1.0, mix=mix)
        for strength in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0):
            pred = apply_hmm(keys, log_probs, matrices, strength)
            param = f"mix={mix:g};strength={strength:g};alpha=1"
            for subset in ("canonical", "numbered_blank", "augmented"):
                rows.append(evaluate_subset(name, subset, keys, labels, pred, soft_targets, n_classes, "hmm", param))
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    selected = []
    coverage = []
    for name, spec in RUNS.items():
        rows = evaluate_run(name, spec)
        all_rows.extend(rows)
        for subset in ("canonical", "numbered_blank", "augmented"):
            subset_rows = [row for row in rows if row["target_subset"] == subset and row["mode"] == "hmm"]
            best = max(subset_rows, key=lambda row: float(row["confidence_weighted_soft_kappa"]))
            selected.append(best)
        augmented = next(row for row in rows if row["target_subset"] == "augmented" and row["mode"] == "raw")
        canonical = next(row for row in rows if row["target_subset"] == "canonical" and row["mode"] == "raw")
        total = 132558
        coverage.append(
            {
                "run": name,
                "total_frames": total,
                "canonical_valid": canonical["frames"],
                "added_numbered_annotation_frames": augmented["numbered_blank_frames"],
                "missing_after_numbered_annotations": total - augmented["frames"],
                "before_labeled": canonical["frames"],
                "before_missing": total - canonical["frames"],
                "after_labeled": augmented["frames"],
                "after_missing": total - augmented["frames"],
                "added_annotation_share": augmented["numbered_blank_frames"] / total,
            }
        )

    for path, rows in (
        (OUT_DIR / "sensitivity_metrics.csv", all_rows),
        (OUT_DIR / "selected_by_confidence_weighted_soft_kappa.csv", selected),
        (OUT_DIR / "coverage.csv", coverage),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps({"selected": selected, "coverage": coverage}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"selected": selected, "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
