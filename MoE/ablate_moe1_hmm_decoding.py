"""HMM/Viterbi temporal decoding ablations for selected PinSoRo MoE1 logits.

This is a post-processing alternative to the mean/hysteresis smoothers. It keeps
MoE emissions fixed and decodes each session/role/head sequence with a transition
model estimated from train-internal labels.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "MoE" / "experiments"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "moe1_hmm_decoding_ablation"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
CLASS_COUNTS = {"task": 4, "social": 5}
HEADS = ("task", "social")
MAX_CLASSES = max(CLASS_COUNTS.values())
Key = tuple[str, str, str, str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HMM/Viterbi decoding ablations for MoE1.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cc-expert-root", type=Path)
    parser.add_argument("--cr-expert-root", type=Path)
    parser.add_argument("--cc-combiner-root", type=Path)
    parser.add_argument("--cr-combiner-root", type=Path)
    parser.add_argument("--transition-strengths", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--transition-mixes", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--transition-alpha", type=float, default=1.0)
    return parser.parse_args()


def expert_root(domain: str, args: argparse.Namespace | None = None) -> Path:
    if args is not None:
        custom = args.cc_expert_root if domain == "CC" else args.cr_expert_root
        if custom is not None:
            return custom
    return EXPERIMENT_ROOT / f"moe1_{domain.lower()}_metadata_head_experts"


def combiner_root(domain: str, args: argparse.Namespace | None = None) -> Path:
    if args is not None:
        custom = args.cc_combiner_root if domain == "CC" else args.cr_combiner_root
        if custom is not None:
            return custom
    return EXPERIMENT_ROOT / f"moe1_{domain.lower()}_metadata_head_combiners"


def score_path(domain: str, feature: str, split: str, args: argparse.Namespace | None = None) -> Path:
    run = expert_root(domain, args) / f"{domain.lower()}_{feature}_dyadic_tcn_k11_seed13"
    if split == "train":
        return run / "diagnostics" / "train_internal" / "val_prediction_scores.csv.gz"
    if split == "val":
        return run / "val_prediction_scores.csv.gz"
    raise ValueError(split)


def combiner_summary_path(domain: str, args: argparse.Namespace | None = None) -> Path:
    return combiner_root(domain, args) / "two_head" / "summary.json"


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


def read_domain_data(domain: str, split: str, args: argparse.Namespace) -> tuple[list[Key], np.ndarray, dict[str, np.ndarray]]:
    by_feature = {feature: read_scores(score_path(domain, feature, split, args)) for feature in FEATURES}
    keys = sorted(set.intersection(*(set(rows) for rows in by_feature.values())))
    if not keys:
        raise RuntimeError(f"No aligned rows for {domain} {split}")
    labels = np.asarray([by_feature[FEATURES[0]][key]["y_true"] for key in keys], dtype=np.int64)
    logits = {feature: np.stack([by_feature[feature][key]["logits"] for key in keys]) for feature in FEATURES}
    return keys, labels, logits


def read_weights(domain: str, args: argparse.Namespace) -> dict[str, np.ndarray]:
    summary = json.loads(combiner_summary_path(domain, args).read_text(encoding="utf-8"))
    return {head: np.asarray(weights, dtype=np.float64) for head, weights in summary["weights_by_group"].items()}


def combine_two_head(keys: list[Key], logits: dict[str, np.ndarray], weights: dict[str, np.ndarray]) -> np.ndarray:
    combined = np.empty((len(keys), MAX_CLASSES), dtype=np.float64)
    for head in HEADS:
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        if len(idx) == 0:
            continue
        head_weights = weights[head]
        combined[idx] = sum(float(weight) * logits[feature][idx] for weight, feature in zip(head_weights, FEATURES))
    return combined


def log_softmax_by_head(keys: list[Key], logits: np.ndarray) -> np.ndarray:
    out = np.full_like(logits, -1.0e9, dtype=np.float64)
    for head, n_classes in CLASS_COUNTS.items():
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        if len(idx) == 0:
            continue
        head_logits = logits[idx, :n_classes]
        shifted = head_logits - head_logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        out[idx, :n_classes] = log_probs
    return out


def sequence_groups(keys: list[Key]) -> dict[tuple[str, str, str], list[int]]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        _domain, _source_split, session_id, role, head, _frame_idx = key
        groups[(session_id, role, head)].append(idx)
    for group_key in list(groups):
        groups[group_key].sort(key=lambda idx: keys[idx][5])
    return groups


def transition_matrices(keys: list[Key], labels: np.ndarray, alpha: float, mix: float) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    groups = sequence_groups(keys)
    for head, n_classes in CLASS_COUNTS.items():
        counts = np.full((n_classes, n_classes), float(alpha), dtype=np.float64)
        for (_session, _role, group_head), idxs in groups.items():
            if group_head != head or len(idxs) < 2:
                continue
            y = labels[idxs]
            for prev, nxt in zip(y[:-1], y[1:]):
                if 0 <= prev < n_classes and 0 <= nxt < n_classes:
                    counts[prev, nxt] += 1.0
        learned = counts / counts.sum(axis=1, keepdims=True)
        uniform = np.full_like(learned, 1.0 / n_classes)
        mixed = float(mix) * learned + (1.0 - float(mix)) * uniform
        matrices[head] = np.log(np.clip(mixed, 1.0e-12, 1.0))
    return matrices


def viterbi_decode(emissions: np.ndarray, transition_log: np.ndarray, strength: float) -> np.ndarray:
    n_steps, n_classes = emissions.shape
    if n_steps == 0:
        return np.empty(0, dtype=np.int64)
    scores = np.empty((n_steps, n_classes), dtype=np.float64)
    back = np.zeros((n_steps, n_classes), dtype=np.int64)
    # Uniform start prior: no train-label class prior at sequence start.
    scores[0] = emissions[0]
    trans = float(strength) * transition_log
    for t in range(1, n_steps):
        candidate = scores[t - 1][:, None] + trans
        back[t] = np.argmax(candidate, axis=0)
        scores[t] = emissions[t] + candidate[back[t], np.arange(n_classes)]
    pred = np.empty(n_steps, dtype=np.int64)
    pred[-1] = int(np.argmax(scores[-1]))
    for t in range(n_steps - 2, -1, -1):
        pred[t] = back[t + 1, pred[t + 1]]
    return pred


def apply_hmm(keys: list[Key], log_probs: np.ndarray, matrices: dict[str, np.ndarray], strength: float) -> np.ndarray:
    pred = np.full(len(keys), -1, dtype=np.int64)
    for (_session, _role, head), idxs in sequence_groups(keys).items():
        n_classes = CLASS_COUNTS[head]
        idx = np.asarray(idxs, dtype=np.int64)
        pred[idx] = viterbi_decode(log_probs[idx, :n_classes], matrices[head], strength)
    return pred


def kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    n = confusion.sum()
    if n == 0:
        return float("nan")
    accuracy = np.trace(confusion) / n
    expected = confusion.sum(axis=1) @ confusion.sum(axis=0) / (n * n)
    return float((accuracy - expected) / (1.0 - expected)) if expected < 1.0 else float("nan")


def flip_counts(keys: list[Key], labels: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    n_sequences = 0
    n_transitions = 0
    true_flips = 0
    pred_flips = 0
    for idxs in sequence_groups(keys).values():
        if len(idxs) < 2:
            continue
        idx = np.asarray(idxs, dtype=np.int64)
        n_sequences += 1
        n_transitions += len(idx) - 1
        true_flips += int((labels[idx][1:] != labels[idx][:-1]).sum())
        pred_flips += int((pred[idx][1:] != pred[idx][:-1]).sum())
    return n_sequences, n_transitions, true_flips, pred_flips


def evaluate(keys: list[Key], labels: np.ndarray, pred: np.ndarray, domain: str, mode: str, param: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for head, n_classes in CLASS_COUNTS.items():
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        n_sequences, n_transitions, true_flips, pred_flips = flip_counts([keys[i] for i in idx], labels[idx], pred[idx])
        rows.append({
            "domain": domain,
            "mode": mode,
            "param": param,
            "head": head,
            "n_frames": int(len(idx)),
            "accuracy": float((labels[idx] == pred[idx]).mean()) if len(idx) else float("nan"),
            "kappa": kappa(labels[idx], pred[idx], n_classes),
            "n_sequences": n_sequences,
            "n_transitions": n_transitions,
            "true_flips": true_flips,
            "pred_flips": pred_flips,
            "true_flip_rate": true_flips / n_transitions if n_transitions else float("nan"),
            "pred_flip_rate": pred_flips / n_transitions if n_transitions else float("nan"),
            "flip_rate_ratio": pred_flips / true_flips if true_flips else float("inf") if pred_flips else 0.0,
            "excess_flips": pred_flips - true_flips,
        })
    return rows


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, klass: int) -> tuple[float, float, float, int, int, int, int]:
    tp = int(((y_true == klass) & (y_pred == klass)).sum())
    fp = int(((y_true != klass) & (y_pred == klass)).sum())
    fn = int(((y_true == klass) & (y_pred != klass)).sum())
    support = int((y_true == klass).sum())
    predicted = int((y_pred == klass).sum())
    precision = tp / predicted if predicted else float("nan")
    recall = tp / support if support else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall and not math.isnan(precision) and not math.isnan(recall) else float("nan")
    return precision, recall, f1, support, predicted, tp, fp


def class_metric_rows(keys: list[Key], labels: np.ndarray, pred: np.ndarray, domain: str, mode: str, param: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for head, n_classes in CLASS_COUNTS.items():
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        for klass in range(n_classes):
            precision, recall, f1, support, predicted, tp, fp = precision_recall_f1(labels[idx], pred[idx], klass)
            rows.append({
                "domain": domain,
                "mode": mode,
                "param": param,
                "head": head,
                "class": klass,
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": support - tp,
            })
    return rows


def add_mean_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["domain"]), str(row["mode"]), str(row["param"]))].append(row)
    mean_rows: list[dict[str, object]] = []
    for (domain, mode, param), group in sorted(grouped.items()):
        kappas = [float(row["kappa"]) for row in group]
        accuracies = [float(row["accuracy"]) for row in group]
        transitions = sum(int(row["n_transitions"]) for row in group)
        true_flips = sum(int(row["true_flips"]) for row in group)
        pred_flips = sum(int(row["pred_flips"]) for row in group)
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


def run_domain(domain: str, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_keys, train_labels, _train_logits = read_domain_data(domain, "train", args)
    val_keys, val_labels, val_logits_by_feature = read_domain_data(domain, "val", args)
    combined_logits = combine_two_head(val_keys, val_logits_by_feature, read_weights(domain, args))
    log_probs = log_softmax_by_head(val_keys, combined_logits)
    base_pred = combined_logits.argmax(axis=1)

    rows = evaluate(val_keys, val_labels, base_pred, domain, "baseline", "none")
    class_rows = class_metric_rows(val_keys, val_labels, base_pred, domain, "baseline", "none")

    for mix in args.transition_mixes:
        matrices = transition_matrices(train_keys, train_labels, args.transition_alpha, mix)
        for strength in args.transition_strengths:
            if strength == 0.0:
                continue
            pred = apply_hmm(val_keys, log_probs, matrices, strength)
            mode = "hmm_uniform_start"
            param = f"mix={mix:g};strength={strength:g};alpha={args.transition_alpha:g}"
            rows.extend(evaluate(val_keys, val_labels, pred, domain, mode, param))
            class_rows.extend(class_metric_rows(val_keys, val_labels, pred, domain, mode, param))
    return rows, class_rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    all_class_rows: list[dict[str, object]] = []
    for domain in ("CC", "CR"):
        raw_domain_rows, class_rows = run_domain(domain, args)
        domain_rows = add_mean_rows(raw_domain_rows)
        write_csv(args.output_root / f"{domain.lower()}_hmm_results.csv", domain_rows)
        write_csv(args.output_root / f"{domain.lower()}_class_metrics.csv", class_rows)
        rows.extend(domain_rows)
        all_class_rows.extend(class_rows)
    write_csv(args.output_root / "class_metrics.csv", all_class_rows)

    mean_by_setting: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["head"] == "mean":
            mean_by_setting[(str(row["mode"]), str(row["param"]))].append(row)
    combined_rows: list[dict[str, object]] = []
    for (mode, param), group in sorted(mean_by_setting.items()):
        if len(group) != 2:
            continue
        by_domain = {str(row["domain"]): row for row in group}
        combined_rows.append({
            "mode": mode,
            "param": param,
            "cc_mean_kappa": by_domain["CC"]["kappa"],
            "cr_mean_kappa": by_domain["CR"]["kappa"],
            "combined_mean_kappa": float(np.mean([float(by_domain["CC"]["kappa"]), float(by_domain["CR"]["kappa"])])),
            "cc_pred_flips": by_domain["CC"]["pred_flips"],
            "cr_pred_flips": by_domain["CR"]["pred_flips"],
            "total_pred_flips": int(by_domain["CC"]["pred_flips"]) + int(by_domain["CR"]["pred_flips"]),
            "total_true_flips": int(by_domain["CC"]["true_flips"]) + int(by_domain["CR"]["true_flips"]),
        })
    combined_rows.sort(key=lambda row: float(row["combined_mean_kappa"]), reverse=True)
    write_csv(args.output_root / "combined_hmm_results.csv", combined_rows)
    print(json.dumps(combined_rows[:20], indent=2), flush=True)


if __name__ == "__main__":
    main()
