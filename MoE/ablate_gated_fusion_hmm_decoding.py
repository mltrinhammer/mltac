"""HMM/Viterbi decoding for PinSoRo NOXI-settings gated-fusion outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.pinsoro_data import PinSoRoWindowDataset, read_pinsoro_window_manifests  # noqa: E402


EXPERIMENT_ROOT = PROJECT_ROOT / "MoE" / "experiments"
DEFAULT_RUN_ROOT = EXPERIMENT_ROOT / "pinsoro_noxi_settings_gated_fusion"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "pinsoro_noxi_settings_gated_fusion_hmm"
RUN_NAMES = {
    "CC": "pinsoro_cc_audio_text_visual_gated_dyadic_shared_noxi_settings_seed13",
    "CR": "pinsoro_cr_audio_text_visual_gated_dyadic_shared_noxi_settings_seed13",
    "both": "pinsoro_both_audio_text_visual_gated_dyadic_shared_noxi_settings_seed13",
}
CLASS_COUNTS = {"task": 4, "social": 5}
HEADS = ("task", "social")
MAX_CLASSES = max(CLASS_COUNTS.values())
Key = tuple[str, str, str, str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HMM decoding on PinSoRo gated-fusion logits.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--domain-scopes", nargs="+", choices=("CC", "CR", "both"), default=["CC", "CR", "both"])
    parser.add_argument("--cc-run-name", default=RUN_NAMES["CC"])
    parser.add_argument("--cr-run-name", default=RUN_NAMES["CR"])
    parser.add_argument("--both-run-name", default=RUN_NAMES["both"])
    parser.add_argument("--transition-strengths", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    parser.add_argument("--transition-mixes", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--transition-alpha", type=float, default=1.0)
    parser.add_argument("--max-cached-tensors", type=int, default=8)
    return parser.parse_args()


def open_text(path: Path):
    return gzip.open(path, "rt", newline="", encoding="utf-8") if path.suffix == ".gz" else path.open(newline="", encoding="utf-8")


def read_scores(path: Path) -> tuple[list[Key], np.ndarray, np.ndarray]:
    rows: dict[Key, dict[str, object]] = {}
    with open_text(path) as handle:
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
    keys = sorted(rows)
    labels = np.asarray([rows[key]["y_true"] for key in keys], dtype=np.int64)
    logits = np.stack([rows[key]["logits"] for key in keys])
    return keys, labels, logits


def sequence_groups(keys: list[Key]) -> dict[tuple[str, str, str], list[int]]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        _domain, _source_split, session_id, role, head, _frame_idx = key
        groups[(session_id, role, head)].append(idx)
    for idxs in groups.values():
        idxs.sort(key=lambda idx: keys[idx][5])
    return groups


def train_label_sequences(config: dict[str, object], domain_scope: str, max_cached_tensors: int) -> tuple[list[Key], np.ndarray]:
    manifests = [Path(path) for path in config["manifest"]]
    windows = read_pinsoro_window_manifests(manifests, PROJECT_ROOT, str(config["train_split"]))
    if domain_scope != "both":
        windows = [window for window in windows if window.domain == domain_scope]
    dataset = PinSoRoWindowDataset(windows, max_cached_tensors=max_cached_tensors, project_root=PROJECT_ROOT)

    seen: set[tuple[str, str, str]] = set()
    keys: list[Key] = []
    labels: list[int] = []
    for window in windows:
        for role_idx, role in enumerate(window.roles):
            role_key = (window.domain, window.session_id, role)
            if role_key in seen or not window.supervised[role_idx]:
                continue
            seen.add(role_key)
            full = dataset.load_full_role(window, role_idx)
            for head in HEADS:
                y = np.asarray(full[f"{head}_y"])[: window.session_aligned_len]
                mask = np.asarray(full[f"{head}_mask"])[: window.session_aligned_len].astype(bool)
                for frame_idx in np.flatnonzero(mask):
                    keys.append((window.domain, window.source_split, window.session_id, role, head, int(frame_idx)))
                    labels.append(int(y[frame_idx]))
    order = sorted(range(len(keys)), key=lambda idx: keys[idx])
    return [keys[idx] for idx in order], np.asarray([labels[idx] for idx in order], dtype=np.int64)


def transition_matrices(keys: list[Key], labels: np.ndarray, alpha: float, mix: float) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    groups = sequence_groups(keys)
    for head, n_classes in CLASS_COUNTS.items():
        counts = np.full((n_classes, n_classes), float(alpha), dtype=np.float64)
        for (_session, _role, group_head), idxs in groups.items():
            if group_head != head or len(idxs) < 2:
                continue
            y = labels[np.asarray(idxs, dtype=np.int64)]
            for prev, nxt in zip(y[:-1], y[1:]):
                if 0 <= prev < n_classes and 0 <= nxt < n_classes:
                    counts[prev, nxt] += 1.0
        learned = counts / counts.sum(axis=1, keepdims=True)
        uniform = np.full_like(learned, 1.0 / n_classes)
        mixed = float(mix) * learned + (1.0 - float(mix)) * uniform
        matrices[head] = np.log(np.clip(mixed, 1.0e-12, 1.0))
    return matrices


def log_softmax_by_head(keys: list[Key], logits: np.ndarray) -> np.ndarray:
    out = np.full_like(logits, -1.0e9, dtype=np.float64)
    for head, n_classes in CLASS_COUNTS.items():
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        if len(idx) == 0:
            continue
        values = logits[idx, :n_classes]
        shifted = values - values.max(axis=1, keepdims=True)
        out[idx, :n_classes] = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return out


def viterbi_decode(emissions: np.ndarray, transition_log: np.ndarray, strength: float) -> np.ndarray:
    n_steps, n_classes = emissions.shape
    scores = np.empty((n_steps, n_classes), dtype=np.float64)
    back = np.zeros((n_steps, n_classes), dtype=np.int64)
    scores[0] = emissions[0]
    trans = float(strength) * transition_log
    for step in range(1, n_steps):
        candidate = scores[step - 1][:, None] + trans
        back[step] = np.argmax(candidate, axis=0)
        scores[step] = emissions[step] + candidate[back[step], np.arange(n_classes)]
    pred = np.empty(n_steps, dtype=np.int64)
    pred[-1] = int(np.argmax(scores[-1]))
    for step in range(n_steps - 2, -1, -1):
        pred[step] = back[step + 1, pred[step + 1]]
    return pred


def apply_hmm(keys: list[Key], log_probs: np.ndarray, matrices: dict[str, np.ndarray], strength: float) -> np.ndarray:
    pred = np.full(len(keys), -1, dtype=np.int64)
    for (_session, _role, head), idxs in sequence_groups(keys).items():
        idx = np.asarray(idxs, dtype=np.int64)
        n_classes = CLASS_COUNTS[head]
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


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, klass: int) -> tuple[float, float, float, int, int, int]:
    tp = int(((y_true == klass) & (y_pred == klass)).sum())
    fp = int(((y_true != klass) & (y_pred == klass)).sum())
    fn = int(((y_true == klass) & (y_pred != klass)).sum())
    support = int((y_true == klass).sum())
    predicted = int((y_pred == klass).sum())
    precision = tp / predicted if predicted else float("nan")
    recall = tp / support if support else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall and not math.isnan(precision) and not math.isnan(recall) else float("nan")
    return precision, recall, f1, support, predicted, tp


def flip_counts(keys: list[Key], labels: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    n_sequences = n_transitions = true_flips = pred_flips = 0
    for idxs in sequence_groups(keys).values():
        if len(idxs) < 2:
            continue
        idx = np.asarray(idxs, dtype=np.int64)
        n_sequences += 1
        n_transitions += len(idx) - 1
        true_flips += int((labels[idx][1:] != labels[idx][:-1]).sum())
        pred_flips += int((pred[idx][1:] != pred[idx][:-1]).sum())
    return n_sequences, n_transitions, true_flips, pred_flips


def evaluate(keys: list[Key], labels: np.ndarray, pred: np.ndarray, scope: str, mode: str, param: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for head, n_classes in CLASS_COUNTS.items():
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        n_sequences, n_transitions, true_flips, pred_flips = flip_counts([keys[i] for i in idx], labels[idx], pred[idx])
        rows.append({
            "scope": scope,
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
        })
    return rows


def class_rows(keys: list[Key], labels: np.ndarray, pred: np.ndarray, scope: str, mode: str, param: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for head, n_classes in CLASS_COUNTS.items():
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        for klass in range(n_classes):
            precision, recall, f1, support, predicted, tp = precision_recall_f1(labels[idx], pred[idx], klass)
            rows.append({
                "scope": scope,
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
            })
    return rows


def add_mean_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scope"]), str(row["mode"]), str(row["param"]))].append(row)
    out = list(rows)
    for (scope, mode, param), group in sorted(grouped.items()):
        kappas = [float(row["kappa"]) for row in group]
        accuracies = [float(row["accuracy"]) for row in group]
        transitions = sum(int(row["n_transitions"]) for row in group)
        true_flips = sum(int(row["true_flips"]) for row in group)
        pred_flips = sum(int(row["pred_flips"]) for row in group)
        out.append({
            "scope": scope,
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
        })
    return out


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


def run_name_for_scope(scope: str, args: argparse.Namespace) -> str:
    if scope == "CC":
        return str(args.cc_run_name)
    if scope == "CR":
        return str(args.cr_run_name)
    return str(args.both_run_name)


def run_scope(scope: str, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_dir = args.run_root / run_name_for_scope(scope, args)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    train_keys, train_labels = train_label_sequences(config, scope, args.max_cached_tensors)
    val_keys, val_labels, val_logits = read_scores(run_dir / "val_prediction_scores.csv.gz")
    log_probs = log_softmax_by_head(val_keys, val_logits)
    base_pred = val_logits.argmax(axis=1)

    metric_rows = evaluate(val_keys, val_labels, base_pred, scope, "baseline", "none")
    cls_rows = class_rows(val_keys, val_labels, base_pred, scope, "baseline", "none")
    for mix in args.transition_mixes:
        matrices = transition_matrices(train_keys, train_labels, args.transition_alpha, mix)
        for strength in args.transition_strengths:
            pred = apply_hmm(val_keys, log_probs, matrices, strength)
            param = f"mix={mix:g};strength={strength:g};alpha={args.transition_alpha:g}"
            metric_rows.extend(evaluate(val_keys, val_labels, pred, scope, "hmm_uniform_start", param))
            cls_rows.extend(class_rows(val_keys, val_labels, pred, scope, "hmm_uniform_start", param))
    return add_mean_rows(metric_rows), cls_rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    all_class_rows: list[dict[str, object]] = []
    for scope in args.domain_scopes:
        rows, cls_rows = run_scope(scope, args)
        write_csv(args.output_root / f"{scope.lower()}_hmm_results.csv", rows)
        write_csv(args.output_root / f"{scope.lower()}_class_metrics.csv", cls_rows)
        all_rows.extend(rows)
        all_class_rows.extend(cls_rows)
    write_csv(args.output_root / "hmm_results.csv", all_rows)
    write_csv(args.output_root / "class_metrics.csv", all_class_rows)
    best = sorted(
        [row for row in all_rows if row["head"] == "mean"],
        key=lambda row: float(row["kappa"]),
        reverse=True,
    )
    write_csv(args.output_root / "best_mean_results.csv", best)
    print(json.dumps(best[:20], indent=2), flush=True)


if __name__ == "__main__":
    main()
