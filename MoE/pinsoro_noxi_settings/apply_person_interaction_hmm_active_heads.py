"""Apply HMM/Viterbi smoothing to person-interaction PinSoRo logits."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEADS = ("task", "social")
CLASS_COUNTS = {"task": 4, "social": 5}
MAX_CLASSES = 5
Key = tuple[str, str, str, str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", default="CC")
    parser.add_argument("--train-split", default="train_internal")
    parser.add_argument("--transition-strengths", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    parser.add_argument("--transition-mixes", nargs="+", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--transition-alpha", type=float, default=1.0)
    parser.add_argument("--write-test", action="store_true")
    parser.add_argument(
        "--active-heads",
        nargs="+",
        choices=HEADS,
        default=list(HEADS),
        help="Heads to include when selecting the best HMM setting and writing predictions.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_train_labels(manifests: list[Path], domain: str, model_split: str) -> tuple[list[Key], np.ndarray]:
    """Read unique role-level full-frame train labels from dyadic manifests."""
    role_tensors: dict[tuple[str, str, str, str, Path], tuple[int, int]] = {}
    for manifest in manifests:
        with resolve(manifest).open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["domain"] != domain or row["model_split"] != model_split:
                    continue
                common = (row["domain"], row["source_split"], row["session_id"])
                for role, col in (("purple", "purple_tensor_relative_path"), ("yellow", "yellow_tensor_relative_path")):
                    path = resolve(Path(row[col]))
                    role_tensors[(*common, role, path)] = (0, int(row["session_aligned_len"]))

    keys: list[Key] = []
    labels: list[int] = []
    for (row_domain, source_split, session_id, role, path), (_start, aligned_len) in sorted(role_tensors.items()):
        with np.load(path) as data:
            task_y = np.asarray(data["task_y"], dtype=np.int64)[:aligned_len]
            task_mask = np.asarray(data["task_mask"], dtype=bool)[:aligned_len]
            social_y = np.asarray(data["social_y"], dtype=np.int64)[:aligned_len]
            social_mask = np.asarray(data["social_mask"], dtype=bool)[:aligned_len]
        for head, y, mask in (("task", task_y, task_mask), ("social", social_y, social_mask)):
            for frame_idx in np.flatnonzero(mask):
                label = int(y[frame_idx])
                if 0 <= label < CLASS_COUNTS[head]:
                    keys.append((row_domain, source_split, session_id, role, head, int(frame_idx)))
                    labels.append(label)
    return keys, np.asarray(labels, dtype=np.int64)


def read_scores(path: Path, require_labels: bool) -> tuple[list[Key], np.ndarray, np.ndarray]:
    keys: list[Key] = []
    labels: list[int] = []
    logits: list[np.ndarray] = []
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            head = row["head"]
            n_classes = CLASS_COUNTS[head]
            y_text = row.get("y_true", "")
            if require_labels and y_text == "":
                continue
            key = (
                row["domain"],
                row["source_split"],
                row["session_id"],
                row["role"],
                head,
                int(row["frame_idx"]),
            )
            value = np.full(MAX_CLASSES, -1.0e9, dtype=np.float64)
            value[:n_classes] = [float(row[f"logit_{idx}"]) for idx in range(n_classes)]
            keys.append(key)
            labels.append(int(y_text) if y_text != "" else -1)
            logits.append(value)
    return keys, np.asarray(labels, dtype=np.int64), np.stack(logits)


def sequence_groups(keys: list[Key]) -> dict[tuple[str, str, str], list[int]]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        _domain, _source_split, session_id, role, head, _frame_idx = key
        groups[(session_id, role, head)].append(idx)
    for idxs in groups.values():
        idxs.sort(key=lambda idx: keys[idx][5])
    return groups


def transition_matrices(keys: list[Key], labels: np.ndarray, alpha: float, mix: float) -> dict[str, np.ndarray]:
    groups = sequence_groups(keys)
    matrices: dict[str, np.ndarray] = {}
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


def f1_scores(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[float, float]:
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    supports = confusion.sum(axis=1)
    values = []
    for klass in range(n_classes):
        tp = confusion[klass, klass]
        fp = confusion[:, klass].sum() - tp
        fn = confusion[klass, :].sum() - tp
        denom = 2 * tp + fp + fn
        values.append(float(2 * tp / denom) if denom else 0.0)
    return float(np.mean(values)), float(np.average(values, weights=supports)) if supports.sum() else float("nan")


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


def filter_heads(keys: list[Key], labels: np.ndarray, logits: np.ndarray | None, active_heads: tuple[str, ...]):
    idx = np.asarray([i for i, key in enumerate(keys) if key[4] in active_heads], dtype=np.int64)
    filtered_keys = [keys[int(i)] for i in idx]
    filtered_labels = labels[idx]
    if logits is None:
        return filtered_keys, filtered_labels, None
    return filtered_keys, filtered_labels, logits[idx]


def filter_domain_scores(keys: list[Key], labels: np.ndarray, logits: np.ndarray, domain: str):
    idx = np.asarray([i for i, key in enumerate(keys) if key[0] == domain], dtype=np.int64)
    return [keys[int(i)] for i in idx], labels[idx], logits[idx]


def evaluate(
    keys: list[Key],
    labels: np.ndarray,
    pred: np.ndarray,
    mode: str,
    param: str,
    active_heads: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for head in active_heads:
        n_classes = CLASS_COUNTS[head]
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        n_sequences, n_transitions, true_flips, pred_flips = flip_counts([keys[i] for i in idx], labels[idx], pred[idx])
        macro_f1, weighted_f1 = f1_scores(labels[idx], pred[idx], n_classes)
        rows.append({
            "mode": mode,
            "param": param,
            "head": head,
            "n_frames": int(len(idx)),
            "kappa": kappa(labels[idx], pred[idx], n_classes),
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "accuracy": float((labels[idx] == pred[idx]).mean()) if len(idx) else float("nan"),
            "n_sequences": n_sequences,
            "n_transitions": n_transitions,
            "true_flips": true_flips,
            "pred_flips": pred_flips,
            "true_flip_rate": true_flips / n_transitions if n_transitions else float("nan"),
            "pred_flip_rate": pred_flips / n_transitions if n_transitions else float("nan"),
        })
    kappas = [float(row["kappa"]) for row in rows]
    rows.append({
        "mode": mode,
        "param": param,
        "head": "mean",
        "n_frames": sum(int(row["n_frames"]) for row in rows),
        "kappa": float(np.mean(kappas)),
        "macro_f1": float(np.mean([float(row["macro_f1"]) for row in rows])),
        "weighted_f1": float(np.mean([float(row["weighted_f1"]) for row in rows])),
        "accuracy": float(np.mean([float(row["accuracy"]) for row in rows])),
        "n_sequences": sum(int(row["n_sequences"]) for row in rows),
        "n_transitions": sum(int(row["n_transitions"]) for row in rows),
        "true_flips": sum(int(row["true_flips"]) for row in rows),
        "pred_flips": sum(int(row["pred_flips"]) for row in rows),
        "true_flip_rate": sum(int(row["true_flips"]) for row in rows) / sum(int(row["n_transitions"]) for row in rows),
        "pred_flip_rate": sum(int(row["pred_flips"]) for row in rows) / sum(int(row["n_transitions"]) for row in rows),
    })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path: Path, keys: list[Key], pred: np.ndarray) -> None:
    rows = [
        {
            "domain": key[0],
            "source_split": key[1],
            "session_id": key[2],
            "role": key[3],
            "head": key[4],
            "frame_idx": key[5],
            "y_pred": int(value),
        }
        for key, value in zip(keys, pred)
    ]
    write_csv(path, rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_keys, train_labels = read_train_labels(args.manifest, args.domain, args.train_split)
    val_keys, val_labels, val_logits = read_scores(args.run_dir / "val_prediction_scores.csv.gz", require_labels=True)
    val_keys, val_labels, val_logits = filter_domain_scores(val_keys, val_labels, val_logits, args.domain)
    active_heads = tuple(args.active_heads)
    train_keys, train_labels, _ = filter_heads(train_keys, train_labels, None, active_heads)
    val_keys, val_labels, val_logits = filter_heads(val_keys, val_labels, val_logits, active_heads)
    val_log_probs = log_softmax_by_head(val_keys, val_logits)
    base_pred = val_logits.argmax(axis=1)

    rows = evaluate(val_keys, val_labels, base_pred, "baseline", "none", active_heads)
    for mix in args.transition_mixes:
        matrices = transition_matrices(train_keys, train_labels, args.transition_alpha, mix)
        for strength in args.transition_strengths:
            pred = apply_hmm(val_keys, val_log_probs, matrices, strength)
            rows.extend(
                evaluate(
                    val_keys,
                    val_labels,
                    pred,
                    "hmm_uniform_start",
                    f"mix={mix:g};strength={strength:g};alpha={args.transition_alpha:g}",
                    active_heads,
                )
            )

    write_csv(args.output_dir / "val_hmm_results.csv", rows)
    mean_rows = [row for row in rows if row["head"] == "mean"]
    mean_rows.sort(key=lambda row: float(row["kappa"]), reverse=True)
    best = mean_rows[0]
    (args.output_dir / "best_hmm_setting.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    if args.write_test:
        parts = dict(item.split("=") for item in str(best["param"]).split(";") if "=" in item)
        mix = float(parts["mix"])
        strength = float(parts["strength"])
        matrices = transition_matrices(train_keys, train_labels, args.transition_alpha, mix)
        test_keys, _test_labels, test_logits = read_scores(args.run_dir / "test_prediction_scores.csv.gz", require_labels=False)
        test_keys, _test_labels, test_logits = filter_domain_scores(test_keys, _test_labels, test_logits, args.domain)
        test_keys, _test_labels, test_logits = filter_heads(test_keys, _test_labels, test_logits, active_heads)
        test_pred = apply_hmm(test_keys, log_softmax_by_head(test_keys, test_logits), matrices, strength)
        write_predictions(args.output_dir / "test_predictions_hmm.csv", test_keys, test_pred)

    print(json.dumps({"best": best, "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
