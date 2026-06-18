"""Export PinSoRo soft-confidence MoE1 two-head predictions with HMM decoding."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acm_pipeline.pinsoro_train_utils import CLASS_LABELS, CLASS_COUNTS, HEAD_OUTPUT_NAMES


EXPERIMENT_ROOT = PROJECT_ROOT / "MoE" / "experiments"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
HEADS = ("task", "social")
MAX_CLASSES = max(CLASS_COUNTS.values())
Key = tuple[str, str, str, str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expert-root",
        type=Path,
        default=EXPERIMENT_ROOT / "moe1_soft_confidence_metadata_head_experts",
    )
    parser.add_argument(
        "--combiner-summary",
        type=Path,
        default=EXPERIMENT_ROOT
        / "moe1_soft_confidence_metadata_head_combiners"
        / "two_head"
        / "summary.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mix", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def run_dir(root: Path, domain: str, feature: str) -> Path:
    return root / f"{domain.lower()}_{feature}_dyadic_tcn_k11_seed13"


def score_path(root: Path, domain: str, feature: str, split: str) -> Path:
    base = run_dir(root, domain, feature)
    if split == "train":
        return base / "diagnostics" / "train_internal" / "val_prediction_scores.csv.gz"
    if split == "test":
        return base / "test_prediction_scores.csv.gz"
    raise ValueError(split)


def read_scores(path: Path) -> dict[Key, dict[str, object]]:
    rows: dict[Key, dict[str, object]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            head = row["head"]
            n_classes = CLASS_COUNTS[head]
            logits = np.full(MAX_CLASSES, -1.0e9, dtype=np.float64)
            logits[:n_classes] = [float(row[f"logit_{idx}"]) for idx in range(n_classes)]
            key = (
                row["domain"],
                row["source_split"],
                row["session_id"],
                row["role"],
                head,
                int(row["frame_idx"]),
            )
            y_true = row.get("y_true", "")
            rows[key] = {
                "y_true": None if y_true == "" else int(y_true),
                "logits": logits,
            }
    return rows


def read_domain(root: Path, domain: str, split: str) -> tuple[list[Key], np.ndarray | None, dict[str, np.ndarray]]:
    by_feature = {feature: read_scores(score_path(root, domain, feature, split)) for feature in FEATURES}
    keys = sorted(set.intersection(*(set(rows) for rows in by_feature.values())))
    if not keys:
        raise RuntimeError(f"No aligned score rows for {domain} {split}.")
    labels = [by_feature[FEATURES[0]][key]["y_true"] for key in keys]
    y = None if any(value is None for value in labels) else np.asarray(labels, dtype=np.int64)
    logits = {
        feature: np.stack([by_feature[feature][key]["logits"] for key in keys])
        for feature in FEATURES
    }
    return keys, y, logits


def read_weights(path: Path) -> dict[str, np.ndarray]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    return {
        head: np.asarray(values, dtype=np.float64)
        for head, values in summary["weights_by_group"].items()
    }


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
        out[idx, :n_classes] = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return out


def sequence_groups(keys: list[Key]) -> dict[tuple[str, str, str], list[int]]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        _domain, _source_split, session_id, role, head, _frame_idx = key
        groups[(session_id, role, head)].append(idx)
    for idxs in groups.values():
        idxs.sort(key=lambda idx: keys[idx][5])
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
        matrices[head] = np.log(np.clip(float(mix) * learned + (1.0 - float(mix)) * uniform, 1.0e-12, 1.0))
    return matrices


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
        pred[idx] = viterbi_decode(log_probs[idx, : CLASS_COUNTS[head]], matrices[head], strength)
    return pred


def write_submission(output_dir: Path, keys: list[Key], pred: np.ndarray) -> int:
    grouped: dict[tuple[str, str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for key, value in zip(keys, pred):
        domain, _source_split, session_id, role, head, frame_idx = key
        if domain == "CR" and role == "yellow":
            continue
        grouped[(domain, session_id, role, head)].append((frame_idx, int(value)))

    written = 0
    for (domain, session_id, role, head), values in sorted(grouped.items()):
        values.sort()
        expected = list(range(values[0][0], values[0][0] + len(values)))
        actual = [frame_idx for frame_idx, _value in values]
        if actual != expected or expected[0] != 0:
            raise RuntimeError(f"Non-contiguous frames for {domain}/{session_id}/{role}/{head}.")
        session_dir = output_dir / f"pinsoro-{domain.lower()}" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / f"{role}.{HEAD_OUTPUT_NAMES[head]}.prediction.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            for _frame_idx, value in values:
                handle.write(f"{CLASS_LABELS[head][value]}\n")
        written += 1
    (output_dir / ".complete").write_text(f"files={written}\n", encoding="utf-8")
    return written


def main() -> None:
    args = parse_args()
    weights = read_weights(args.combiner_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    manifest_rows = []
    for domain in ("CC", "CR"):
        train_keys, train_labels, _train_logits = read_domain(args.expert_root, domain, "train")
        if train_labels is None:
            raise RuntimeError(f"Training labels missing for {domain}.")
        test_keys, _test_labels, test_logits = read_domain(args.expert_root, domain, "test")
        matrices = transition_matrices(train_keys, train_labels, args.alpha, args.mix)
        combined = combine_two_head(test_keys, test_logits, weights)
        pred = apply_hmm(test_keys, log_softmax_by_head(test_keys, combined), matrices, args.strength)
        written = write_submission(args.output_dir, test_keys, pred)
        total += written
        manifest_rows.append({"domain": domain, "rows": len(test_keys), "files": written})
    (args.output_dir / "export_manifest.json").write_text(
        json.dumps(
            {
                "expert_root": str(args.expert_root),
                "combiner_summary": str(args.combiner_summary),
                "mix": args.mix,
                "strength": args.strength,
                "alpha": args.alpha,
                "domains": manifest_rows,
                "files": total,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {total} PinSoRo files to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
