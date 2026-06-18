"""HMM/Viterbi decoding with optional emission prior correction.

Diagnostic follow-up to ablate_moe1_hmm_decoding.py. It tests whether global
class-balance correction can recover underpredicted classes after temporal HMM
smoothing. Validation priors are optimistic diagnostics, not deployable settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MOE_ROOT = Path(__file__).resolve().parent
if str(MOE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOE_ROOT))

import ablate_moe1_hmm_decoding as hmm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HMM prior-correction diagnostics for MoE1.")
    parser.add_argument("--output-root", type=Path, default=hmm.EXPERIMENT_ROOT / "moe1_hmm_prior_decoding_ablation")
    parser.add_argument("--mix", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=8.0)
    parser.add_argument("--transition-alpha", type=float, default=1.0)
    parser.add_argument("--prior-strengths", nargs="+", type=float, default=[0.25, 0.5, 1.0, 1.5, 2.0])
    return parser.parse_args()


def class_priors(keys: list[hmm.Key], labels: np.ndarray, smoothing: float = 1.0) -> dict[str, np.ndarray]:
    priors = {}
    for head, n_classes in hmm.CLASS_COUNTS.items():
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        counts = np.bincount(labels[idx], minlength=n_classes).astype(np.float64) + smoothing
        priors[head] = counts / counts.sum()
    return priors


def prior_bias_for_keys(
    keys: list[hmm.Key],
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    strength: float,
) -> np.ndarray:
    bias = np.zeros((len(keys), hmm.MAX_CLASSES), dtype=np.float64)
    for head, n_classes in hmm.CLASS_COUNTS.items():
        idx = np.asarray([i for i, key in enumerate(keys) if key[4] == head], dtype=np.int64)
        if len(idx) == 0:
            continue
        correction = np.log(np.clip(target[head], 1e-12, 1.0)) - np.log(np.clip(source[head], 1e-12, 1.0))
        bias[idx, :n_classes] = float(strength) * correction
    return bias


def target_priors(kind: str, train_priors: dict[str, np.ndarray], val_priors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    targets = {}
    for head, n_classes in hmm.CLASS_COUNTS.items():
        if kind == "uniform":
            targets[head] = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
        elif kind == "val_oracle":
            targets[head] = val_priors[head]
        elif kind == "sqrt_val_oracle":
            p = np.sqrt(val_priors[head])
            targets[head] = p / p.sum()
        elif kind == "train":
            targets[head] = train_priors[head]
        else:
            raise ValueError(kind)
    return targets


def run_domain(domain: str, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_keys, train_labels, _train_logits = hmm.read_domain_data(domain, "train", args)
    val_keys, val_labels, val_logits_by_feature = hmm.read_domain_data(domain, "val", args)
    combined_logits = hmm.combine_two_head(val_keys, val_logits_by_feature, hmm.read_weights(domain, args))
    log_probs = hmm.log_softmax_by_head(val_keys, combined_logits)
    base_pred = combined_logits.argmax(axis=1)
    matrices = hmm.transition_matrices(train_keys, train_labels, args.transition_alpha, args.mix)
    train_priors = class_priors(train_keys, train_labels)
    val_priors = class_priors(val_keys, val_labels)

    rows = hmm.evaluate(val_keys, val_labels, base_pred, domain, "baseline", "none")
    class_rows = hmm.class_metric_rows(val_keys, val_labels, base_pred, domain, "baseline", "none")
    hmm_pred = hmm.apply_hmm(val_keys, log_probs, matrices, args.strength)
    rows.extend(hmm.evaluate(val_keys, val_labels, hmm_pred, domain, "hmm_no_prior", f"mix={args.mix:g};strength={args.strength:g}"))
    class_rows.extend(hmm.class_metric_rows(val_keys, val_labels, hmm_pred, domain, "hmm_no_prior", f"mix={args.mix:g};strength={args.strength:g}"))

    for kind in ("uniform", "sqrt_val_oracle", "val_oracle", "train"):
        target = target_priors(kind, train_priors, val_priors)
        for prior_strength in args.prior_strengths:
            adjusted = log_probs + prior_bias_for_keys(val_keys, train_priors, target, prior_strength)
            pred = hmm.apply_hmm(val_keys, adjusted, matrices, args.strength)
            mode = f"hmm_prior_{kind}"
            param = f"mix={args.mix:g};strength={args.strength:g};prior={prior_strength:g}"
            rows.extend(hmm.evaluate(val_keys, val_labels, pred, domain, mode, param))
            class_rows.extend(hmm.class_metric_rows(val_keys, val_labels, pred, domain, mode, param))
    return rows, class_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
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
    # Make this object compatible with imported root-aware functions.
    args.cc_expert_root = None
    args.cr_expert_root = None
    args.cc_combiner_root = None
    args.cr_combiner_root = None
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    all_class_rows = []
    for domain in ("CC", "CR"):
        raw_rows, class_rows = run_domain(domain, args)
        domain_rows = hmm.add_mean_rows(raw_rows)
        write_csv(args.output_root / f"{domain.lower()}_prior_hmm_results.csv", domain_rows)
        write_csv(args.output_root / f"{domain.lower()}_class_metrics.csv", class_rows)
        rows.extend(domain_rows)
        all_class_rows.extend(class_rows)
    write_csv(args.output_root / "class_metrics.csv", all_class_rows)

    grouped = defaultdict(list)
    for row in rows:
        if row["head"] == "mean":
            grouped[(str(row["mode"]), str(row["param"]))].append(row)
    combined = []
    for (mode, param), group in sorted(grouped.items()):
        if len(group) != 2:
            continue
        by_domain = {str(row["domain"]): row for row in group}
        combined.append({
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
    combined.sort(key=lambda row: float(row["combined_mean_kappa"]), reverse=True)
    write_csv(args.output_root / "combined_prior_hmm_results.csv", combined)
    print(json.dumps(combined[:20], indent=2), flush=True)


if __name__ == "__main__":
    main()
