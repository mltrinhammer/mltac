"""Annotator-aware validation metrics for PinSoRo MoE predictions.

Adds secondary held-out metrics that use numbered validation annotations where
canonical validation labels are blank:
- soft NLL against annotator vote distributions
- expected kappa from a soft confusion matrix
- any-annotator match accuracy
"""

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
MOE_ROOT = PROJECT_ROOT / "MoE"
if str(MOE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOE_ROOT))

from ablate_moe1_hmm_decoding import (  # noqa: E402
    CLASS_COUNTS,
    HEADS,
    apply_hmm,
    combine_two_head,
    log_softmax_by_head,
    FEATURES,
    MAX_CLASSES,
    read_domain_data,
    read_weights,
    transition_matrices,
)
from src.acm_pipeline.pinsoro import LABEL_MAPS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate annotator-aware validation metrics.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, default=MOE_ROOT / "moe_data_soft_labels" / "outputs" / "raw_manifest.csv")
    parser.add_argument("--cache-root", type=Path, default=MOE_ROOT / "moe_data_soft_labels" / "cache" / "pinsoro")
    parser.add_argument("--cc-expert-root", type=Path, required=True)
    parser.add_argument("--cr-expert-root", type=Path, required=True)
    parser.add_argument("--cc-combiner-root", type=Path, required=True)
    parser.add_argument("--cr-combiner-root", type=Path, required=True)
    parser.add_argument("--transition-strengths", nargs="+", type=float, default=[4.0, 8.0, 12.0])
    parser.add_argument("--transition-mixes", nargs="+", type=float, default=[0.75, 1.0])
    parser.add_argument("--transition-alpha", type=float, default=1.0)
    parser.add_argument(
        "--val-score-subdir",
        default="diagnostics/val_full",
        help="Run-dir relative directory containing full-frame val_prediction_scores.csv.gz.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)




def expert_root(domain: str, args: argparse.Namespace) -> Path:
    return args.cc_expert_root if domain == "CC" else args.cr_expert_root


def full_val_score_path(domain: str, feature: str, args: argparse.Namespace) -> Path:
    return (
        expert_root(domain, args)
        / f"{domain.lower()}_{feature}_dyadic_tcn_k11_seed13"
        / args.val_score_subdir
        / "val_prediction_scores.csv.gz"
    )


def read_full_scores(path: Path) -> dict[tuple[str, str, str, str, str, int], dict[str, object]]:
    rows: dict[tuple[str, str, str, str, str, int], dict[str, object]] = {}
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
            for idx in range(n_classes):
                logits[idx] = float(row[f"logit_{idx}"])
            rows[key] = {"logits": logits}
    return rows


def read_full_val_domain_data(domain: str, args: argparse.Namespace):
    by_feature = {feature: read_full_scores(full_val_score_path(domain, feature, args)) for feature in FEATURES}
    keys = sorted(set.intersection(*(set(rows) for rows in by_feature.values())))
    if not keys:
        raise RuntimeError(f"No aligned full validation score rows for {domain}.")
    logits = {
        feature: np.stack([by_feature[feature][key]["logits"] for key in keys])
        for feature in FEATURES
    }
    return keys, logits


def raw_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows[(row["source_split"], row["session_id"], row["role"])] = row
    return rows


def read_label_lines(path: Path, head: str) -> list[int]:
    label_map = LABEL_MAPS[f"{head}_engagement"]
    values = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            values.append(label_map.get(line.strip().lower(), -1))
    return values


def numbered_paths(canonical_path: Path) -> list[Path]:
    pattern = canonical_path.name.replace(".annotation.csv", ".*.annotation.csv")
    paths = []
    for path in sorted(canonical_path.parent.glob(pattern)):
        suffix = path.name.removesuffix(".annotation.csv").rsplit(".", maxsplit=1)[-1]
        if suffix.isdigit():
            paths.append(path)
    return paths


def build_targets(
    rows_by_role: dict[tuple[str, str, str], dict[str, str]], cache_root: Path
) -> dict[tuple[str, str, str, str, int], dict[str, object]]:
    targets: dict[tuple[str, str, str, str, int], dict[str, object]] = {}
    for (source_split, session_id, role), row in rows_by_role.items():
        if not row.get("model_split", "").startswith("val") and row.get("model_split") != "val_internal":
            continue
        for head in HEADS:
            rel = row.get(f"{head}_target_relative_path", "")
            if not rel:
                continue
            canonical_path = cache_root / rel
            if not canonical_path.is_file():
                continue
            canonical = read_label_lines(canonical_path, head)
            annotators = [read_label_lines(path, head) for path in numbered_paths(canonical_path)]
            n_classes = CLASS_COUNTS[head]
            for idx, label in enumerate(canonical):
                dist = np.zeros(n_classes, dtype=np.float64)
                source = "canonical"
                n_votes = 0
                max_vote_fraction = 1.0
                if 0 <= label < n_classes:
                    dist[label] = 1.0
                else:
                    votes = []
                    for annotator in annotators:
                        if idx < len(annotator) and 0 <= annotator[idx] < n_classes:
                            votes.append(annotator[idx])
                    if not votes:
                        continue
                    for vote in votes:
                        dist[vote] += 1.0
                    n_votes = len(votes)
                    dist /= float(n_votes)
                    max_vote_fraction = float(dist.max())
                    source = "numbered"
                targets[(source_split, session_id, role, head, idx)] = {
                    "dist": dist,
                    "source": source,
                    "n_votes": n_votes,
                    "max_vote_fraction": max_vote_fraction,
                    "valid_classes": np.flatnonzero(dist > 0.0),
                }
    return targets


def soft_kappa(confusion: np.ndarray) -> float:
    n = float(confusion.sum())
    if n <= 0.0:
        return float("nan")
    observed = float(np.trace(confusion) / n)
    expected = float(confusion.sum(axis=1) @ confusion.sum(axis=0) / (n * n))
    return float((observed - expected) / (1.0 - expected)) if expected < 1.0 else float("nan")


def metrics_for_predictions(keys, logits, pred, targets, domain: str, mode: str, param: str) -> list[dict[str, object]]:
    rows = []
    for head in HEADS:
        n_classes = CLASS_COUNTS[head]
        confusion = np.zeros((n_classes, n_classes), dtype=np.float64)
        confusion_conf = np.zeros((n_classes, n_classes), dtype=np.float64)
        nll_sum = 0.0
        nll_conf_sum = 0.0
        weight_sum = 0.0
        conf_weight_sum = 0.0
        any_match = 0.0
        any_match_conf = 0.0
        canonical_frames = 0
        numbered_frames = 0
        for i, key in enumerate(keys):
            key_domain, source_split, session_id, role, key_head, frame_idx = key
            if key_domain != domain or key_head != head:
                continue
            target = targets.get((source_split, session_id, role, head, frame_idx))
            if target is None:
                continue
            dist = target["dist"]
            y_pred = int(pred[i])
            if not 0 <= y_pred < n_classes:
                continue
            probs = np.exp(logits[i, :n_classes] - np.max(logits[i, :n_classes]))
            probs /= probs.sum()
            nll = float(-(dist * np.log(np.clip(probs, 1.0e-12, 1.0))).sum())
            conf_w = float(target["max_vote_fraction"])
            confusion[:, y_pred] += dist
            confusion_conf[:, y_pred] += conf_w * dist
            nll_sum += nll
            nll_conf_sum += conf_w * nll
            weight_sum += 1.0
            conf_weight_sum += conf_w
            matched = float(y_pred in set(int(v) for v in target["valid_classes"]))
            any_match += matched
            any_match_conf += conf_w * matched
            if target["source"] == "canonical":
                canonical_frames += 1
            else:
                numbered_frames += 1
        if weight_sum == 0.0:
            continue
        rows.append(
            {
                "domain": domain,
                "mode": mode,
                "param": param,
                "head": head,
                "n_frames": int(weight_sum),
                "canonical_frames": canonical_frames,
                "numbered_frames": numbered_frames,
                "coverage_multiplier": float(weight_sum / max(canonical_frames, 1)),
                "soft_nll": float(nll_sum / weight_sum),
                "soft_nll_confidence_weighted": float(nll_conf_sum / conf_weight_sum) if conf_weight_sum else float("nan"),
                "expected_kappa": soft_kappa(confusion),
                "expected_kappa_confidence_weighted": soft_kappa(confusion_conf),
                "any_annotator_match_accuracy": float(any_match / weight_sum),
                "any_annotator_match_accuracy_confidence_weighted": float(any_match_conf / conf_weight_sum) if conf_weight_sum else float("nan"),
            }
        )
    return rows


def add_mean_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = list(rows)
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["domain"]), str(row["mode"]), str(row["param"]))].append(row)
    metric_names = [
        "soft_nll",
        "soft_nll_confidence_weighted",
        "expected_kappa",
        "expected_kappa_confidence_weighted",
        "any_annotator_match_accuracy",
        "any_annotator_match_accuracy_confidence_weighted",
    ]
    for (domain, mode, param), group in sorted(grouped.items()):
        if len(group) != len(HEADS):
            continue
        mean = {"domain": domain, "mode": mode, "param": param, "head": "mean"}
        mean["n_frames"] = sum(int(row["n_frames"]) for row in group)
        mean["canonical_frames"] = sum(int(row["canonical_frames"]) for row in group)
        mean["numbered_frames"] = sum(int(row["numbered_frames"]) for row in group)
        mean["coverage_multiplier"] = float(mean["n_frames"] / max(mean["canonical_frames"], 1))
        for metric in metric_names:
            values = [float(row[metric]) for row in group]
            mean[metric] = float(np.mean(values))
        out.append(mean)
    return out


def run_domain(domain: str, args: argparse.Namespace, targets) -> list[dict[str, object]]:
    train_keys, train_labels, _ = read_domain_data(domain, "train", args)
    val_keys, val_logits_by_feature = read_full_val_domain_data(domain, args)
    combined_logits = combine_two_head(val_keys, val_logits_by_feature, read_weights(domain, args))
    log_probs = log_softmax_by_head(val_keys, combined_logits)
    rows = metrics_for_predictions(val_keys, combined_logits, combined_logits.argmax(axis=1), targets, domain, "baseline", "none")
    for mix in args.transition_mixes:
        matrices = transition_matrices(train_keys, train_labels, args.transition_alpha, mix)
        for strength in args.transition_strengths:
            if strength == 0.0:
                continue
            pred = apply_hmm(val_keys, log_probs, matrices, strength)
            param = f"mix={mix:g};strength={strength:g};alpha={args.transition_alpha:g}"
            rows.extend(metrics_for_predictions(val_keys, combined_logits, pred, targets, domain, "hmm_uniform_start", param))
    return add_mean_rows(rows)


def combined_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["head"] == "mean":
            grouped[(str(row["mode"]), str(row["param"]))].append(row)
    out = []
    metrics = [
        "soft_nll",
        "soft_nll_confidence_weighted",
        "expected_kappa",
        "expected_kappa_confidence_weighted",
        "any_annotator_match_accuracy",
        "any_annotator_match_accuracy_confidence_weighted",
    ]
    for (mode, param), group in grouped.items():
        if len(group) != 2:
            continue
        by_domain = {str(row["domain"]): row for row in group}
        row = {
            "mode": mode,
            "param": param,
            "cc_frames": by_domain["CC"]["n_frames"],
            "cr_frames": by_domain["CR"]["n_frames"],
            "cc_numbered_frames": by_domain["CC"]["numbered_frames"],
            "cr_numbered_frames": by_domain["CR"]["numbered_frames"],
        }
        for metric in metrics:
            row[f"cc_{metric}"] = by_domain["CC"][metric]
            row[f"cr_{metric}"] = by_domain["CR"][metric]
            row[f"combined_{metric}"] = float(np.mean([float(by_domain["CC"][metric]), float(by_domain["CR"][metric])]))
        out.append(row)
    out.sort(key=lambda row: float(row["combined_expected_kappa"]), reverse=True)
    return out


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    targets = build_targets(raw_rows(args.raw_manifest), args.cache_root)
    rows = []
    for domain in ("CC", "CR"):
        domain_rows = run_domain(domain, args, targets)
        write_csv(args.output_root / f"{domain.lower()}_annotator_aware_metrics.csv", domain_rows)
        rows.extend(domain_rows)
    write_csv(args.output_root / "annotator_aware_metrics.csv", rows)
    combined = combined_rows(rows)
    write_csv(args.output_root / "combined_annotator_aware_metrics.csv", combined)
    print(json.dumps(combined[:10], indent=2), flush=True)


if __name__ == "__main__":
    main()
