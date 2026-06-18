"""Evaluate MoE1 two-head + HMM predictions against numbered annotator labels.

This keeps the current trained experts, combiner weights, and HMM transition
model fixed. The script exports full validation score rows, including canonical
blank frames, then swaps only the labels used for scoring to files such as
``purple.task_engagement.1.annotation.csv``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOE_ROOT = PROJECT_ROOT / "MoE"
EXPERIMENT_ROOT = MOE_ROOT / "experiments"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "moe1_hmm_annotator_label_eval"
DEFAULT_CACHE_ROOT = MOE_ROOT / "moe_data_soft_labels" / "cache" / "pinsoro"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MOE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOE_ROOT))

import ablate_moe1_hmm_decoding as hmm  # noqa: E402
from evaluate_moe1_metadata_head_checkpoint import (  # noqa: E402
    METADATA_FIELDS,
    MetadataWindowDataset,
    PinSoRoDyadicMetadataHeadTCN,
    make_loader,
    read_metadata,
    reconstruct,
)
from src.acm_pipeline.pinsoro import read_class_labels  # noqa: E402
from src.acm_pipeline.pinsoro_data import read_pinsoro_window_manifests  # noqa: E402
from src.acm_pipeline.pinsoro_train_utils import write_prediction_scores  # noqa: E402

FEATURES = hmm.FEATURES
CLASS_COUNTS = hmm.CLASS_COUNTS
HEAD_TO_FILE_STEM = {"task": "task_engagement", "social": "social_engagement"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score current MoE1 HMM predictions against numbered PinSoRo annotator labels."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--cc-expert-root", type=Path, default=EXPERIMENT_ROOT / "moe1_cc_metadata_head_experts")
    parser.add_argument("--cr-expert-root", type=Path, default=EXPERIMENT_ROOT / "moe1_cr_metadata_head_experts")
    parser.add_argument("--cc-combiner-root", type=Path, default=EXPERIMENT_ROOT / "moe1_cc_metadata_head_combiners")
    parser.add_argument("--cr-combiner-root", type=Path, default=EXPERIMENT_ROOT / "moe1_cr_metadata_head_combiners")
    parser.add_argument("--annotators", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--hmm-strength", type=float, default=8.0)
    parser.add_argument("--hmm-mix", type=float, default=1.0)
    parser.add_argument("--hmm-alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-cached-tensors", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


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


def expert_run_dir(root: Path, feature: str, domain: str) -> Path:
    return root / f"{domain.lower()}_{feature}_dyadic_tcn_k11_seed13"


def full_score_path(domain: str, feature: str, args: argparse.Namespace) -> Path:
    root = args.cc_expert_root if domain == "CC" else args.cr_expert_root
    run_name = expert_run_dir(root, feature, domain).name
    return args.output_root / "full_val_scores" / domain.lower() / run_name / "val_prediction_scores.csv.gz"


def export_full_scores(run_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    score_file = output_dir / "val_prediction_scores.csv.gz"
    if score_file.is_file() and score_file.stat().st_size > 0:
        return
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifests = [Path(item) for item in config["manifest"]]
    windows = read_pinsoro_window_manifests(manifests, PROJECT_ROOT, "val_internal")
    if not windows:
        raise RuntimeError(f"No val_internal windows found for {run_dir}")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    table = read_metadata(Path(config["metadata"]))
    mmap_config = config.get("mmap_cache_root")
    mmap_cache_root = None if mmap_config is None else Path(mmap_config)
    dataset = MetadataWindowDataset(
        windows,
        args.max_cached_tensors,
        mmap_cache_root,
        PROJECT_ROOT,
        metadata_table=table,
        metadata_stats=config["metadata_stats"],
        metadata_mode=config["metadata_mode"],
    )
    loader_args = Namespace(batch_size=args.batch_size, num_workers=args.num_workers, seed=int(config["seed"]))
    loader = make_loader(dataset, loader_args, shuffle=False, pin_memory=device.type == "cuda")
    model = PinSoRoDyadicMetadataHeadTCN(
        int(config["n_features_per_role"]),
        len(METADATA_FIELDS),
        int(config["hidden_channels"]),
        int(config["levels"]),
        int(config["kernel_size"]),
        float(config["dropout"]),
        float(config["metadata_dropout"]),
        bool(config.get("causal_tcn", True)),
    ).to(device)
    checkpoint = torch.load(run_dir / "model_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstructed = reconstruct(model, dataset, loader, device)
    write_prediction_scores(score_file, reconstructed, supervised_only=False)


def ensure_full_scores(domain: str, args: argparse.Namespace) -> None:
    root = args.cc_expert_root if domain == "CC" else args.cr_expert_root
    for feature in FEATURES:
        run_dir = expert_run_dir(root, feature, domain)
        export_full_scores(run_dir, full_score_path(domain, feature, args).parent, args)


def read_scores_allow_blank(path: Path) -> dict[hmm.Key, dict[str, object]]:
    rows: dict[hmm.Key, dict[str, object]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            head = row["head"]
            n_classes = CLASS_COUNTS[head]
            key = (row["domain"], row["source_split"], row["session_id"], row["role"], head, int(row["frame_idx"]))
            logits = np.full(hmm.MAX_CLASSES, -1.0e9, dtype=np.float64)
            logits[:n_classes] = [float(row[f"logit_{idx}"]) for idx in range(n_classes)]
            y_text = str(row.get("y_true", "")).strip()
            rows[key] = {"y_true": int(y_text) if y_text else -1, "logits": logits}
    return rows


def read_full_val_domain_data(domain: str, args: argparse.Namespace) -> tuple[list[hmm.Key], np.ndarray, dict[str, np.ndarray]]:
    by_feature = {feature: read_scores_allow_blank(full_score_path(domain, feature, args)) for feature in FEATURES}
    keys = sorted(set.intersection(*(set(rows) for rows in by_feature.values())))
    if not keys:
        raise RuntimeError(f"No aligned full validation rows for {domain}")
    labels = np.asarray([by_feature[FEATURES[0]][key]["y_true"] for key in keys], dtype=np.int64)
    logits = {feature: np.stack([by_feature[feature][key]["logits"] for key in keys]) for feature in FEATURES}
    return keys, labels, logits


def annotation_path(cache_root: Path, key: hmm.Key, annotator: int | None) -> Path:
    _domain, source_split, session_id, role, head, _frame_idx = key
    stem = HEAD_TO_FILE_STEM[head]
    name = f"{role}.{stem}.annotation.csv" if annotator is None else f"{role}.{stem}.{annotator}.annotation.csv"
    return cache_root / source_split / session_id / name


def load_label_file(path: Path, head: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    return read_class_labels(path, HEAD_TO_FILE_STEM[head])


def labels_from_files(keys: list[hmm.Key], cache_root: Path, annotator: int | None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    labels = np.full(len(keys), -1, dtype=np.int64)
    valid = np.zeros(len(keys), dtype=bool)
    cache: dict[tuple[Path, str], tuple[np.ndarray, np.ndarray] | None] = {}
    file_paths: set[Path] = set()
    present_files: set[Path] = set()
    missing_files: set[Path] = set()
    out_of_range = 0
    for idx, key in enumerate(keys):
        path = annotation_path(cache_root, key, annotator)
        file_paths.add(path)
        cache_key = (path, key[4])
        if cache_key not in cache:
            cache[cache_key] = load_label_file(path, key[4])
        loaded = cache[cache_key]
        if loaded is None:
            missing_files.add(path)
            continue
        present_files.add(path)
        file_labels, file_mask = loaded
        frame_idx = key[5]
        if frame_idx >= len(file_labels):
            out_of_range += 1
            continue
        if file_mask[frame_idx] > 0:
            labels[idx] = int(file_labels[frame_idx])
            valid[idx] = True
    return labels, valid, {
        "candidate_files": len(file_paths),
        "present_files": len(present_files),
        "missing_files": len(missing_files),
        "out_of_range_rows": out_of_range,
        "valid_rows": int(valid.sum()),
        "invalid_or_blank_rows": int((~valid).sum()),
    }


def subset(keys: list[hmm.Key], labels: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> tuple[list[hmm.Key], np.ndarray, np.ndarray]:
    idx = np.flatnonzero(valid)
    return [keys[i] for i in idx], labels[idx], pred[idx]


def add_coverage(rows: list[dict[str, object]], coverage: dict[str, object]) -> list[dict[str, object]]:
    return [{**row, **coverage} for row in rows]


def score_label_source(
    keys: list[hmm.Key],
    labels: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    domain: str,
    label_source: str,
    mode: str,
    param: str,
    coverage: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sub_keys, sub_labels, sub_pred = subset(keys, labels, pred, valid)
    if not sub_keys:
        return [{
            "domain": domain,
            "label_source": label_source,
            "mode": mode,
            "param": param,
            "head": "mean",
            "n_frames": 0,
            "accuracy": float("nan"),
            "kappa": float("nan"),
            **coverage,
        }], []
    metric_rows = hmm.add_mean_rows(hmm.evaluate(sub_keys, sub_labels, sub_pred, domain, mode, param))
    class_rows = hmm.class_metric_rows(sub_keys, sub_labels, sub_pred, domain, mode, param)
    for row in metric_rows:
        row["label_source"] = label_source
    for row in class_rows:
        row["label_source"] = label_source
    return add_coverage(metric_rows, coverage), add_coverage(class_rows, coverage)


def canonical_valid(labels: np.ndarray) -> np.ndarray:
    return labels >= 0


def predictions_for_domain(domain: str, args: argparse.Namespace) -> tuple[list[hmm.Key], np.ndarray, np.ndarray, np.ndarray]:
    train_keys, train_labels, _train_logits = hmm.read_domain_data(domain, "train", args)
    ensure_full_scores(domain, args)
    val_keys, val_labels, val_logits_by_feature = read_full_val_domain_data(domain, args)
    combined_logits = hmm.combine_two_head(val_keys, val_logits_by_feature, hmm.read_weights(domain, args))
    log_probs = hmm.log_softmax_by_head(val_keys, combined_logits)
    matrices = hmm.transition_matrices(train_keys, train_labels, args.hmm_alpha, args.hmm_mix)
    return val_keys, val_labels, combined_logits.argmax(axis=1), hmm.apply_hmm(val_keys, log_probs, matrices, args.hmm_strength)


def mean_or_nan(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def combined_summary(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        if row.get("head") == "mean":
            groups[(str(row["label_source"]), str(row["mode"]), str(row["param"]))].append(row)
    rows: list[dict[str, object]] = []
    for (label_source, mode, param), group in sorted(groups.items()):
        by_domain = {str(row["domain"]): row for row in group}
        cc = by_domain.get("CC")
        cr = by_domain.get("CR")
        kappas = [float(row["kappa"]) for row in (cc, cr) if row is not None]
        rows.append({
            "label_source": label_source,
            "mode": mode,
            "param": param,
            "cc_mean_kappa": "" if cc is None else cc["kappa"],
            "cr_mean_kappa": "" if cr is None else cr["kappa"],
            "combined_mean_kappa": mean_or_nan(kappas),
            "cc_n_frames": "" if cc is None else cc["n_frames"],
            "cr_n_frames": "" if cr is None else cr["n_frames"],
            "total_n_frames": sum(int(row["n_frames"]) for row in group),
            "total_valid_rows": sum(int(row["valid_rows"]) for row in group),
            "total_blank_or_invalid_rows": sum(int(row["invalid_or_blank_rows"]) for row in group),
        })
    rows.sort(key=lambda row: (str(row["label_source"]), str(row["mode"])))
    return rows


def run_domain(domain: str, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    keys, canonical_labels, baseline_pred, hmm_pred = predictions_for_domain(domain, args)
    all_rows: list[dict[str, object]] = []
    all_class_rows: list[dict[str, object]] = []
    param = f"mix={args.hmm_mix:g};strength={args.hmm_strength:g};alpha={args.hmm_alpha:g}"
    label_sources = [("canonical", *labels_from_files(keys, args.cache_root, None))]
    label_sources.extend(
        (f"annotator_{annotator}", *labels_from_files(keys, args.cache_root, annotator))
        for annotator in args.annotators
    )
    for label_source, labels, valid, coverage in label_sources:
        for mode, pred in (("baseline", baseline_pred), ("hmm_uniform_start", hmm_pred)):
            rows, class_rows = score_label_source(
                keys,
                labels,
                pred,
                valid,
                domain,
                label_source,
                mode,
                "none" if mode == "baseline" else param,
                coverage,
            )
            all_rows.extend(rows)
            all_class_rows.extend(class_rows)
    return all_rows, all_class_rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    for domain in ("CC", "CR"):
        domain_rows, domain_class_rows = run_domain(domain, args)
        metric_rows.extend(domain_rows)
        class_rows.extend(domain_class_rows)
        write_csv(args.output_root / f"{domain.lower()}_metrics_by_label_source.csv", domain_rows)
        write_csv(args.output_root / f"{domain.lower()}_class_metrics_by_label_source.csv", domain_class_rows)
    summary_rows = combined_summary(metric_rows)
    write_csv(args.output_root / "metrics_by_label_source.csv", metric_rows)
    write_csv(args.output_root / "class_metrics_by_label_source.csv", class_rows)
    write_csv(args.output_root / "combined_summary.csv", summary_rows)
    (args.output_root / "config.json").write_text(json.dumps({
        "cache_root": str(args.cache_root),
        "cc_expert_root": str(args.cc_expert_root),
        "cr_expert_root": str(args.cr_expert_root),
        "cc_combiner_root": str(args.cc_combiner_root),
        "cr_combiner_root": str(args.cr_combiner_root),
        "annotators": args.annotators,
        "hmm_strength": args.hmm_strength,
        "hmm_mix": args.hmm_mix,
        "hmm_alpha": args.hmm_alpha,
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
