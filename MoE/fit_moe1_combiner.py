"""Fit/evaluate MoE 1 combiner ablations from frozen expert scores."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections.abc import Callable, Hashable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "MoE" / "experiments"
DEFAULT_METADATA = PROJECT_ROOT / "MoE" / "moe_data" / "outputs" / "participant_metadata.csv"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
CLASS_COUNTS = {"task": 4, "social": 5}
MAX_CLASSES = max(CLASS_COUNTS.values())
HEADS = ("task", "social")
METADATA_FEATURES = ("bias", "age_z", "gender_1", "gender_2", "gender_unknown")
DEFAULT_MODES = (
    "best_single",
    "uniform",
    "prob_uniform",
    "shared",
    "prob_shared",
    "two_head",
    "prob_two_head",
    "role_head",
    "metadata_router",
    "val_shared_upper",
    "val_two_head_upper",
    "val_role_head_upper",
    "val_metadata_router_upper",
)
Key = tuple[str, str, str, str, str, int]
GroupFn = Callable[[Key], Hashable]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoE 1 combiner ablations for CC or CR.")
    parser.add_argument("--domain", choices=("CC", "CR", "cc", "cr"), default="CC")
    parser.add_argument("--expert-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--metadata-router-epochs", type=int, default=80)
    parser.add_argument("--metadata-router-lr", type=float, default=0.05)
    parser.add_argument("--metadata-router-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--metadata-router-batch-size", type=int, default=8192)
    parser.add_argument("--metadata-router-max-rows", type=int, default=200000)
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=DEFAULT_MODES)
    args = parser.parse_args()
    args.domain = args.domain.upper()
    domain_lower = args.domain.lower()
    if args.expert_root is None:
        args.expert_root = EXPERIMENT_ROOT / f"moe1_{domain_lower}_experts"
    if args.output_root is None:
        args.output_root = EXPERIMENT_ROOT / f"moe1_{domain_lower}_combiners"
    return args


def expert_run_dir(root: Path, feature: str, domain: str) -> Path:
    return root / f"{domain.lower()}_{feature}_dyadic_tcn_k11_seed13"


def score_path(root: Path, feature: str, split: str, domain: str) -> Path:
    run = expert_run_dir(root, feature, domain)
    if split == "train":
        return run / "diagnostics" / "train_internal" / "val_prediction_scores.csv.gz"
    if split == "val":
        return run / "val_prediction_scores.csv.gz"
    raise ValueError(split)


def read_scores(path: Path) -> dict[Key, dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: dict[Key, dict[str, object]] = {}
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


def aligned_split(
    root: Path, split: str, domain: str
) -> tuple[list[Key], np.ndarray, dict[str, np.ndarray]]:
    by_feature = {feature: read_scores(score_path(root, feature, split, domain)) for feature in FEATURES}
    keys = sorted(set.intersection(*(set(rows) for rows in by_feature.values())))
    if not keys:
        raise RuntimeError(f"No aligned score rows for split {split}.")
    labels = np.asarray([by_feature[FEATURES[0]][key]["y_true"] for key in keys], dtype=np.int64)
    logits = {
        feature: np.stack([by_feature[feature][key]["logits"] for key in keys])
        for feature in FEATURES
    }
    return keys, labels, logits


def read_metadata(path: Path) -> dict[tuple[str, str, str], tuple[float | None, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing metadata table: {path}. Run ACM/MoE/prepare_moe_metadata.py first."
        )
    table: dict[tuple[str, str, str], tuple[float | None, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            age = None if row.get("age", "") == "" else float(row["age"])
            gender = str(row.get("gender", "")).strip()
            table[(row["source_split"], row["session_id"], row["role"])] = (age, gender)
    return table


def metadata_stats(keys: list[Key], table: dict[tuple[str, str, str], tuple[float | None, str]]) -> dict[str, float]:
    seen: set[tuple[str, str, str]] = set()
    ages = []
    for key in keys:
        meta_key = (key[1], key[2], key[3])
        if meta_key in seen:
            continue
        seen.add(meta_key)
        age, _gender = table.get(meta_key, (None, ""))
        if age is not None:
            ages.append(age)
    if not ages:
        raise RuntimeError("No age metadata found for fitting split.")
    mean = float(np.mean(ages))
    std = float(np.std(ages))
    return {"age_mean": mean, "age_std": std if std > 1.0e-6 else 1.0}


def metadata_matrix(
    keys: list[Key],
    table: dict[tuple[str, str, str], tuple[float | None, str]],
    stats: dict[str, float],
) -> np.ndarray:
    values = np.zeros((len(keys), len(METADATA_FEATURES)), dtype=np.float32)
    values[:, 0] = 1.0
    missing = 0
    for idx, key in enumerate(keys):
        meta_key = (key[1], key[2], key[3])
        age, gender = table.get(meta_key, (None, ""))
        if age is None:
            missing += 1
            age_z = 0.0
        else:
            age_z = (age - stats["age_mean"]) / stats["age_std"]
        values[idx, 1] = age_z
        if gender == "1":
            values[idx, 2] = 1.0
        elif gender == "2":
            values[idx, 3] = 1.0
        else:
            values[idx, 4] = 1.0
    if missing:
        print(f"metadata warning: {missing} rows missing age; using train mean", flush=True)
    return values


def simplex_weights(step: float) -> list[np.ndarray]:
    if not 0.0 < step <= 1.0:
        raise ValueError("step must be in (0, 1].")
    n = round(1.0 / step)
    if not math.isclose(n * step, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("step must divide 1.0 exactly enough for a simplex grid.")
    weights = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            c = n - a - b
            weights.append(np.asarray([a, b, c], dtype=np.float64) / n)
    return weights


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def combine_logits(logits: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    return sum(float(weight) * logits[feature] for weight, feature in zip(weights, FEATURES))


def combine_probs(logits: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    return sum(float(weight) * softmax_np(logits[feature]) for weight, feature in zip(weights, FEATURES))


def cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def cross_entropy_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probs[np.arange(len(labels)), labels], 1.0e-12, 1.0)
    return float(-np.log(clipped).mean())


def kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    n = confusion.sum()
    if n == 0:
        return float("nan")
    accuracy = np.trace(confusion) / n
    expected = confusion.sum(axis=1) @ confusion.sum(axis=0) / (n * n)
    return float((accuracy - expected) / (1.0 - expected)) if expected < 1.0 else float("nan")


def indices_for(keys: list[Key], predicate: Callable[[Key], bool]) -> np.ndarray:
    return np.asarray([idx for idx, key in enumerate(keys) if predicate(key)], dtype=np.int64)


def fit_group_weights(
    keys: list[Key],
    labels: np.ndarray,
    logits: dict[str, np.ndarray],
    weights_grid: list[np.ndarray],
    group_fn: GroupFn,
    combine_kind: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    groups = sorted({group_fn(key) for key in keys}, key=str)
    weights_by_group: dict[str, np.ndarray] = {}
    loss_by_group: dict[str, float] = {}
    for group in groups:
        idx = indices_for(keys, lambda key, group=group: group_fn(key) == group)
        group_logits = {feature: values[idx] for feature, values in logits.items()}
        group_probs = (
            {feature: softmax_np(values) for feature, values in group_logits.items()}
            if combine_kind == "prob"
            else {}
        )
        group_labels = labels[idx]
        best_weight = weights_grid[0]
        best_loss = float("inf")
        for weights in weights_grid:
            if combine_kind == "logit":
                loss = cross_entropy_from_logits(combine_logits(group_logits, weights), group_labels)
            elif combine_kind == "prob":
                combined = sum(float(weight) * group_probs[feature] for weight, feature in zip(weights, FEATURES))
                loss = cross_entropy_from_probs(combined, group_labels)
            else:
                raise ValueError(combine_kind)
            if loss < best_loss:
                best_loss = loss
                best_weight = weights
        group_name = str(group)
        weights_by_group[group_name] = best_weight
        loss_by_group[group_name] = best_loss
    return weights_by_group, loss_by_group


def predict_grouped(
    keys: list[Key],
    logits: dict[str, np.ndarray],
    weights_by_group: dict[str, np.ndarray],
    group_fn: GroupFn,
    combine_kind: str,
) -> np.ndarray:
    pred = np.empty(len(keys), dtype=np.int64)
    groups = sorted({group_fn(key) for key in keys}, key=str)
    for group in groups:
        idx = indices_for(keys, lambda key, group=group: group_fn(key) == group)
        group_logits = {feature: values[idx] for feature, values in logits.items()}
        weights = weights_by_group[str(group)]
        if combine_kind == "logit":
            combined = combine_logits(group_logits, weights)
        elif combine_kind == "prob":
            combined = combine_probs(group_logits, weights)
        else:
            raise ValueError(combine_kind)
        pred[idx] = combined.argmax(axis=1)
    return pred


def fit_metadata_router(
    args: argparse.Namespace,
    keys: list[Key],
    labels: np.ndarray,
    logits: dict[str, np.ndarray],
    metadata: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, list[float]]]:
    rng = np.random.default_rng(args.seed)
    params: dict[str, np.ndarray] = {}
    losses: dict[str, float] = {}
    mean_weights: dict[str, list[float]] = {}
    for head in HEADS:
        idx = indices_for(keys, lambda key, head=head: key[4] == head)
        if args.metadata_router_max_rows > 0 and len(idx) > args.metadata_router_max_rows:
            idx = np.sort(rng.choice(idx, size=args.metadata_router_max_rows, replace=False))
        x = torch.as_tensor(metadata[idx], dtype=torch.float32)
        y = torch.as_tensor(labels[idx], dtype=torch.long)
        stacked = np.stack([logits[feature][idx] for feature in FEATURES], axis=1).astype(np.float32)
        expert_logits = torch.as_tensor(stacked, dtype=torch.float32)
        weight = torch.zeros((x.shape[1], len(FEATURES)), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([weight], lr=args.metadata_router_lr)
        batch_size = max(1, args.metadata_router_batch_size)
        last_loss = float("nan")
        for _epoch in range(args.metadata_router_epochs):
            order = torch.randperm(x.shape[0])
            for start in range(0, x.shape[0], batch_size):
                batch_idx = order[start : start + batch_size]
                gate = torch.softmax(x[batch_idx] @ weight, dim=1)
                combined = (gate.unsqueeze(-1) * expert_logits[batch_idx]).sum(dim=1)
                loss = F.cross_entropy(combined, y[batch_idx])
                if args.metadata_router_weight_decay > 0:
                    loss = loss + args.metadata_router_weight_decay * weight.square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())
        with torch.no_grad():
            gate = torch.softmax(x @ weight, dim=1).cpu().numpy()
        params[head] = weight.detach().cpu().numpy()
        losses[head] = last_loss
        mean_weights[head] = gate.mean(axis=0).tolist()
    return params, losses, mean_weights


def predict_metadata_router(
    keys: list[Key],
    logits: dict[str, np.ndarray],
    metadata: np.ndarray,
    params: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, list[float]]]:
    pred = np.empty(len(keys), dtype=np.int64)
    mean_weights: dict[str, list[float]] = {}
    for head in HEADS:
        idx = indices_for(keys, lambda key, head=head: key[4] == head)
        x = torch.as_tensor(metadata[idx], dtype=torch.float32)
        weight = torch.as_tensor(params[head], dtype=torch.float32)
        stacked = np.stack([logits[feature][idx] for feature in FEATURES], axis=1).astype(np.float32)
        expert_logits = torch.as_tensor(stacked, dtype=torch.float32)
        with torch.no_grad():
            gate = torch.softmax(x @ weight, dim=1)
            combined = (gate.unsqueeze(-1) * expert_logits).sum(dim=1)
        pred[idx] = combined.argmax(dim=1).cpu().numpy()
        mean_weights[head] = gate.mean(dim=0).cpu().numpy().tolist()
    return pred, mean_weights


def metric_rows(
    keys: list[Key], labels: np.ndarray, pred: np.ndarray, mode: str, feature: str | None = None
) -> tuple[list[dict[str, object]], float]:
    rows = []
    kappas = []
    for head in HEADS:
        idx = indices_for(keys, lambda key, head=head: key[4] == head)
        score = kappa(labels[idx], pred[idx], CLASS_COUNTS[head])
        kappas.append(score)
        rows.append(
            {
                "domain": keys[idx[0]][0] if len(idx) else "",
                "mode": mode,
                "feature": feature or "combined",
                "head": head,
                "n_frames": int(len(idx)),
                "kappa": score,
                "accuracy": float((labels[idx] == pred[idx]).mean()),
            }
        )
    return rows, float(np.nanmean(kappas))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def weights_to_json(weights_by_group: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {group: weights.tolist() for group, weights in weights_by_group.items()}


def best_single(val_keys: list[Key], val_labels: np.ndarray, val_logits: dict[str, np.ndarray]) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    scores = {}
    for feature in FEATURES:
        pred = val_logits[feature].argmax(axis=1)
        feature_rows, score = metric_rows(val_keys, val_labels, pred, "best_single", feature)
        rows.extend(feature_rows)
        scores[feature] = score
    best_feature = max(scores, key=scores.get)
    summary = {
        "mode": "best_single",
        "val_mean_kappa": scores[best_feature],
        "best_feature": best_feature,
        "feature_val_mean_kappa": scores,
    }
    return rows, summary


def mode_config(mode: str) -> tuple[str, GroupFn, str, str]:
    if mode == "uniform":
        return "val", lambda _key: "all", "logit", "fixed"
    if mode == "prob_uniform":
        return "val", lambda _key: "all", "prob", "fixed"
    if mode == "shared":
        return "train", lambda _key: "all", "logit", "fit"
    if mode == "prob_shared":
        return "train", lambda _key: "all", "prob", "fit"
    if mode == "two_head":
        return "train", lambda key: key[4], "logit", "fit"
    if mode == "prob_two_head":
        return "train", lambda key: key[4], "prob", "fit"
    if mode == "role_head":
        return "train", lambda key: f"{key[3]}_{key[4]}", "logit", "fit"
    if mode == "val_shared_upper":
        return "val", lambda _key: "all", "logit", "fit"
    if mode == "val_two_head_upper":
        return "val", lambda key: key[4], "logit", "fit"
    if mode == "val_role_head_upper":
        return "val", lambda key: f"{key[3]}_{key[4]}", "logit", "fit"
    raise ValueError(mode)


def run_metadata_mode(
    args: argparse.Namespace,
    mode: str,
    train_data: tuple[list[Key], np.ndarray, dict[str, np.ndarray]],
    val_data: tuple[list[Key], np.ndarray, dict[str, np.ndarray]],
) -> dict[str, object]:
    train_keys, train_labels, train_logits = train_data
    val_keys, val_labels, val_logits = val_data
    metadata_table = read_metadata(args.metadata)
    fit_keys = val_keys if mode.startswith("val_") else train_keys
    fit_labels = val_labels if mode.startswith("val_") else train_labels
    fit_logits = val_logits if mode.startswith("val_") else train_logits
    stats = metadata_stats(fit_keys, metadata_table)
    fit_meta = metadata_matrix(fit_keys, metadata_table, stats)
    val_meta = metadata_matrix(val_keys, metadata_table, stats)
    params, fit_loss_by_head, fit_mean_weights = fit_metadata_router(
        args, fit_keys, fit_labels, fit_logits, fit_meta
    )
    pred, val_mean_weights = predict_metadata_router(val_keys, val_logits, val_meta, params)
    rows, val_score = metric_rows(val_keys, val_labels, pred, mode)
    for row in rows:
        row["mean_val_weights"] = json.dumps({str(row["head"]): val_mean_weights[str(row["head"])]})
    output_dir = args.output_root / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics_by_domain.csv", rows)
    summary = {
        "mode": mode,
        "domain": args.domain,
        "features": list(FEATURES),
        "metadata_features": list(METADATA_FEATURES),
        "metadata_path": str(args.metadata),
        "fit_split": "val" if mode.startswith("val_") else "train",
        "optimistic_upper_bound": mode.startswith("val_"),
        "metadata_stats": stats,
        "metadata_router_epochs": args.metadata_router_epochs,
        "metadata_router_lr": args.metadata_router_lr,
        "metadata_router_weight_decay": args.metadata_router_weight_decay,
        "metadata_router_max_rows": args.metadata_router_max_rows,
        "fit_loss_by_head": fit_loss_by_head,
        "fit_mean_weights_by_head": fit_mean_weights,
        "val_mean_weights_by_head": val_mean_weights,
        "coefficients_by_head": {head: params[head].tolist() for head in HEADS},
        "val_mean_kappa": val_score,
        "n_train_rows": len(train_keys),
        "n_val_rows": len(val_keys),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_mode(
    args: argparse.Namespace,
    mode: str,
    train_data: tuple[list[Key], np.ndarray, dict[str, np.ndarray]],
    val_data: tuple[list[Key], np.ndarray, dict[str, np.ndarray]],
) -> dict[str, object]:
    train_keys, train_labels, train_logits = train_data
    val_keys, val_labels, val_logits = val_data
    output_dir = args.output_root / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "best_single":
        rows, summary = best_single(val_keys, val_labels, val_logits)
        summary.update({"domain": args.domain, "features": list(FEATURES), "n_val_rows": len(val_keys)})
        write_csv(output_dir / "metrics_by_domain.csv", rows)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    if mode in {"metadata_router", "val_metadata_router_upper"}:
        return run_metadata_mode(args, mode, train_data, val_data)

    fit_split, group_fn, combine_kind, action = mode_config(mode)
    if action == "fixed":
        weights_by_group = {"all": np.asarray([1.0 / 3.0] * 3, dtype=np.float64)}
        fit_loss_by_group: dict[str, float] = {}
    else:
        grid = simplex_weights(args.step)
        fit_keys, fit_labels, fit_logits = train_data if fit_split == "train" else val_data
        weights_by_group, fit_loss_by_group = fit_group_weights(
            fit_keys, fit_labels, fit_logits, grid, group_fn, combine_kind
        )

    pred = predict_grouped(val_keys, val_logits, weights_by_group, group_fn, combine_kind)
    rows, val_score = metric_rows(val_keys, val_labels, pred, mode)
    for row in rows:
        head = str(row["head"])
        matching = {
            group: weights
            for group, weights in weights_by_group.items()
            if group == "all" or group.endswith(f"_{head}") or group == head
        }
        row["weights"] = json.dumps(weights_to_json(matching), sort_keys=True)
    write_csv(output_dir / "metrics_by_domain.csv", rows)
    summary = {
        "mode": mode,
        "domain": args.domain,
        "features": list(FEATURES),
        "step": args.step,
        "combine_kind": combine_kind,
        "fit_split": fit_split if action == "fit" else None,
        "optimistic_upper_bound": mode.startswith("val_"),
        "fit_loss_by_group": fit_loss_by_group,
        "val_mean_kappa": val_score,
        "weights_by_group": weights_to_json(weights_by_group),
        "n_train_rows": len(train_keys),
        "n_val_rows": len(val_keys),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    train_data = aligned_split(args.expert_root, "train", args.domain)
    val_data = aligned_split(args.expert_root, "val", args.domain)
    summaries = [run_mode(args, mode, train_data, val_data) for mode in args.modes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "comparison.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    for summary in summaries:
        print(
            f"{summary['mode']} {summary['domain']} val_mean_kappa={summary['val_mean_kappa']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
