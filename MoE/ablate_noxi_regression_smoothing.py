"""Continuous temporal smoothing ablations for NOXI metadata-head MoE outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import fit_noxi_moe1_combiner as base


EXPERIMENT_ROOT = Path(__file__).resolve().parent / "experiments"
FEATURES = base.FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate continuous smoothers for NOXI regression MoE outputs.")
    parser.add_argument("--corpus", choices=("noxi", "noxi_j"), default="noxi")
    parser.add_argument("--expert-root", type=Path)
    parser.add_argument("--combiner-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--branch", choices=("metadata_head", "plain"), default="metadata_head")
    parser.add_argument("--include-upper", action="store_true")
    return parser.parse_args()


def prediction_path(root: Path, corpus: str, feature: str, split: str, branch: str) -> Path:
    suffix = "_metadata_head" if branch == "metadata_head" else ""
    run = root / f"{corpus}_{feature}_dyadic_tcn_k11{suffix}_seed13"
    if split == "train":
        return run / "diagnostics" / "train_internal" / "val_predictions.csv"
    if split == "val":
        return run / "val_predictions.csv"
    raise ValueError(split)


def defaults(args: argparse.Namespace) -> argparse.Namespace:
    suffix = "_metadata_head" if args.branch == "metadata_head" else ""
    if args.expert_root is None:
        args.expert_root = EXPERIMENT_ROOT / f"noxi_moe1_{args.corpus}{suffix}_experts"
    if args.combiner_root is None:
        args.combiner_root = EXPERIMENT_ROOT / f"noxi_moe1_{args.corpus}{suffix}_combiners"
    if args.output_root is None:
        args.output_root = EXPERIMENT_ROOT / f"noxi_moe1_{args.corpus}{suffix}_smoothing_ablation"
    return args


def read_predictions(path: Path) -> dict[base.Key, dict[str, float]]:
    return base.read_predictions(path)


def aligned_split(args: argparse.Namespace, split: str) -> tuple[list[base.Key], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    by_feature = {
        feature: read_predictions(prediction_path(args.expert_root, args.corpus, feature, split, args.branch))
        for feature in FEATURES
    }
    keys = sorted(set.intersection(*(set(rows) for rows in by_feature.values())))
    if not keys:
        raise RuntimeError(f"No aligned prediction rows for {args.corpus} {split}")
    reference = by_feature[FEATURES[0]]
    labels = np.asarray([reference[key]["y_true"] for key in keys], dtype=np.float64)
    mask = np.asarray(
        [
            reference[key]["target_mask"] > 0.0
            and all(by_feature[feature][key]["covered"] > 0.0 for feature in FEATURES)
            for key in keys
        ],
        dtype=bool,
    )
    predictions = {
        feature: np.asarray([by_feature[feature][key]["y_pred"] for key in keys], dtype=np.float64)
        for feature in FEATURES
    }
    finite = np.isfinite(labels)
    for values in predictions.values():
        finite &= np.isfinite(values)
    return keys, labels, mask & finite, predictions


def read_mode_weights(path: Path) -> dict[str, dict[str, np.ndarray]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        mode: {group: np.asarray(weights, dtype=np.float64) for group, weights in groups.items()}
        for mode, groups in data["weights"].items()
    }


def best_single(summary_path: Path) -> str:
    with summary_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    singles = [row for row in rows if row["mode"].startswith("single_")]
    if not singles:
        raise RuntimeError(f"No single expert rows in {summary_path}")
    best = max(singles, key=lambda row: float(row["ccc"]))
    return best["mode"].removeprefix("single_")


def combiner_predictions(
    keys: list[base.Key],
    predictions: dict[str, np.ndarray],
    weights_by_mode: dict[str, dict[str, np.ndarray]],
    best_feature: str,
    include_upper: bool,
) -> dict[str, np.ndarray]:
    out = {
        f"single_{feature}": values.copy()
        for feature, values in predictions.items()
    }
    out["best_single"] = predictions[best_feature].copy()
    out["uniform"] = base.combine(predictions, np.asarray([1.0 / len(FEATURES)] * len(FEATURES), dtype=np.float64))
    if "shared" in weights_by_mode:
        out["shared"] = base.predict_grouped(keys, predictions, weights_by_mode["shared"], lambda _key: "all")
    if "role" in weights_by_mode:
        out["role"] = base.predict_grouped(keys, predictions, weights_by_mode["role"], lambda key: key[2])
    if include_upper:
        if "val_shared_upper" in weights_by_mode:
            out["val_shared_upper"] = base.predict_grouped(keys, predictions, weights_by_mode["val_shared_upper"], lambda _key: "all")
        if "val_role_upper" in weights_by_mode:
            out["val_role_upper"] = base.predict_grouped(keys, predictions, weights_by_mode["val_role_upper"], lambda key: key[2])
    return out


def sequence_indices(keys: list[base.Key]) -> list[np.ndarray]:
    groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        dataset, session_id, role, target_channel, frame_idx = key
        groups[(f"{dataset}/{session_id}", role, target_channel)].append(idx)
    return [
        np.asarray(sorted(idxs, key=lambda idx: keys[idx][4]), dtype=np.int64)
        for idxs in groups.values()
    ]


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values)
    if len(values) == 0:
        return out
    out[0] = values[0]
    for idx in range(1, len(values)):
        out[idx] = alpha * values[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def moving_average(values: np.ndarray, radius: int, causal: bool) -> np.ndarray:
    out = np.empty_like(values)
    for idx in range(len(values)):
        start = max(0, idx - radius)
        stop = idx + 1 if causal else min(len(values), idx + radius + 1)
        out[idx] = np.mean(values[start:stop])
    return out


def median_filter(values: np.ndarray, radius: int, causal: bool) -> np.ndarray:
    out = np.empty_like(values)
    for idx in range(len(values)):
        start = max(0, idx - radius)
        stop = idx + 1 if causal else min(len(values), idx + radius + 1)
        out[idx] = np.median(values[start:stop])
    return out


def deadband(values: np.ndarray, threshold: float, alpha: float) -> np.ndarray:
    out = np.empty_like(values)
    if len(values) == 0:
        return out
    out[0] = values[0]
    for idx in range(1, len(values)):
        delta = values[idx] - out[idx - 1]
        if abs(delta) < threshold:
            out[idx] = out[idx - 1]
        else:
            out[idx] = out[idx - 1] + alpha * delta
    return out


def apply_smoother(pred: np.ndarray, groups: list[np.ndarray], kind: str, param: str) -> np.ndarray:
    out = pred.copy()
    for idx in groups:
        values = pred[idx]
        if kind == "identity":
            smoothed = values
        elif kind == "ema":
            smoothed = ema(values, float(param))
        elif kind == "causal_mean":
            smoothed = moving_average(values, int(param), causal=True)
        elif kind == "center_mean":
            smoothed = moving_average(values, int(param), causal=False)
        elif kind == "causal_median":
            smoothed = median_filter(values, int(param), causal=True)
        elif kind == "center_median":
            smoothed = median_filter(values, int(param), causal=False)
        elif kind == "deadband":
            threshold, alpha = (float(part) for part in param.split(":"))
            smoothed = deadband(values, threshold, alpha)
        else:
            raise ValueError(kind)
        out[idx] = np.clip(smoothed, 0.0, 1.0)
    return out


def smoother_grid() -> list[tuple[str, str]]:
    grid = [("identity", "none")]
    grid.extend(("ema", f"{alpha:g}") for alpha in (0.05, 0.1, 0.2, 0.35, 0.5, 0.7))
    grid.extend((kind, str(radius)) for kind in ("causal_mean", "center_mean", "causal_median", "center_median") for radius in (5, 12, 25, 50, 100))
    grid.extend(("deadband", f"{threshold:g}:{alpha:g}") for threshold in (0.005, 0.01, 0.02, 0.05) for alpha in (0.2, 0.5, 1.0))
    return grid


def evaluate(keys: list[base.Key], y_true: np.ndarray, mask: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    return base.evaluate(keys, y_true, mask, pred)


def per_session_rows(
    keys: list[base.Key],
    y_true: np.ndarray,
    mask: np.ndarray,
    pred: np.ndarray,
    prefix: dict[str, object],
) -> list[dict[str, object]]:
    rows = []
    sessions = sorted({(key[0], key[1]) for key in keys})
    for dataset, session_id in sessions:
        session_mask = mask & np.asarray([key[0] == dataset and key[1] == session_id for key in keys], dtype=bool)
        row = base.metric_row({"dataset": dataset, "session_id": session_id}, y_true, pred, session_mask)
        rows.append({**prefix, **row})
    return rows


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
    args = defaults(parse_args())
    args.output_root.mkdir(parents=True, exist_ok=True)
    keys, y_true, mask, feature_predictions = aligned_split(args, "val")
    groups = sequence_indices(keys)
    weights_by_mode = read_mode_weights(args.combiner_root / "weights.json")
    selected = combiner_predictions(
        keys,
        feature_predictions,
        weights_by_mode,
        best_single(args.combiner_root / "summary.csv"),
        args.include_upper,
    )

    rows = []
    session_rows = []
    for source_mode, pred in selected.items():
        for smoother, param in smoother_grid():
            smoothed = apply_smoother(pred, groups, smoother, param)
            row = {
                "corpus": args.corpus,
                "branch": args.branch,
                "source_mode": source_mode,
                "smoother": smoother,
                "param": param,
                **evaluate(keys, y_true, mask, smoothed),
            }
            rows.append(row)
            session_rows.extend(
                per_session_rows(
                    keys,
                    y_true,
                    mask,
                    smoothed,
                    {
                        "corpus": args.corpus,
                        "branch": args.branch,
                        "source_mode": source_mode,
                        "smoother": smoother,
                        "param": param,
                    },
                )
            )

    rows.sort(key=lambda row: float(row["ccc"]), reverse=True)
    write_csv(args.output_root / "summary.csv", rows)
    write_csv(args.output_root / "session_metrics.csv", session_rows)
    (args.output_root / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(rows[:20], indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
