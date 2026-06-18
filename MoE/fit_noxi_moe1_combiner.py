"""Fit/evaluate NOXI MoE combiner ablations from frozen expert predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Callable, Hashable
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "MoE" / "experiments"
FEATURES = ("visual_videomae", "audio_w2vbert2", "text_xlm_roberta")
DEFAULT_MODES = ("best_single", "uniform", "shared", "role", "val_shared_upper", "val_role_upper")
Key = tuple[str, str, str, int, int]
GroupFn = Callable[[Key], Hashable]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NOXI regression combiner ablations.")
    parser.add_argument("--corpus", default="noxi")
    parser.add_argument("--expert-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=DEFAULT_MODES)
    args = parser.parse_args()
    if args.expert_root is None:
        args.expert_root = EXPERIMENT_ROOT / f"noxi_moe1_{args.corpus}_experts"
    if args.output_root is None:
        args.output_root = EXPERIMENT_ROOT / f"noxi_moe1_{args.corpus}_combiners"
    return args


def prediction_path(root: Path, corpus: str, feature: str, split: str) -> Path:
    run = root / f"{corpus}_{feature}_dyadic_tcn_k11_seed13"
    if split == "train":
        return run / "diagnostics" / "train_internal" / "val_predictions.csv"
    if split == "val":
        return run / "val_predictions.csv"
    raise ValueError(split)


def read_predictions(path: Path) -> dict[Key, dict[str, float]]:
    rows: dict[Key, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["dataset"],
                row["session_id"],
                row["role"],
                int(row["target_channel"]),
                int(row["frame_idx"]),
            )
            rows[key] = {
                "y_true": float(row["y_true"]),
                "y_pred": float(row["y_pred"]),
                "target_mask": float(row["target_mask"]),
                "covered": float(row["covered"]),
            }
    return rows


def aligned_split(
    root: Path,
    corpus: str,
    split: str,
) -> tuple[list[Key], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    by_feature = {
        feature: read_predictions(prediction_path(root, corpus, feature, split))
        for feature in FEATURES
    }
    keys = sorted(set.intersection(*(set(rows) for rows in by_feature.values())))
    if not keys:
        raise RuntimeError(f"No aligned prediction rows for split {split}.")
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
    mask &= finite
    return keys, labels, mask, predictions


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


def combine(predictions: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    output = np.zeros_like(predictions[FEATURES[0]], dtype=np.float64)
    for weight, feature in zip(weights, FEATURES):
        output += float(weight) * predictions[feature]
    return output


def valid_mask(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return mask & np.isfinite(y_true) & np.isfinite(y_pred)


def mse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    valid = valid_mask(y_true, y_pred, mask)
    if not np.any(valid):
        return float("nan")
    diff = y_pred[valid] - y_true[valid]
    return float(np.mean(diff * diff))


def ccc(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    valid = valid_mask(y_true, y_pred, mask)
    if not np.any(valid):
        return float("nan")
    yt = y_true[valid]
    yp = y_pred[valid]
    mean_t = float(np.mean(yt))
    mean_p = float(np.mean(yp))
    var_t = float(np.var(yt))
    var_p = float(np.var(yp))
    cov = float(np.mean((yt - mean_t) * (yp - mean_p)))
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    return float((2.0 * cov) / denom) if denom > 0.0 else float("nan")


def pearson(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    valid = valid_mask(y_true, y_pred, mask)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    yt = y_true[valid]
    yp = y_pred[valid]
    std_t = float(np.std(yt))
    std_p = float(np.std(yp))
    if std_t <= 0.0 or std_p <= 0.0:
        return float("nan")
    return float(np.mean((yt - np.mean(yt)) * (yp - np.mean(yp))) / (std_t * std_p))


def metric_row(prefix: dict[str, object], y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    valid = valid_mask(y_true, y_pred, mask)
    diff = y_pred[valid] - y_true[valid] if np.any(valid) else np.asarray([], dtype=np.float64)
    return {
        **prefix,
        "n_frames": int(np.count_nonzero(valid)),
        "ccc": ccc(y_true, y_pred, mask),
        "mae": float(np.mean(np.abs(diff))) if diff.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else float("nan"),
        "pearson": pearson(y_true, y_pred, mask),
    }


def evaluate(keys: list[Key], labels: np.ndarray, mask: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    row = metric_row({"group": "overall"}, labels, pred, mask)
    for role in ("novice", "expert"):
        role_mask = mask & np.asarray([key[2] == role for key in keys], dtype=bool)
        role_row = metric_row({"group": role}, labels, pred, role_mask)
        for field in ("n_frames", "ccc", "mae", "rmse", "pearson"):
            row[f"{role}_{field}"] = role_row[field]
    return row


def indices_for(keys: list[Key], predicate: Callable[[Key], bool]) -> np.ndarray:
    return np.asarray([idx for idx, key in enumerate(keys) if predicate(key)], dtype=np.int64)


def fit_group_weights(
    keys: list[Key],
    labels: np.ndarray,
    mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    weights_grid: list[np.ndarray],
    group_fn: GroupFn,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    weights_by_group: dict[str, np.ndarray] = {}
    loss_by_group: dict[str, float] = {}
    for group in sorted({group_fn(key) for key in keys}, key=str):
        idx = indices_for(keys, lambda key, group=group: group_fn(key) == group)
        group_predictions = {feature: values[idx] for feature, values in predictions.items()}
        best_weight = weights_grid[0]
        best_loss = float("inf")
        for weights in weights_grid:
            loss = mse(labels[idx], combine(group_predictions, weights), mask[idx])
            if np.isfinite(loss) and loss < best_loss:
                best_loss = loss
                best_weight = weights
        weights_by_group[str(group)] = best_weight
        loss_by_group[str(group)] = best_loss
    return weights_by_group, loss_by_group


def predict_grouped(keys: list[Key], predictions: dict[str, np.ndarray], weights_by_group: dict[str, np.ndarray], group_fn: GroupFn) -> np.ndarray:
    pred = np.empty(len(keys), dtype=np.float64)
    for group in sorted({group_fn(key) for key in keys}, key=str):
        idx = indices_for(keys, lambda key, group=group: group_fn(key) == group)
        group_predictions = {feature: values[idx] for feature, values in predictions.items()}
        pred[idx] = combine(group_predictions, weights_by_group[str(group)])
    return pred


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def weights_for_json(weights_by_mode: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, list[float]]]:
    return {
        mode: {group: [float(value) for value in weights] for group, weights in sorted(groups.items())}
        for mode, groups in sorted(weights_by_mode.items())
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    weights_grid = simplex_weights(args.step)

    train_keys, train_labels, train_mask, train_predictions = aligned_split(args.expert_root, args.corpus, "train")
    val_keys, val_labels, val_mask, val_predictions = aligned_split(args.expert_root, args.corpus, "val")

    rows: list[dict[str, object]] = []
    weights_by_mode: dict[str, dict[str, np.ndarray]] = {}
    losses_by_mode: dict[str, dict[str, float]] = {}

    if "best_single" in args.modes:
        single_rows = []
        for feature in FEATURES:
            single_rows.append(
                {
                    "mode": f"single_{feature}",
                    "fit_split": "none",
                    "optimistic": "no",
                    **evaluate(val_keys, val_labels, val_mask, val_predictions[feature]),
                }
            )
        rows.extend(single_rows)
        best = max(single_rows, key=lambda row: float(row["ccc"]))
        rows.append({"mode": "best_single", **{key: value for key, value in best.items() if key != "mode"}})

    if "uniform" in args.modes:
        weights = np.asarray([1.0 / len(FEATURES)] * len(FEATURES), dtype=np.float64)
        weights_by_mode["uniform"] = {"all": weights}
        rows.append({"mode": "uniform", "fit_split": "fixed", "optimistic": "no", **evaluate(val_keys, val_labels, val_mask, combine(val_predictions, weights))})

    if "shared" in args.modes:
        weights, losses = fit_group_weights(train_keys, train_labels, train_mask, train_predictions, weights_grid, lambda _key: "all")
        weights_by_mode["shared"] = weights
        losses_by_mode["shared"] = losses
        rows.append({"mode": "shared", "fit_split": "train_internal", "optimistic": "no", **evaluate(val_keys, val_labels, val_mask, predict_grouped(val_keys, val_predictions, weights, lambda _key: "all"))})

    if "role" in args.modes:
        weights, losses = fit_group_weights(train_keys, train_labels, train_mask, train_predictions, weights_grid, lambda key: key[2])
        weights_by_mode["role"] = weights
        losses_by_mode["role"] = losses
        rows.append({"mode": "role", "fit_split": "train_internal", "optimistic": "no", **evaluate(val_keys, val_labels, val_mask, predict_grouped(val_keys, val_predictions, weights, lambda key: key[2]))})

    if "val_shared_upper" in args.modes:
        weights, losses = fit_group_weights(val_keys, val_labels, val_mask, val_predictions, weights_grid, lambda _key: "all")
        weights_by_mode["val_shared_upper"] = weights
        losses_by_mode["val_shared_upper"] = losses
        rows.append({"mode": "val_shared_upper", "fit_split": "val_internal", "optimistic": "yes", **evaluate(val_keys, val_labels, val_mask, predict_grouped(val_keys, val_predictions, weights, lambda _key: "all"))})

    if "val_role_upper" in args.modes:
        weights, losses = fit_group_weights(val_keys, val_labels, val_mask, val_predictions, weights_grid, lambda key: key[2])
        weights_by_mode["val_role_upper"] = weights
        losses_by_mode["val_role_upper"] = losses
        rows.append({"mode": "val_role_upper", "fit_split": "val_internal", "optimistic": "yes", **evaluate(val_keys, val_labels, val_mask, predict_grouped(val_keys, val_predictions, weights, lambda key: key[2]))})

    write_csv(args.output_root / "summary.csv", rows)
    (args.output_root / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")
    (args.output_root / "weights.json").write_text(
        json.dumps(
            {
                "features": list(FEATURES),
                "step": args.step,
                "weights": weights_for_json(weights_by_mode),
                "fit_mse": losses_by_mode,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output_root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
