from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.acm_pipeline.data import ManifestExample
from src.acm_pipeline.metrics import RegressionMetrics, regression_metrics


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(prefix: dict[str, object], metrics: RegressionMetrics) -> dict[str, object]:
    return {
        **prefix,
        "n_frames": metrics.n_frames,
        "ccc": metrics.ccc,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "pearson": metrics.pearson,
    }


def grouped_metric_outputs(run_dir: Path, reconstructed: list[dict[str, object]]) -> dict[str, float]:
    """Write overall/dataset/role/session metrics from reconstructed sessions."""

    # Every baseline reconstructs frame-level validation predictions first.
    # This shared metric writer keeps TCN and XGBoost evaluations comparable.
    all_true = np.concatenate([item["y_true"] for item in reconstructed])
    all_pred = np.concatenate([item["y_pred"] for item in reconstructed])
    all_mask = np.concatenate([item["target_mask"] for item in reconstructed])
    overall = regression_metrics(all_true, all_pred, all_mask)
    write_csv(run_dir / "metrics_overall.csv", ["group", "n_frames", "ccc", "mae", "rmse", "pearson"], [metric_row({"group": "overall"}, overall)])

    dataset_rows = []
    role_rows = []
    session_rows = []
    for dataset in sorted({item["example"].dataset for item in reconstructed}):
        subset = [item for item in reconstructed if item["example"].dataset == dataset]
        metrics = regression_metrics(
            np.concatenate([item["y_true"] for item in subset]),
            np.concatenate([item["y_pred"] for item in subset]),
            np.concatenate([item["target_mask"] for item in subset]),
        )
        dataset_rows.append(metric_row({"dataset": dataset}, metrics))
    for role in sorted({item["example"].role for item in reconstructed}):
        subset = [item for item in reconstructed if item["example"].role == role]
        metrics = regression_metrics(
            np.concatenate([item["y_true"] for item in subset]),
            np.concatenate([item["y_pred"] for item in subset]),
            np.concatenate([item["target_mask"] for item in subset]),
        )
        role_rows.append(metric_row({"role": role}, metrics))
    for item in reconstructed:
        example = item["example"]
        metrics = regression_metrics(item["y_true"], item["y_pred"], item["target_mask"])
        session_rows.append(metric_row({"dataset": example.dataset, "session_id": example.session_id, "role": example.role}, metrics))

    metric_fields = ["n_frames", "ccc", "mae", "rmse", "pearson"]
    write_csv(run_dir / "metrics_by_dataset.csv", ["dataset", *metric_fields], dataset_rows)
    write_csv(run_dir / "metrics_by_role.csv", ["role", *metric_fields], role_rows)
    write_csv(run_dir / "metrics_by_session.csv", ["dataset", "session_id", "role", *metric_fields], session_rows)
    return {"ccc": overall.ccc, "mae": overall.mae, "rmse": overall.rmse, "pearson": overall.pearson}


def write_prediction_csv(path: Path, reconstructed: list[dict[str, object]]) -> None:
    """Write frame-level validation predictions for later analysis plots."""

    fieldnames = ["dataset", "session_id", "role", "frame_idx", "y_true", "y_pred", "target_mask", "covered"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in reconstructed:
            example = item["example"]
            assert isinstance(example, ManifestExample)
            y_true = item["y_true"]
            y_pred = item["y_pred"]
            target_mask = item["target_mask"]
            covered = item["covered"]
            assert isinstance(y_true, np.ndarray)
            assert isinstance(y_pred, np.ndarray)
            assert isinstance(target_mask, np.ndarray)
            assert isinstance(covered, np.ndarray)
            for frame_idx in range(len(y_true)):
                writer.writerow(
                    {
                        "dataset": example.dataset,
                        "session_id": example.session_id,
                        "role": example.role,
                        "frame_idx": frame_idx,
                        "y_true": float(y_true[frame_idx]),
                        "y_pred": float(y_pred[frame_idx]),
                        "target_mask": float(target_mask[frame_idx]),
                        "covered": float(covered[frame_idx]),
                    }
                )

