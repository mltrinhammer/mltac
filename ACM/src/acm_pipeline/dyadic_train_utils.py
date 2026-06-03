from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.acm_pipeline.metrics import RegressionMetrics, regression_metrics


ROLE_ORDER = ("novice", "expert")


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


def grouped_dyadic_metric_outputs(run_dir: Path, reconstructed: list[dict[str, object]]) -> dict[str, float]:
    """Write dyadic metrics overall, by role channel, dataset, and session."""

    # Overall metrics flatten both role channels. Role-specific metrics are
    # written separately so novice/expert performance remains visible.
    all_true = np.concatenate([item["y_true"].reshape(-1) for item in reconstructed])
    all_pred = np.concatenate([item["y_pred"].reshape(-1) for item in reconstructed])
    all_mask = np.concatenate([item["target_mask"].reshape(-1) for item in reconstructed])
    overall = regression_metrics(all_true, all_pred, all_mask)
    write_csv(run_dir / "metrics_overall.csv", ["group", "n_frames", "ccc", "mae", "rmse", "pearson"], [metric_row({"group": "overall"}, overall)])

    role_rows = []
    for channel, role in enumerate(ROLE_ORDER):
        metrics = regression_metrics(
            np.concatenate([item["y_true"][:, channel] for item in reconstructed]),
            np.concatenate([item["y_pred"][:, channel] for item in reconstructed]),
            np.concatenate([item["target_mask"][:, channel] for item in reconstructed]),
        )
        role_rows.append(metric_row({"role": role, "target_channel": channel}, metrics))
    write_csv(run_dir / "metrics_by_role.csv", ["role", "target_channel", "n_frames", "ccc", "mae", "rmse", "pearson"], role_rows)

    dataset_rows = []
    for dataset in sorted({item["example"].dataset for item in reconstructed}):
        subset = [item for item in reconstructed if item["example"].dataset == dataset]
        metrics = regression_metrics(
            np.concatenate([item["y_true"].reshape(-1) for item in subset]),
            np.concatenate([item["y_pred"].reshape(-1) for item in subset]),
            np.concatenate([item["target_mask"].reshape(-1) for item in subset]),
        )
        dataset_rows.append(metric_row({"dataset": dataset}, metrics))
    write_csv(run_dir / "metrics_by_dataset.csv", ["dataset", "n_frames", "ccc", "mae", "rmse", "pearson"], dataset_rows)

    session_rows = []
    for item in reconstructed:
        example = item["example"]
        metrics = regression_metrics(item["y_true"].reshape(-1), item["y_pred"].reshape(-1), item["target_mask"].reshape(-1))
        session_rows.append(metric_row({"dataset": example.dataset, "session_id": example.session_id}, metrics))
    write_csv(run_dir / "metrics_by_session.csv", ["dataset", "session_id", "n_frames", "ccc", "mae", "rmse", "pearson"], session_rows)
    return {"ccc": overall.ccc, "mae": overall.mae, "rmse": overall.rmse, "pearson": overall.pearson}


def write_dyadic_prediction_csv(path: Path, reconstructed: list[dict[str, object]]) -> None:
    """Write frame-level dyadic validation predictions in long format."""

    fieldnames = ["dataset", "session_id", "role", "target_channel", "frame_idx", "y_true", "y_pred", "target_mask", "covered"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in reconstructed:
            example = item["example"]
            y_true = item["y_true"]
            y_pred = item["y_pred"]
            target_mask = item["target_mask"]
            covered = item["covered"]
            assert isinstance(y_true, np.ndarray)
            assert isinstance(y_pred, np.ndarray)
            assert isinstance(target_mask, np.ndarray)
            assert isinstance(covered, np.ndarray)
            for channel, role in enumerate(ROLE_ORDER):
                for frame_idx in range(y_true.shape[0]):
                    writer.writerow(
                        {
                            "dataset": example.dataset,
                            "session_id": example.session_id,
                            "role": role,
                            "target_channel": channel,
                            "frame_idx": frame_idx,
                            "y_true": float(y_true[frame_idx, channel]),
                            "y_pred": float(y_pred[frame_idx, channel]),
                            "target_mask": float(target_mask[frame_idx, channel]),
                            "covered": float(covered[frame_idx]),
                        }
                    )

