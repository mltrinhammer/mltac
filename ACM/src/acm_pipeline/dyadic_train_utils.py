from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.acm_pipeline.metrics import RegressionMetrics, regression_metrics


ROLE_ORDER = ("novice", "expert")

DATASET_SUBMISSION_DIRS = {
    ("noxi", "test_internal"): "noxi-base",
    ("noxi", "test_additional"): "noxi-additional",
    ("noxij", "test_internal"): "noxi-j",
    ("noxij", "test"): "noxi-j",
    ("mpiigroupinteraction", "test_internal"): "mpiigroupinteraction",
    ("mpiigroupinteraction", "test"): "mpiigroupinteraction",
}


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


def submission_dataset_dir(dataset: str, model_split: str | None) -> str:
    """Map internal dataset/split names onto organizer submission folders."""

    split = model_split or ""
    mapped = DATASET_SUBMISSION_DIRS.get((dataset, split))
    if mapped is not None:
        return mapped
    if dataset == "noxi":
        if split:
            return f"noxi-{split}"
        return "noxi"
    if dataset == "noxij":
        return "noxi-j"
    return dataset


def prediction_filename_for_role(role_name: str) -> str:
    """Return the organizer submission filename for one role/channel."""

    return f"{role_name}.engagement.prediction.csv"


def write_organizer_submission_tree(output_dir: Path, reconstructed: list[dict[str, object]]) -> None:
    """Write session-wise organizer-format prediction CSVs.

    The exported tree matches the ACM MultiMediate submission layout for test
    splits and falls back to split-qualified dataset folder names for internal
    train/validation runs so the same exporter remains usable during local
    validation.
    """

    for item in reconstructed:
        example = item["example"]
        y_pred = item["y_pred"]
        covered = item["covered"]
        assert isinstance(y_pred, np.ndarray)
        assert isinstance(covered, np.ndarray)

        role_names = tuple(getattr(example, "role_names", ROLE_ORDER))
        if y_pred.ndim != 2:
            raise ValueError(f"Expected y_pred to have shape [time, channels], got {y_pred.shape}")
        if len(role_names) != y_pred.shape[1]:
            raise ValueError(
                f"Role name count does not match prediction channels for {example.dataset}/{example.session_id}: "
                f"{len(role_names)} vs {y_pred.shape[1]}"
            )

        dataset_dir = submission_dataset_dir(example.dataset, getattr(example, "model_split", None))
        session_dir = output_dir / dataset_dir / str(example.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        for channel, role_name in enumerate(role_names):
            # Build the per-frame prediction series for this channel,
            # forward-filling then backward-filling gaps so the evaluator's
            # minimum-coverage threshold is met.
            series = y_pred[:, channel].copy()
            # Mark uncovered or NaN frames as needing fill.
            needs_fill = (~covered.astype(bool)) | np.isnan(series)
            if needs_fill.any() and not needs_fill.all():
                # Forward-fill.
                for i in range(1, len(series)):
                    if needs_fill[i] and not needs_fill[i - 1]:
                        series[i] = series[i - 1]
                        needs_fill[i] = False
                # Backward-fill remaining leading gaps.
                still_bad = np.isnan(series) | needs_fill
                if still_bad.any():
                    first_valid = np.where(~still_bad)[0]
                    if len(first_valid) > 0:
                        series[:first_valid[0]] = series[first_valid[0]]
                        needs_fill[:first_valid[0]] = False

            file_path = session_dir / prediction_filename_for_role(role_name)
            with file_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                for frame_idx in range(len(series)):
                    pred_val = series[frame_idx]
                    if np.isnan(pred_val):
                        pred_text = ""
                    else:
                        pred_text = f"{float(pred_val):.10f}"
                    writer.writerow([pred_text])

