"""Losses, reconstruction, and classification metrics for PinSoRo."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn


HEADS = ("task", "social")
CLASS_COUNTS = {"task": 4, "social": 5}


def masked_multitask_cross_entropy(logits: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    for head in HEADS:
        target = batch[f"{head}_y"]
        mask = batch[f"{head}_mask"].bool()
        if torch.any(mask):
            losses.append(nn.functional.cross_entropy(logits[head][mask], target[mask]))
    if not losses:
        raise RuntimeError("Batch contains no supervised PinSoRo labels.")
    return torch.stack(losses).mean()


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float | int]:
    valid = (y_true >= 0) & (y_pred >= 0)
    true = y_true[valid].astype(np.int64, copy=False)
    pred = y_pred[valid].astype(np.int64, copy=False)
    n = int(len(true))
    if n == 0:
        return {"n_frames": 0, "kappa": float("nan"), "macro_f1": float("nan"), "accuracy": float("nan")}

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (true, pred), 1)
    accuracy = float(np.trace(confusion) / n)
    expected = float(confusion.sum(axis=1) @ confusion.sum(axis=0) / (n * n))
    kappa = float((accuracy - expected) / (1.0 - expected)) if expected < 1.0 else float("nan")
    f1_values = []
    for class_id in range(n_classes):
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1_values.append(float(2 * tp / denom) if denom else 0.0)
    return {"n_frames": n, "kappa": kappa, "macro_f1": float(np.mean(f1_values)), "accuracy": accuracy}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(reconstructed: list[dict[str, object]], group_key: str | None = None) -> list[dict[str, object]]:
    groups = ["overall"] if group_key is None else sorted({str(item[group_key]) for item in reconstructed})
    rows: list[dict[str, object]] = []
    for group in groups:
        subset = reconstructed if group_key is None else [item for item in reconstructed if str(item[group_key]) == group]
        for head in HEADS:
            true = np.concatenate([item[f"{head}_y"] for item in subset])
            pred = np.concatenate([item[f"{head}_pred"] for item in subset])
            mask = np.concatenate([item[f"{head}_mask"] for item in subset]).astype(bool)
            metrics = classification_metrics(true[mask], pred[mask], CLASS_COUNTS[head])
            rows.append({group_key or "group": group, "head": head, **metrics})
    return rows


def domain_role_metric_rows(reconstructed: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups = sorted({(str(item["domain"]), str(item["role"])) for item in reconstructed})
    for domain, role in groups:
        subset = [item for item in reconstructed if item["domain"] == domain and item["role"] == role]
        for head in HEADS:
            true = np.concatenate([item[f"{head}_y"] for item in subset])
            pred = np.concatenate([item[f"{head}_pred"] for item in subset])
            mask = np.concatenate([item[f"{head}_mask"] for item in subset]).astype(bool)
            metrics = classification_metrics(true[mask], pred[mask], CLASS_COUNTS[head])
            rows.append({"domain": domain, "role": role, "head": head, **metrics})
    return rows


def write_metric_outputs(run_dir: Path, reconstructed: list[dict[str, object]]) -> dict[str, float]:
    overall = metric_rows(reconstructed)
    write_csv(run_dir / "metrics_overall.csv", overall)
    write_csv(run_dir / "metrics_by_domain.csv", metric_rows(reconstructed, "domain"))
    write_csv(run_dir / "metrics_by_role.csv", metric_rows(reconstructed, "role"))
    write_csv(run_dir / "metrics_by_domain_role.csv", domain_role_metric_rows(reconstructed))

    kappas = [float(row["kappa"]) for row in overall if np.isfinite(float(row["kappa"]))]
    return {"mean_kappa": float(np.mean(kappas)) if kappas else float("nan")}


def write_predictions(path: Path, reconstructed: list[dict[str, object]]) -> None:
    fieldnames = ["domain", "source_split", "session_id", "role", "head", "frame_idx", "y_true", "y_pred"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in reconstructed:
            for head in HEADS:
                y = item[f"{head}_y"]
                pred = item[f"{head}_pred"]
                mask = item[f"{head}_mask"]
                for frame_idx in np.flatnonzero(mask):
                    writer.writerow({"domain": item["domain"], "source_split": item["source_split"], "session_id": item["session_id"], "role": item["role"], "head": head, "frame_idx": int(frame_idx), "y_true": int(y[frame_idx]), "y_pred": int(pred[frame_idx])})


def write_test_predictions(path: Path, reconstructed: list[dict[str, object]]) -> None:
    """Write test predictions, excluding the unsupervised CR yellow role."""

    fieldnames = ["domain", "source_split", "session_id", "role", "head", "frame_idx", "y_pred"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in reconstructed:
            if item["domain"] == "CR" and item["role"] == "yellow":
                continue
            covered = item["covered"]
            for head in HEADS:
                pred = item[f"{head}_pred"]
                for frame_idx in np.flatnonzero(covered):
                    writer.writerow({"domain": item["domain"], "source_split": item["source_split"], "session_id": item["session_id"], "role": item["role"], "head": head, "frame_idx": int(frame_idx), "y_pred": int(pred[frame_idx])})
