from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


EPS = 1e-8


@dataclass(frozen=True)
class RegressionMetrics:
    n_frames: int
    ccc: float
    mae: float
    rmse: float
    pearson: float


def _valid_numpy(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask is not None:
        valid &= np.asarray(mask).reshape(-1) > 0
    return y_true[valid], y_pred[valid]


def concordance_ccc(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Concordance correlation coefficient, the primary engagement metric."""

    yt, yp = _valid_numpy(y_true, y_pred, mask)
    if yt.size < 2:
        return float("nan")
    mean_t = float(np.mean(yt))
    mean_p = float(np.mean(yp))
    var_t = float(np.var(yt))
    var_p = float(np.var(yp))
    cov = float(np.mean((yt - mean_t) * (yp - mean_p)))
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    if denom <= EPS:
        return float("nan")
    return float((2.0 * cov) / denom)


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    yt, yp = _valid_numpy(y_true, y_pred, mask)
    if yt.size < 2:
        return float("nan")
    if float(np.std(yt)) <= EPS or float(np.std(yp)) <= EPS:
        return float("nan")
    return float(np.corrcoef(yt, yp)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> RegressionMetrics:
    """Compute frame-level metrics after applying the target mask."""

    yt, yp = _valid_numpy(y_true, y_pred, mask)
    if yt.size == 0:
        return RegressionMetrics(0, float("nan"), float("nan"), float("nan"), float("nan"))
    err = yp - yt
    return RegressionMetrics(
        n_frames=int(yt.size),
        ccc=concordance_ccc(yt, yp),
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        pearson=pearson_r(yt, yp),
    )


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over valid target frames only."""

    valid = mask > 0
    if not torch.any(valid):
        return torch.tensor(0.0, device=pred.device)
    diff = pred[valid] - target[valid]
    return torch.mean(diff * diff)


def ccc_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Differentiable 1-CCC loss over valid target frames."""

    valid = mask > 0
    if torch.sum(valid) < 2:
        return torch.tensor(0.0, device=pred.device)
    pred_v = pred[valid]
    target_v = target[valid]
    mean_p = torch.mean(pred_v)
    mean_t = torch.mean(target_v)
    var_p = torch.mean((pred_v - mean_p) ** 2)
    var_t = torch.mean((target_v - mean_t) ** 2)
    cov = torch.mean((pred_v - mean_p) * (target_v - mean_t))
    ccc = (2.0 * cov) / (var_p + var_t + (mean_p - mean_t) ** 2 + EPS)
    return 1.0 - ccc

