"""
Evaluation metrics for CoP prediction.

All functions accept numpy arrays of shape (N, 2) for predictions and targets.
CoP values are in CENTIMETERS.  Threshold mapping:
  10 mm  →  1.0 cm
  20 mm  →  2.0 cm
  30 mm  →  3.0 cm
"""

from typing import Dict

import numpy as np


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Numerically safe Pearson correlation coefficient.
    Returns 0.0 if either variable has (near-)zero variance.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_cop_metrics(
    preds:   np.ndarray,   # (N, 2)  predicted [CoPX, CoPY]
    targets: np.ndarray,   # (N, 2)  ground-truth [CoPX, CoPY]
) -> Dict[str, float]:
    """
    Returns a dict with:
        rmse_x, rmse_y, rmse_total   — root mean squared error
        mae_x,  mae_y,  mae_total    — mean absolute error
        r2_x, r2_y                   — coefficient of determination
        corr_x, corr_y, corr_total   — Pearson correlation (per-axis and radial)
        within_10mm, within_20mm, within_30mm  — % of samples within threshold
    """
    assert preds.shape == targets.shape and preds.ndim == 2 and preds.shape[1] == 2

    diff   = preds - targets
    errors = np.linalg.norm(diff, axis=1)   # (N,) Euclidean distance

    rmse_x = float(np.sqrt(np.mean(diff[:, 0] ** 2)))
    rmse_y = float(np.sqrt(np.mean(diff[:, 1] ** 2)))
    rmse_t = float(np.sqrt(np.mean(errors ** 2)))

    mae_x = float(np.mean(np.abs(diff[:, 0])))
    mae_y = float(np.mean(np.abs(diff[:, 1])))
    mae_t = float(np.mean(errors))

    r2_x = _safe_r2(targets[:, 0], preds[:, 0])
    r2_y = _safe_r2(targets[:, 1], preds[:, 1])

    # Pearson correlations (per-axis and radial distance)
    corr_x = _safe_corr(targets[:, 0], preds[:, 0])
    corr_y = _safe_corr(targets[:, 1], preds[:, 1])
    # radial correlation between ||gt|| and ||pred||
    corr_total = _safe_corr(
        np.linalg.norm(targets, axis=1),
        np.linalg.norm(preds, axis=1),
    )

    return {
        "rmse_x":      rmse_x,
        "rmse_y":      rmse_y,
        "rmse_total":  rmse_t,
        "mae_x":       mae_x,
        "mae_y":       mae_y,
        "mae_total":   mae_t,
        "r2_x":        r2_x,
        "r2_y":        r2_y,
        "corr_x":      corr_x,
        "corr_y":      corr_y,
        "corr_total":  corr_total,
        "within_10mm": float((errors < 1.0).mean() * 100),   # 10mm = 1 cm
        "within_20mm": float((errors < 2.0).mean() * 100),   # 20mm = 2 cm
        "within_30mm": float((errors < 3.0).mean() * 100),   # 30mm = 3 cm
    }


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    p = f"{prefix} " if prefix else ""
    print(
        f"{p}RMSE (X/Y/Total): {metrics['rmse_x']:.3f} / "
        f"{metrics['rmse_y']:.3f} / {metrics['rmse_total']:.3f}  cm"
    )
    print(
        f"{p}MAE  (X/Y/Total): {metrics['mae_x']:.3f} / "
        f"{metrics['mae_y']:.3f} / {metrics['mae_total']:.3f}  cm"
    )
    print(
        f"{p}R2   (X/Y):       {metrics['r2_x']:.3f} / {metrics['r2_y']:.3f}"
    )
    print(
        f"{p}Corr (X/Y/Total): {metrics['corr_x']:.3f} / "
        f"{metrics['corr_y']:.3f} / {metrics['corr_total']:.3f}"
    )
    print(
        f"{p}Within (10/20/30 mm): "
        f"{metrics['within_10mm']:.1f}% / "
        f"{metrics['within_20mm']:.1f}% / "
        f"{metrics['within_30mm']:.1f}%"
    )
