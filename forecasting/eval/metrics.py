"""Forecasting metrics for traffic time series.

Conventions match the DCRNN / Graph WaveNet / LargeST / STAEformer literature:
- MAE, RMSE, MAPE, p95: zero-valued ground truths are masked out by default
  (a stalled / missing speed sensor reads 0; including it inflates error and
  makes results incomparable to every published baseline). Pass mask_zero=False
  to compute on all valid entries instead.
- sMAPE: own zero handling (excludes entries where |y| + |y_hat| == 0).
- R^2: standard coefficient of determination (NaN-masked only).

All functions accept numpy arrays of identical shape and return a Python float.
"""
from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray

# Absolute tolerance used to identify "missing / stalled sensor" zero readings
# after inverse-scaling. The literature checks exact zero, but float32 round-trip
# through (x - mean) / std and back leaves tiny non-zero residuals (~1e-6),
# which would falsely count as valid speeds and inflate MAE.
_ZERO_TOL = 1e-3


def _check(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def _valid_mask(y_true: np.ndarray, mask_zero: bool) -> np.ndarray:
    m = ~np.isnan(y_true)
    if mask_zero:
        m &= np.abs(y_true) > _ZERO_TOL
    return m


def mae(y_true: ArrayLike, y_pred: ArrayLike, mask_zero: bool = True) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    m = _valid_mask(y_true, mask_zero=mask_zero)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def rmse(y_true: ArrayLike, y_pred: ArrayLike, mask_zero: bool = True) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    m = _valid_mask(y_true, mask_zero=mask_zero)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute percentage error, as a fraction (0.05 = 5 %).
    Zero ground-truth entries are masked out (DCRNN convention).
    """
    y_true, y_pred = _check(y_true, y_pred)
    m = _valid_mask(y_true, mask_zero=True)
    if not m.any():
        return float("nan")
    # mean(|y - y_hat| / |y|) — divide by |y| (not y) to avoid sign flips on
    # the rare slightly-negative inverse-transform residuals that survive the
    # _ZERO_TOL filter.
    return float(np.mean(np.abs(y_true[m] - y_pred[m]) / np.abs(y_true[m])))


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Symmetric MAPE as a fraction.
    smape = mean( |y - y_hat| / ((|y| + |y_hat|) / 2) ).
    Range [0, 2]. Entries where (|y| + |y_hat|) == 0 are masked out.
    """
    y_true, y_pred = _check(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    m = ~np.isnan(y_true) & (denom > 0)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m]) / denom[m]))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination. Returns nan if the target is constant."""
    y_true, y_pred = _check(y_true, y_pred)
    m = _valid_mask(y_true, mask_zero=False)
    if not m.any():
        return float("nan")
    yt = y_true[m]
    ss_res = float(np.sum((yt - y_pred[m]) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def p95(y_true: ArrayLike, y_pred: ArrayLike, mask_zero: bool = True) -> float:
    """95th percentile of absolute error. Reports congestion tail behaviour."""
    y_true, y_pred = _check(y_true, y_pred)
    m = _valid_mask(y_true, mask_zero=mask_zero)
    if not m.any():
        return float("nan")
    return float(np.percentile(np.abs(y_true[m] - y_pred[m]), 95))


def all_metrics(y_true: ArrayLike, y_pred: ArrayLike,
                mask_zero: bool = True) -> dict[str, float]:
    """Convenience: returns all six metrics as a dict.

    mask_zero applies to mae/rmse/p95 (mape always masks zeros; smape/r2 have
    their own conventions).
    """
    return {
        "mae": mae(y_true, y_pred, mask_zero=mask_zero),
        "rmse": rmse(y_true, y_pred, mask_zero=mask_zero),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "p95": p95(y_true, y_pred, mask_zero=mask_zero),
    }
