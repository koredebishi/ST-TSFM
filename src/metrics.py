"""Forecast error metrics.

A reading of 0 means a stalled or missing sensor, so MAE, RMSE, MAPE and
the 95th-percentile error skip true values with |y| <= 1e-3 (scaling and
unscaling leaves tiny non-zero values where the raw reading was 0). This
is the convention of DCRNN, Graph WaveNet, LargeST and STAEformer.
sMAPE skips only entries where |y| + |y_hat| is 0. R^2 uses every entry.
"""
from __future__ import annotations

import numpy as np

ZERO = 1e-3


def _pair(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def valid(y_true):
    """Entries that count: a finite, non-zero true value."""
    return np.isfinite(y_true) & (np.abs(y_true) > ZERO)


def mae(y_true, y_pred):
    y_true, y_pred = _pair(y_true, y_pred)
    m = valid(y_true)
    return float(np.mean(np.abs(y_true[m] - y_pred[m]))) if m.any() else float("nan")


def rmse(y_true, y_pred):
    y_true, y_pred = _pair(y_true, y_pred)
    m = valid(y_true)
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2))) if m.any() else float("nan")


def mape(y_true, y_pred):
    """As a fraction (0.05 = 5%)."""
    y_true, y_pred = _pair(y_true, y_pred)
    m = valid(y_true)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m]) / np.abs(y_true[m])))


def smape(y_true, y_pred):
    """mean(|y - y_hat| / ((|y| + |y_hat|) / 2)), as a fraction."""
    y_true, y_pred = _pair(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    m = np.isfinite(y_true) & (denom > 0)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m]) / denom[m]))


def r2(y_true, y_pred):
    y_true, y_pred = _pair(y_true, y_pred)
    m = np.isfinite(y_true)
    yt = y_true[m]
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if not m.any() or ss_tot == 0:
        return float("nan")
    return float(1 - np.sum((yt - y_pred[m]) ** 2) / ss_tot)


def p95(y_true, y_pred):
    """95th percentile of the absolute error."""
    y_true, y_pred = _pair(y_true, y_pred)
    m = valid(y_true)
    return float(np.percentile(np.abs(y_true[m] - y_pred[m]), 95)) if m.any() else float("nan")


def per_sensor_mae(y_true, y_pred):
    """MAE of each sensor over all windows and steps; NaN for a sensor with
    no valid readings (LargeST-SD has a few sensors that always read 0)."""
    m = valid(y_true)
    err = np.where(m, np.abs(y_true - y_pred), 0.0).sum(axis=(0, 1), dtype=np.float64)
    count = m.sum(axis=(0, 1))
    return np.where(count > 0, err / np.maximum(count, 1), np.nan)


def smape_nonzero(y_true, y_pred):
    """sMAPE on the same entries MAE scores (non-zero true values)."""
    y_true, y_pred = _pair(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    m = valid(y_true) & (denom > 0)
    return float(np.mean(np.abs(y_true[m] - y_pred[m]) / denom[m]))


def mase(y_true, y_pred, scale):
    """Mean of |error| / scale, where scale (one value per sensor, last
    axis) is that sensor's one-step naive MAE on the training period."""
    y_true, y_pred = _pair(y_true, y_pred)
    s = np.broadcast_to(scale, y_true.shape)
    m = valid(y_true) & np.isfinite(s) & (s > 0)
    return float(np.mean(np.abs(y_true[m] - y_pred[m]) / s[m]))
