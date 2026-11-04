"""Historical average (HA): forecast each step with the training-period mean
of the same sensor at the same minute of the week (Li et al., 2018).

Zero readings are left out of the means. A minute of the week never seen
in training falls back to the sensor's training mean (0 for a sensor that
always reads 0).
"""
from __future__ import annotations

import numpy as np

from src.data import (INPUT_STEPS, TARGET_STEPS, largest_sd_flow, load_series,
                      window_anchors)
from src.metrics import ZERO

MINUTES_PER_WEEK = 7 * 24 * 60


def minute_of_week(timestamps):
    return (timestamps.dayofweek.values * 1440 + timestamps.hour.values * 60
            + timestamps.minute.values).astype(np.int64)


def training_rows(name):
    """Time steps used to fit HA.

    METR-LA / PEMS-BAY: the first n_train + 12 steps, as in the DCRNN
    data layout. LargeST-SD: every step that a training window covers.
    """
    train = window_anchors(name)["train"]
    if name == "largest_sd":
        return np.unique((train[:, None] + np.arange(-INPUT_STEPS + 1, TARGET_STEPS + 1)).ravel())
    return np.arange(len(train) + INPUT_STEPS)


def fit(name):
    """Mean of each (minute of week, sensor) over the training steps."""
    values, timestamps, _ = load_series(name)
    if name == "largest_sd":
        values = largest_sd_flow()  # the values the models were scored on
    key = minute_of_week(timestamps)
    rows = training_rows(name)
    v, k = values[rows], key[rows]
    keep = np.abs(v) > ZERO

    table = np.full((MINUTES_PER_WEEK, v.shape[1]), np.nan)
    for j in range(v.shape[1]):
        sums = np.bincount(k[keep[:, j]], weights=v[keep[:, j], j], minlength=MINUTES_PER_WEEK)
        counts = np.bincount(k[keep[:, j]], minlength=MINUTES_PER_WEEK)
        table[:, j] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    n_kept = keep.sum(axis=0)
    fallback = np.where(n_kept > 0, np.where(keep, v, 0).sum(axis=0) / np.maximum(n_kept, 1), 0.0)
    return table, fallback, key


def forecast(name, part="test"):
    """HA forecasts for the train, val or test windows, (windows, 12, sensors)."""
    table, fallback, key = fit(name)
    anchors = window_anchors(name)[part]
    pred = table[key[anchors[:, None] + 1 + np.arange(TARGET_STEPS)]]
    return np.where(np.isnan(pred), fallback, pred)
