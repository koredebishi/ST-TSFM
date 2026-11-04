"""Shift severity: how far apart the in-distribution and shifted values are.

Three measures between two samples of readings:
  w1     Wasserstein-1 distance (the main measure)
  ks     two-sample Kolmogorov-Smirnov statistic
  dmean  absolute difference of the means

S1 and S2 compare, for each sensor, its readings in the training-period
in-distribution windows with its readings in the test-period shifted
windows, then average over sensors. S3 compares each held-out sensor with
the pooled retained sensors over the test period, then averages over
held-out sensors. Zero readings are dropped first, and a sensor with fewer
than 10 readings left is skipped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from src.data import TARGET_STEPS, load_series, window_anchors
from src.metrics import ZERO
from src.splits import heldout_sensors, in_pool, step_masks

MEASURES = {
    "w1": lambda a, b: float(wasserstein_distance(a, b)),
    "ks": lambda a, b: float(ks_2samp(a, b, method="asymp").statistic),
    "dmean": lambda a, b: float(abs(a.mean() - b.mean())),
}
MIN_READINGS = 10


def target_rows(anchors):
    """Time steps covered by the targets of these windows (each once)."""
    return np.unique((np.asarray(anchors)[:, None] + 1 + np.arange(TARGET_STEPS)).ravel())


def nonzero(values):
    return values[np.abs(values) > ZERO]


def _pool_rows(name, shift):
    """Rows of the training in-distribution pool and the test shifted pool."""
    values, timestamps, _ = load_series(name)
    anchors = window_anchors(name)
    in_dist, shifted = step_masks(timestamps, shift)
    train = anchors["train"][in_pool(anchors["train"], in_dist)]
    test = anchors["test"][in_pool(anchors["test"], shifted)]
    return values, target_rows(train), target_rows(test)


def temporal_severity(name, shift):
    """S1 or S2: per-sensor distance, averaged over sensors."""
    values, rows_a, rows_b = _pool_rows(name, shift)
    per_sensor = {m: [] for m in MEASURES}
    for j in range(values.shape[1]):
        a, b = nonzero(values[rows_a, j]), nonzero(values[rows_b, j])
        if len(a) < MIN_READINGS or len(b) < MIN_READINGS:
            continue  # a sensor that is always 0 has no distribution
        for m, measure in MEASURES.items():
            per_sensor[m].append(measure(a, b))
    return {m: float(np.mean(v)) for m, v in per_sensor.items()}


def spatial_severity(name):
    """S3: each held-out sensor against the pooled retained sensors."""
    values, _, _ = load_series(name)
    rows = target_rows(window_anchors(name)["test"])
    retained, held_out = heldout_sensors(name)
    reference = nonzero(values[np.ix_(rows, retained)].ravel())
    per_sensor = {m: [] for m in MEASURES}
    for j in held_out:
        v = nonzero(values[rows, j])
        if len(v) >= MIN_READINGS:
            for m, measure in MEASURES.items():
                per_sensor[m].append(measure(v, reference))
    return {m: float(np.mean(v)) for m, v in per_sensor.items()}


def severity(name):
    """Severity of S1, S2 and S3 on one dataset, for all three measures."""
    return {"S1": temporal_severity(name, "S1"),
            "S2": temporal_severity(name, "S2"),
            "S3": spatial_severity(name)}


def pooled_w1(name):
    """W1 between all in-distribution and all shifted readings, every sensor
    pooled (the alternative form discussed in Section III-C)."""
    values, _, _ = load_series(name)
    out = {}
    for shift in ("S1", "S2"):
        _, rows_a, rows_b = _pool_rows(name, shift)
        out[shift] = MEASURES["w1"](nonzero(values[rows_a].ravel()), nonzero(values[rows_b].ravel()))
    rows = target_rows(window_anchors(name)["test"])
    retained, held_out = heldout_sensors(name)
    out["S3"] = MEASURES["w1"](nonzero(values[np.ix_(rows, retained)].ravel()),
                               nonzero(values[np.ix_(rows, held_out)].ravel()))
    return out


# ---------------------------------------------------------------- standardized
# Severity on the values as the models see them: each target value is
# standardized by the mean and standard deviation of its window's 512-step
# context. Unlike the raw form, every (window, step) entry counts, and
# windows without a full context or with a flat one (sd <= 1e-3) are left out.
CONTEXT = 512
FLAT = 1e-3


def _context_stats(values):
    """Mean and sd of the CONTEXT steps ending at each row (NaN before that)."""
    roll = pd.DataFrame(values).rolling(window=CONTEXT, min_periods=CONTEXT)
    return roll.mean().to_numpy(), roll.std(ddof=0).to_numpy()


def _standardized(values, mean, sd, anchors, j):
    """Standardized non-zero target values of sensor j over these windows."""
    a = anchors[anchors >= CONTEXT - 1]
    rows = a[:, None] + 1 + np.arange(TARGET_STEPS)
    y, m, s = values[rows, j], mean[a, j][:, None], sd[a, j][:, None]
    keep = (np.abs(y) > ZERO) & (s > FLAT)
    return ((y - m) / np.where(keep, s, 1))[keep]


def standardized_severity(name):
    values, timestamps, _ = load_series(name)
    mean, sd = _context_stats(values)
    anchors = window_anchors(name)
    out = {}
    for shift in ("S1", "S2"):
        in_dist, shifted = step_masks(timestamps, shift)
        train = anchors["train"][in_pool(anchors["train"], in_dist)]
        test = anchors["test"][in_pool(anchors["test"], shifted)]
        per_sensor = {m: [] for m in MEASURES}
        for j in range(values.shape[1]):
            a = _standardized(values, mean, sd, train, j)
            b = _standardized(values, mean, sd, test, j)
            if len(a) >= MIN_READINGS and len(b) >= MIN_READINGS:
                for m, measure in MEASURES.items():
                    per_sensor[m].append(measure(a, b))
        out[shift] = {m: float(np.mean(v)) for m, v in per_sensor.items()}

    retained, held_out = heldout_sensors(name)
    test = anchors["test"]
    reference = np.concatenate([_standardized(values, mean, sd, test, j) for j in retained])
    per_sensor = {m: [] for m in MEASURES}
    for j in held_out:
        v = _standardized(values, mean, sd, test, j)
        if len(v) >= MIN_READINGS:
            for m, measure in MEASURES.items():
                per_sensor[m].append(measure(v, reference))
    out["S3"] = {m: float(np.mean(v)) for m, v in per_sensor.items()}
    return out
