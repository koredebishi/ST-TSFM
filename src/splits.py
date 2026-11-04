"""The three shifts.

S1  weekday -> weekend
S2  peak (07:00-10:00 and 16:00-19:00) -> off-peak
S3  retained sensors -> a held-out cluster of sensors

For S1 and S2 a window belongs to a pool only if all 12 of its target
steps fall in that pool. Windows that cross a boundary are left out of
both pools. S3 splits sensors, not windows: k-means (k = 5, seed 2023) on
sensor coordinates, holding out the tightest cluster.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from src.data import TARGET_STEPS, load_series, sensor_coords, window_anchors

SHIFTS = ("S1", "S2", "S3")


def is_weekend(timestamps):
    return np.asarray(timestamps.dayofweek) >= 5


def is_peak(timestamps):
    hour = np.asarray(timestamps.hour)
    return ((hour >= 7) & (hour < 10)) | ((hour >= 16) & (hour < 19))


def in_pool(anchors, step_in_pool):
    """True for windows whose 12 target steps are all in the pool."""
    rows = np.asarray(anchors)[:, None] + 1 + np.arange(TARGET_STEPS)
    return step_in_pool[rows].all(axis=1)


def step_masks(timestamps, shift):
    """Per time step: (in-distribution, shifted) masks for S1 or S2."""
    if shift == "S1":
        shifted = is_weekend(timestamps)
    elif shift == "S2":
        shifted = ~is_peak(timestamps)
    else:
        raise ValueError(shift)
    return ~shifted, shifted


def test_pools(name, shift, n=None):
    """Boolean masks over the test windows: (in-distribution, shifted).

    n: use only the first n test windows (the common prefix when some
    models have fewer windows).
    """
    _, timestamps, _ = load_series(name)
    anchors = window_anchors(name)["test"][:n]
    in_dist, shifted = step_masks(timestamps, shift)
    return in_pool(anchors, in_dist), in_pool(anchors, shifted)


def heldout_sensors(name, k=5, seed=2023):
    """Column indices of the retained and the held-out sensors (S3)."""
    coords = sensor_coords(name)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(coords)
    spread = [np.linalg.norm(coords[km.labels_ == c] - km.cluster_centers_[c], axis=1).mean()
              for c in range(k)]
    held_out = int(np.argmin(spread))
    return np.where(km.labels_ != held_out)[0], np.where(km.labels_ == held_out)[0]


def weekend_day_pools(name):
    """Test-window masks: weekday, Saturday and Sunday (all 12 target steps
    on that day). Windows that run from Saturday into Sunday are in
    neither day pool."""
    _, timestamps, _ = load_series(name)
    anchors = window_anchors(name)["test"]
    day = np.asarray(timestamps.dayofweek)
    return {"weekday": in_pool(anchors, day < 5),
            "saturday": in_pool(anchors, day == 5),
            "sunday": in_pool(anchors, day == 6)}
