"""S1 / S2 / S3 shift splits.

Rules (fixed before any shift result existed):
- A window is classified by its target: it belongs to a class only if
  all 12 target steps are in that class. Windows that cross a boundary
  are dropped from both sides.
- S1 (temporal): in-distribution = weekday (Mon-Fri), shifted = weekend.
- S2 (demand): peak = [07:00, 10:00) and [16:00, 19:00) local time
  (10:00:00 itself is off-peak); in-distribution = peak, shifted = off-peak.
- S3 (spatial): k-means, k = 5, seed 2023, on sensor (lat, lon); the
  cluster with the smallest mean distance to its centroid is held out.
- All splits use the same fixed windows as the standard evaluation
  (data_loaders/windows.py). S1/S2 train/val pools come from the
  train/val periods and test pools from the test period.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from data_loaders.windows import (SEQ_OUT, frozen_anchors, load_raw,
                                  sensor_locations)

KMEANS_K = 5
KMEANS_SEED = 2023


# ------------------------------------------------------------- classifiers

def is_weekday(ts) -> np.ndarray:
    """True for Mon-Fri. ts: DatetimeIndex-like."""
    return np.asarray(ts.dayofweek) < 5


def is_weekend(ts) -> np.ndarray:
    return np.asarray(ts.dayofweek) >= 5


def is_peak(ts) -> np.ndarray:
    """True inside [07:00, 10:00) ∪ [16:00, 19:00) local time."""
    h = np.asarray(ts.hour)
    return ((h >= 7) & (h < 10)) | ((h >= 16) & (h < 19))


def is_offpeak(ts) -> np.ndarray:
    return ~is_peak(ts)


# ----------------------------------------------------------- core machinery

def classify_targets(anchors: np.ndarray, step_mask: np.ndarray) -> np.ndarray:
    """All-12-target-steps rule.

    anchors: (B,) last-input-step row indices.
    step_mask: (T,) bool — whether each raw time step is in the class.
    Returns (B,) bool — True iff ALL of rows [a+1, a+12] are in the class.
    """
    anchors = np.asarray(anchors)
    # (B, 12) target row indices
    tgt = anchors[:, None] + 1 + np.arange(SEQ_OUT)[None, :]
    return step_mask[tgt].all(axis=1)


@dataclass
class Split:
    """A temporal shift split over the frozen windows."""
    name: str
    dataset: str
    train_anchors: np.ndarray
    val_anchors: np.ndarray
    test_anchors: np.ndarray
    dropped: dict = field(default_factory=dict)
    spec: str = "decisions.md D8 (2026-06-10)"

    def summary(self) -> dict:
        return {
            "split": self.name,
            "dataset": self.dataset,
            "n_train": int(len(self.train_anchors)),
            "n_val": int(len(self.val_anchors)),
            "n_test": int(len(self.test_anchors)),
            "dropped": {k: int(v) for k, v in self.dropped.items()},
            "spec": self.spec,
        }


def _temporal_split(name: str, dataset: str, train_pred, test_pred) -> Split:
    """Shared S1/S2 builder.

    train/val pools = frozen train/val windows wholly inside train_pred;
    test pool = frozen test windows wholly inside test_pred. Windows in
    neither class (boundary-spanning, or matching the other predicate's
    region) are dropped and counted.
    """
    _, ts, _ = load_raw(dataset)
    pools = frozen_anchors(dataset)
    train_mask, test_mask = train_pred(ts), test_pred(ts)

    keep_train = classify_targets(pools["train"], train_mask)
    keep_val = classify_targets(pools["val"], train_mask)
    keep_test = classify_targets(pools["test"], test_mask)
    return Split(
        name=name, dataset=dataset,
        train_anchors=pools["train"][keep_train],
        val_anchors=pools["val"][keep_val],
        test_anchors=pools["test"][keep_test],
        dropped={
            "train": int((~keep_train).sum()),
            "val": int((~keep_val).sum()),
            "test": int((~keep_test).sum()),
        },
    )


# ------------------------------------------------------------- public API

def s1_split(dataset: str) -> Split:
    """S1 temporal: train Mon-Fri, test Sat-Sun."""
    return _temporal_split("S1_weekday_to_weekend", dataset,
                           is_weekday, is_weekend)


def s2_split(dataset: str) -> Split:
    """S2 demand: train peak hours, test off-peak."""
    return _temporal_split("S2_peak_to_offpeak", dataset,
                           is_peak, is_offpeak)


@dataclass
class SensorSplit:
    """S3 spatial holdout: a partition of sensors, not of time."""
    name: str
    dataset: str
    train_sensors: np.ndarray   # column indices into the data matrix
    test_sensors: np.ndarray
    cluster_sizes: list
    holdout_cluster: int
    holdout_tightness: float    # mean distance to centroid (deg)
    seed: int = KMEANS_SEED
    spec: str = "decisions.md D8 (2026-06-10)"

    def summary(self) -> dict:
        return {
            "split": self.name,
            "dataset": self.dataset,
            "n_train_sensors": int(len(self.train_sensors)),
            "n_test_sensors": int(len(self.test_sensors)),
            "holdout_fraction": round(
                len(self.test_sensors)
                / (len(self.test_sensors) + len(self.train_sensors)), 4),
            "cluster_sizes": [int(c) for c in self.cluster_sizes],
            "holdout_cluster": int(self.holdout_cluster),
            "holdout_tightness_deg": round(float(self.holdout_tightness), 6),
            "kmeans_seed": self.seed,
            "spec": self.spec,
        }


def s3_split_from_coords(coords: np.ndarray, dataset: str = "synthetic",
                         k: int = KMEANS_K,
                         seed: int = KMEANS_SEED) -> SensorSplit:
    """Core S3 logic on explicit coordinates (unit-testable without data)."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(coords)
    labels, centers = km.labels_, km.cluster_centers_
    tightness = np.array([
        np.linalg.norm(coords[labels == c] - centers[c], axis=1).mean()
        for c in range(k)
    ])
    holdout = int(np.argmin(tightness))
    test_idx = np.where(labels == holdout)[0]
    train_idx = np.where(labels != holdout)[0]
    return SensorSplit(
        name="S3_spatial_holdout", dataset=dataset,
        train_sensors=train_idx, test_sensors=test_idx,
        cluster_sizes=np.bincount(labels, minlength=k).tolist(),
        holdout_cluster=holdout,
        holdout_tightness=float(tightness[holdout]),
        seed=seed,
    )


def s3_split(dataset: str, k: int = KMEANS_K,
             seed: int = KMEANS_SEED) -> SensorSplit:
    """S3 spatial holdout on the dataset's real sensor coordinates."""
    return s3_split_from_coords(sensor_locations(dataset), dataset, k, seed)
