"""Shift-severity scalar — Wasserstein-1 between observable marginals (§5.1, D3).

For temporal shifts (S1, S2): per-sensor W1 between the observable's values
in the split's TRAIN-pool target rows and TEST-pool target rows, averaged
across sensors ("averaged across sensors", plan §5.1).

For the spatial shift (S3): W1 between the pooled marginal of train-region
sensors and the pooled marginal of held-out sensors, over the SAME time
rows (the frozen test region) — plan §5.1 "between train-region and
held-out-region marginal observable distributions".

Conventions:
- The observable is speed (mph) on LA/BAY and flow (veh/15min) on SD;
  severities are therefore comparable across shifts WITHIN a dataset,
  never across the unit boundary (D3).
- Near-zero readings (|v| <= 1e-3) are dropped before computing W1 — the
  same stalled-sensor convention as eval/metrics.py (§10a), so the
  severity measures distribution change, not sensor outages.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from data_loaders.windows import SEQ_OUT, frozen_anchors, load_raw
from shift.splits import SensorSplit, Split

_ZERO_TOL = 1e-3  # matches eval/metrics.py


def _target_rows(anchors: np.ndarray) -> np.ndarray:
    """Unique raw-row indices covered by the anchors' target windows."""
    anchors = np.asarray(anchors)
    rows = (anchors[:, None] + 1 + np.arange(SEQ_OUT)[None, :]).ravel()
    return np.unique(rows)


def _clean(v: np.ndarray) -> np.ndarray:
    return v[np.abs(v) > _ZERO_TOL]


def w1(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-1 between two empirical 1-d distributions."""
    return float(wasserstein_distance(a, b))


def severity_temporal(dataset: str, split: Split) -> dict:
    """Per-sensor W1 between train-pool and test-pool marginals, averaged."""
    raw, _, _ = load_raw(dataset)
    rows_a = _target_rows(split.train_anchors)
    rows_b = _target_rows(split.test_anchors)
    per_sensor = []
    skipped = 0
    for j in range(raw.shape[1]):
        a, b = _clean(raw[rows_a, j]), _clean(raw[rows_b, j])
        if len(a) < 10 or len(b) < 10:  # all-stalled sensor: no distribution
            skipped += 1
            continue
        per_sensor.append(w1(a, b))
    per_sensor = np.array(per_sensor)
    return {
        "split": split.name,
        "dataset": dataset,
        "severity_w1_mean": float(per_sensor.mean()),
        "severity_w1_median": float(np.median(per_sensor)),
        "severity_w1_q25": float(np.percentile(per_sensor, 25)),
        "severity_w1_q75": float(np.percentile(per_sensor, 75)),
        "n_sensors_used": int(len(per_sensor)),
        "n_sensors_skipped": int(skipped),
        "spec": "paper_plan §5.1 + decisions.md D3",
    }


def severity_spatial(dataset: str, split: SensorSplit) -> dict:
    """Pooled W1 between train-region and holdout-region marginals."""
    raw, _, _ = load_raw(dataset)
    rows = _target_rows(frozen_anchors(dataset)["test"])
    a = _clean(raw[np.ix_(rows, split.train_sensors)].ravel())
    b = _clean(raw[np.ix_(rows, split.test_sensors)].ravel())
    return {
        "split": split.name,
        "dataset": dataset,
        "severity_w1_pooled": w1(a, b),
        "n_values_train_region": int(len(a)),
        "n_values_holdout": int(len(b)),
        "spec": "paper_plan §5.1 + decisions.md D3",
    }
