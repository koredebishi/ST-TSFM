"""Do the shifts reorder the model leaderboard? (Table IV)

Models are ranked by their mean per-sensor MAE, once on the
in-distribution pool and once on the shifted pool, and the two rankings
are compared with Kendall's tau. Intervals come from bootstraps:

- sensors (S1, S2): resample sensors with replacement, 1000 times;
- sensors (S3): the two sensor sets are disjoint, so each is resampled
  on its own;
- days (LargeST-SD S1): resample whole days, 2000 times, because
  windows on the same day are not independent.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

from src.data import load_series, window_anchors
from src.metrics import per_sensor_mae, valid
from src.splits import heldout_sensors, test_pools

SEED = 2023


def ranks(per_sensor):
    """(models, sensors) MAE -> rank of each model (0 = best). Dead
    sensors (NaN) are ignored; ties keep the model order."""
    return np.argsort(np.argsort(np.nanmean(per_sensor, axis=1)))


def tau(rank_a, rank_b):
    return float(kendalltau(rank_a, rank_b).statistic)


def _result(models, a, b, taus):
    return {"tau": tau(ranks(a), ranks(b)),
            "interval": [float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5))],
            "ranked_in_dist": [models[i] for i in np.argsort(np.nanmean(a, axis=1))],
            "ranked_shifted": [models[i] for i in np.argsort(np.nanmean(b, axis=1))]}


def window_shift_tau(models, a, b, n_boot=1000, seed=SEED):
    """S1 / S2: a and b are (models, sensors) on the same sensors."""
    rng = np.random.default_rng(seed)
    taus = []
    for _ in range(n_boot):
        idx = rng.integers(0, a.shape[1], size=a.shape[1])
        taus.append(tau(ranks(a[:, idx]), ranks(b[:, idx])))
    return _result(models, a, b, taus)


def sensor_shift_tau(models, a, b, n_boot=1000, seed=SEED):
    """S3: a is (models, retained sensors), b is (models, held-out sensors)."""
    rng = np.random.default_rng(seed)
    taus = []
    for _ in range(n_boot):
        ia = rng.integers(0, a.shape[1], size=a.shape[1])
        ib = rng.integers(0, b.shape[1], size=b.shape[1])
        taus.append(tau(ranks(a[:, ia]), ranks(b[:, ib])))
    return _result(models, a, b, taus)


def ranking_stability(name, y_true, preds):
    """Tau and interval for S1, S2 and S3. preds: model -> forecasts."""
    models = list(preds)
    out = {}
    for shift in ("S1", "S2"):
        in_dist, shifted = test_pools(name, shift, n=len(y_true))
        a = np.stack([per_sensor_mae(y_true[in_dist], preds[m][in_dist]) for m in models])
        b = np.stack([per_sensor_mae(y_true[shifted], preds[m][shifted]) for m in models])
        out[shift] = window_shift_tau(models, a, b)
    retained, held_out = heldout_sensors(name)
    full = np.stack([per_sensor_mae(y_true, preds[m]) for m in models])
    out["S3"] = sensor_shift_tau(models, full[:, retained], full[:, held_out])
    return out


def day_block_bootstrap(name, y_true, preds, n_boot=2000, seed=SEED):
    """Intervals for LargeST-SD S1 from resampling whole days.

    Returns 95% intervals of the frozen models' degradation and of tau.
    A window's day is the date of its first target step.
    """
    models = list(preds)
    _, timestamps, _ = load_series(name)
    anchors = window_anchors(name)["test"][:len(y_true)]
    day = np.asarray(timestamps)[anchors + 1].astype("datetime64[D]")
    in_dist, shifted = test_pools(name, "S1", n=len(y_true))

    def per_day(mask):
        """Per day: summed absolute error of each model and number of valid entries."""
        idx = np.flatnonzero(mask)
        days = np.unique(day[idx])
        err = np.zeros((len(models), len(days)))
        count = np.zeros(len(days))
        for k, d in enumerate(days):
            rows = idx[day[idx] == d]
            ok = valid(y_true[rows])
            count[k] = ok.sum()
            for i, m in enumerate(models):
                err[i, k] = np.abs(preds[m][rows] - y_true[rows])[ok].sum()
        return err, count

    err_a, n_a = per_day(in_dist)
    err_b, n_b = per_day(shifted)
    rng = np.random.default_rng(seed)
    deg = {m: [] for m in ("chronos2", "timesfm")}
    taus = []
    for _ in range(n_boot):
        da = rng.integers(0, len(n_a), len(n_a))
        db = rng.integers(0, len(n_b), len(n_b))
        mae_a = err_a[:, da].sum(axis=1) / n_a[da].sum()
        mae_b = err_b[:, db].sum(axis=1) / n_b[db].sum()
        for m in deg:
            i = models.index(m)
            deg[m].append(mae_b[i] / mae_a[i] - 1)
        taus.append(float(kendalltau(mae_a, mae_b).statistic))
    interval = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"days": [len(n_a), len(n_b)],
            "degradation": {m: interval(v) for m, v in deg.items()},
            "tau": interval(taus)}
