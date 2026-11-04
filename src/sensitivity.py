"""Do the results depend on how the splits are drawn? (Section V)

S2 is redone with three other peak windows (wider, narrower, one hour
later), and S3 with k = 4, 6 and 8 clusters. S1 has no such choice. For
each variant we report Kendall tau of the full ranking and the frozen
models' degradation, here as the ratio of mean per-sensor MAE.
"""
from __future__ import annotations

import numpy as np

from src.data import load_series, window_anchors
from src.metrics import per_sensor_mae
from src.ranking import sensor_shift_tau, window_shift_tau
from src.splits import heldout_sensors, in_pool

PEAK_WINDOWS = {
    "base [7,10)+[16,19)": ((7, 10), (16, 19)),
    "wide [6,11)+[15,20)": ((6, 11), (15, 20)),
    "narrow [7,9)+[16,18)": ((7, 9), (16, 18)),
    "later [8,11)+[17,20)": ((8, 11), (17, 20)),
}
CLUSTER_COUNTS = (4, 5, 6, 8)
FROZEN = ("chronos2", "timesfm")


def _ratio(a, b):
    return float(np.nanmean(b) / np.nanmean(a) - 1)


def peak_variants(name, y_true, preds):
    _, timestamps, _ = load_series(name)
    hour = np.asarray(timestamps.hour)
    anchors = window_anchors(name)["test"][:len(y_true)]
    models = list(preds)
    out = {}
    for label, ((a1, b1), (a2, b2)) in PEAK_WINDOWS.items():
        peak = ((hour >= a1) & (hour < b1)) | ((hour >= a2) & (hour < b2))
        in_dist, shifted = in_pool(anchors, peak), in_pool(anchors, ~peak)
        if not in_dist.any():
            # LargeST-SD steps are 15 min, so 12 target steps span 3 hours
            # and no window fits inside a 2-hour peak window.
            out[label] = {"tau": None, "note": "no window has all 12 targets in the peak window"}
            continue
        a = {m: per_sensor_mae(y_true[in_dist], preds[m][in_dist]) for m in models}
        b = {m: per_sensor_mae(y_true[shifted], preds[m][shifted]) for m in models}
        r = window_shift_tau(models, np.stack([a[m] for m in models]), np.stack([b[m] for m in models]))
        out[label] = {"tau": r["tau"], "degradation": {m: _ratio(a[m], b[m]) for m in FROZEN}}
    return out


def cluster_variants(name, y_true, preds):
    models = list(preds)
    full = {m: per_sensor_mae(y_true, preds[m]) for m in models}
    out = {}
    for k in CLUSTER_COUNTS:
        retained, held_out = heldout_sensors(name, k=k)
        a = np.stack([full[m][retained] for m in models])
        b = np.stack([full[m][held_out] for m in models])
        r = sensor_shift_tau(models, a, b)
        out[f"k={k}"] = {"tau": r["tau"], "held_out": len(held_out),
                         "degradation": {m: _ratio(full[m][retained], full[m][held_out]) for m in FROZEN}}
    return out
