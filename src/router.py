"""Timestamp router on LargeST-SD (Table V, last row).

For weekdays and for weekends separately, pick the model with the lowest
MAE on the validation windows of that day type. The candidates are HA,
Chronos-2 and TimesFM. Each test window is then forecast by the model
chosen for its day type; a window whose targets are not all on a weekend
counts as a weekday window. Only validation data takes part in the choice.
"""
from __future__ import annotations

import numpy as np

from src.data import TARGET_STEPS, largest_sd_flow, load_series, window_anchors
from src.metrics import mae
from src.splits import in_pool, is_weekend

CANDIDATES = ("ha", "chronos2", "timesfm")


def choose(val_preds):
    """val_preds: model -> LargeST-SD validation forecasts. Returns the
    chosen model per day type and the validation MAEs behind the choice."""
    _, timestamps, _ = load_series("largest_sd")
    anchors = window_anchors("largest_sd")["val"]
    y_val = largest_sd_flow()[anchors[:, None] + 1 + np.arange(TARGET_STEPS)]
    weekend = is_weekend(timestamps)
    pools = {"weekday": in_pool(anchors, ~weekend), "weekend": in_pool(anchors, weekend)}
    val_mae = {day: {m: mae(y_val[mask], val_preds[m][mask]) for m in CANDIDATES}
               for day, mask in pools.items()}
    chosen = {day: min(scores, key=scores.get) for day, scores in val_mae.items()}
    return chosen, val_mae


def route(chosen, y_true, test_preds):
    """MAE of the routed forecasts on the full test period and on weekends."""
    _, timestamps, _ = load_series("largest_sd")
    anchors = window_anchors("largest_sd")["test"][:len(y_true)]
    weekend = in_pool(anchors, is_weekend(timestamps))
    routed = np.where(weekend[:, None, None], test_preds[chosen["weekend"]], test_preds[chosen["weekday"]])
    return {"full_test": mae(y_true, routed), "weekend": mae(y_true[weekend], routed[weekend])}
