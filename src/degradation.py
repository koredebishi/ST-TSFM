"""How much worse a model forecasts on the shifted pool.

degradation = MAE(shifted pool) / MAE(in-distribution pool) - 1

Both pools come from the same test period and the same model, so the
ratio compares like with like. It is reported for the average over all
12 steps and for steps 3, 6 and 12 (15, 30 and 60 minutes on METR-LA and
PEMS-BAY).
"""
from __future__ import annotations

import numpy as np

from src.data import INPUT_STEPS, TARGET_STEPS, load_series, window_anchors
from src.metrics import ZERO, mae, mase, smape_nonzero
from src.splits import heldout_sensors, test_pools

STEPS = (3, 6, 12)


def split_by_shift(name, shift, y_true, y_pred):
    """(true, pred) for the in-distribution pool and for the shifted pool."""
    if shift == "S3":
        retained, held_out = heldout_sensors(name)
        return ((y_true[:, :, retained], y_pred[:, :, retained]),
                (y_true[:, :, held_out], y_pred[:, :, held_out]))
    in_dist, shifted = test_pools(name, shift)
    return ((y_true[in_dist], y_pred[in_dist]),
            (y_true[shifted], y_pred[shifted]))


def mae_summary(y_true, y_pred):
    """MAE over all steps, and at steps 3, 6 and 12."""
    out = {"avg": mae(y_true, y_pred)}
    for s in STEPS:
        out[f"h{s}"] = mae(y_true[:, s - 1], y_pred[:, s - 1])
    return out


def degradation(name, shift, y_true, y_pred):
    (t_id, p_id), (t_sh, p_sh) = split_by_shift(name, shift, y_true, y_pred)
    base, moved = mae_summary(t_id, p_id), mae_summary(t_sh, p_sh)
    # pool sizes: windows for S1 / S2, sensors for S3
    axis = 2 if shift == "S3" else 0
    return {"mae_in_dist": base, "mae_shifted": moved,
            "degradation": {k: moved[k] / base[k] - 1 for k in base},
            "pool_sizes": [t_id.shape[axis], t_sh.shape[axis]]}


def skill_change(name, shift, y_true, y_pred, y_ha):
    """Loss of skill against the historical average (HA) under the shift.

    skill = 1 - MAE(model) / MAE(HA), on each pool; the change is
    skill(in-distribution) - skill(shifted). A regime that is simply harder
    for every model raises both MAEs, so it does not change skill.
    """
    (t_id, p_id), (t_sh, p_sh) = split_by_shift(name, shift, y_true, y_pred)
    (_, h_id), (_, h_sh) = split_by_shift(name, shift, y_true, y_ha)
    skill_id = 1 - mae(t_id, p_id) / mae(t_id, h_id)
    skill_sh = 1 - mae(t_sh, p_sh) / mae(t_sh, h_sh)
    return skill_id - skill_sh


def naive_scale(name):
    """Per sensor: mean |y_t - y_{t-1}| over the training period, using
    only steps where both readings are non-zero (the MASE scale)."""
    values, _, _ = load_series(name)
    train = window_anchors(name)["train"]
    seg = values[train.min() - INPUT_STEPS + 1: train.max() + TARGET_STEPS + 1]
    step = np.abs(seg[1:] - seg[:-1])
    ok = (np.abs(seg[1:]) > ZERO) & (np.abs(seg[:-1]) > ZERO)
    n = ok.sum(axis=0)
    return np.where(n > 0, np.where(ok, step, 0).sum(axis=0) / np.maximum(n, 1), np.nan)


def relative_degradation(name, shift, y_true, y_pred, scale=None):
    """Degradation (average over all steps) under sMAPE and under MASE."""
    (t_id, p_id), (t_sh, p_sh) = split_by_shift(name, shift, y_true, y_pred)
    s_id = s_sh = scale
    if shift == "S3":
        retained, held_out = heldout_sensors(name)
        s_id, s_sh = scale[retained], scale[held_out]
    return {"smape": smape_nonzero(t_sh, p_sh) / smape_nonzero(t_id, p_id) - 1,
            "mase": mase(t_sh, p_sh, s_sh) / mase(t_id, p_id, s_id) - 1}
