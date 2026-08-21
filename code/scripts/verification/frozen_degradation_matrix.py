#!/usr/bin/env python3
"""Frozen-TSFM degradation matrix (free; no model compute).

For each dataset x shift, re-slice the already-banked frozen Chronos-2
forecasts into in-distribution vs shifted subsets and compare MAE. Answers:
is there ANY shift x dataset x horizon where frozen zero-shot Chronos-2
degrades enough for TTRA to recover? (ttra_training_spec assumption #1.)

- S1: weekday (train regime) vs weekend (shifted).        [temporal]
- S2: peak (train regime) vs off-peak (shifted).          [demand]
- S3: train-region sensors vs held-out-cluster sensors.   [spatial]
  NOTE: zero-shot Chronos-2 treats every sensor independently and never used
  the region split, so S3 degradation is expected ~0 by construction.

Reuses data_loaders.windows, shift.splits, eval.metrics. Banked preds:
results/baselines/chronos2/{ds}/y_pred.npy (--save-preds).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data_loaders.windows import (SEQ_OUT, DATASETS, frozen_anchors,  # noqa: E402
                                  load_raw)
from eval.metrics import all_metrics  # noqa: E402
from shift.splits import (classify_targets, is_offpeak, is_peak,  # noqa: E402
                          is_weekday, is_weekend, s3_split)

OUT = REPO / "results/shift/frozen_degradation_matrix.json"
HORIZONS = (3, 6, 12)


def truth(ds: str) -> np.ndarray:
    """(B, 12, N) raw targets aligned with the banked y_pred order."""
    cfg = DATASETS[ds]
    if ds == "largest_sd":
        npz = np.load(cfg["dir"] / "his.npz")
        raw = (npz["data"][..., 0].astype(np.float64) * float(npz["std"])
               + float(npz["mean"]))
        idx = frozen_anchors(ds)["test"]
        y = np.zeros((len(idx), SEQ_OUT, raw.shape[1]))
        for k, i in enumerate(idx):
            y[k] = raw[i + 1 : i + 1 + SEQ_OUT]
        return y
    return np.load(cfg["processed"] / "test.npz")["y"][..., 0]


def mae_block(yt, yp, N):
    per_h = {h: round(all_metrics(yt[:, h - 1], yp[:, h - 1],
                                  mask_zero=True)["mae"], 4) for h in HORIZONS}
    avg = round(all_metrics(yt.reshape(-1, N), yp.reshape(-1, N),
                            mask_zero=True)["mae"], 4)
    return {"per_horizon": per_h, "avg": avg}


def rel(shift_b, base_b):
    out = {f"h{h}": round((shift_b["per_horizon"][h] - base_b["per_horizon"][h])
                          / base_b["per_horizon"][h], 4) for h in HORIZONS}
    out["avg"] = round((shift_b["avg"] - base_b["avg"]) / base_b["avg"], 4)
    return out


def main():
    matrix = {}
    for ds in ("metr_la", "pems_bay", "largest_sd"):
        yp = np.load(REPO / f"results/baselines/chronos2/{ds}/y_pred.npy").astype(np.float64)
        yt = truth(ds)
        assert yp.shape == yt.shape, f"{ds}: {yp.shape} vs {yt.shape}"
        N = yt.shape[2]
        _, ts, _ = load_raw(ds)
        anc = frozen_anchors(ds)["test"]
        cell = {}

        for name, base_pred, shift_pred in (
            ("S1_weekday_to_weekend", is_weekday, is_weekend),
            ("S2_peak_to_offpeak", is_peak, is_offpeak),
        ):
            base_m = classify_targets(anc, np.asarray(base_pred(ts)))
            shift_m = classify_targets(anc, np.asarray(shift_pred(ts)))
            base_b = mae_block(yt[base_m], yp[base_m], N)
            shift_b = mae_block(yt[shift_m], yp[shift_m], N)
            cell[name] = {"in_distribution": base_b, "shifted": shift_b,
                          "n_in": int(base_m.sum()), "n_shift": int(shift_m.sum()),
                          "rel_degradation": rel(shift_b, base_b)}

        # S3 spatial: split sensor columns
        sp = s3_split(ds)
        base_b = mae_block(yt[:, :, sp.train_sensors], yp[:, :, sp.train_sensors],
                           len(sp.train_sensors))
        shift_b = mae_block(yt[:, :, sp.test_sensors], yp[:, :, sp.test_sensors],
                            len(sp.test_sensors))
        cell["S3_spatial_holdout"] = {
            "in_distribution": base_b, "shifted": shift_b,
            "n_train_sensors": int(len(sp.train_sensors)),
            "n_holdout_sensors": int(len(sp.test_sensors)),
            "rel_degradation": rel(shift_b, base_b),
            "note": "expected ~0: zero-shot TSFM never used the region split",
        }
        matrix[ds] = cell
        print(f"[matrix] {ds} done", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": "frozen chronos-2 (zero-shot)",
                               "matrix": matrix}, indent=2))
    # compact readout: avg relative degradation per cell
    print("\n=== avg relative degradation (shifted vs in-distribution) ===")
    for ds, cell in matrix.items():
        for shift, c in cell.items():
            d = c["rel_degradation"]["avg"]
            print(f"  {ds:12s} {shift:24s} avg {d:+.1%}   "
                  f"h12 {c['rel_degradation']['h12']:+.1%}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
