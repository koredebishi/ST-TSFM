#!/usr/bin/env python3
"""Free TimesFM degradation slice — sharded datasets (BAY, SD).

Cross-backbone check of the Chronos-2 finding (frozen TSFM shift-robustness +
severity-vs-degradation decoupling). Assembles the banked TimesFM shard
partials into one y_pred.npy (coverage-asserted, same as --merge), then runs
the same S1/S2/S3 degradation slice via the matrix's own helpers (no dup).
No model compute. Usage: python timesfm_bay_degradation.py <dataset>
(default pems_bay).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data_loaders.windows import frozen_anchors, load_raw  # noqa: E402
from scripts.verification.frozen_degradation_matrix import (  # noqa: E402
    mae_block, rel, truth)
from shift.splits import (classify_targets, is_offpeak, is_peak,  # noqa: E402
                          is_weekday, is_weekend, s3_split)

DS = sys.argv[1] if len(sys.argv) > 1 else "pems_bay"
PART = REPO / f"results/baselines/timesfm/{DS}/partials"
OUT_PRED = REPO / f"results/baselines/timesfm/{DS}/y_pred.npy"
OUT = REPO / f"results/shift/timesfm_{DS}_degradation.json"
HORIZONS = (3, 6, 12)


def assemble():
    n_test = len(frozen_anchors(DS)["test"])
    parts = sorted(PART.glob("pred_*.npy"),
                   key=lambda p: int(p.stem.split("_")[1]))
    y = None
    covered = np.zeros(n_test, dtype=bool)
    for p in parts:
        ws, we = int(p.stem.split("_")[1]), int(p.stem.split("_")[2])
        arr = np.load(p)
        if y is None:
            y = np.full((n_test, arr.shape[1], arr.shape[2]), np.nan, np.float32)
        y[ws:we] = arr
        covered[ws:we] = True
    assert covered.all(), f"incomplete shard coverage: {(~covered).sum()} gaps"
    assert not np.isnan(y).any(), "NaNs after assembly"
    np.save(OUT_PRED, y)
    print(f"[assemble] {len(parts)} shards -> {y.shape} -> {OUT_PRED}", flush=True)
    return y.astype(np.float64)


def main():
    yp = assemble()
    yt = truth(DS)
    assert yp.shape == yt.shape, f"{yp.shape} vs {yt.shape}"
    N = yt.shape[2]
    _, ts, _ = load_raw(DS)
    anc = frozen_anchors(DS)["test"]
    cell = {}
    for name, base_p, shift_p in (("S1_weekday_to_weekend", is_weekday, is_weekend),
                                  ("S2_peak_to_offpeak", is_peak, is_offpeak)):
        bm = classify_targets(anc, np.asarray(base_p(ts)))
        sm = classify_targets(anc, np.asarray(shift_p(ts)))
        b, s = mae_block(yt[bm], yp[bm], N), mae_block(yt[sm], yp[sm], N)
        cell[name] = {"in_distribution": b, "shifted": s,
                      "n_in": int(bm.sum()), "n_shift": int(sm.sum()),
                      "rel_degradation": rel(s, b)}
    sp = s3_split(DS)
    b = mae_block(yt[:, :, sp.train_sensors], yp[:, :, sp.train_sensors], len(sp.train_sensors))
    s = mae_block(yt[:, :, sp.test_sensors], yp[:, :, sp.test_sensors], len(sp.test_sensors))
    cell["S3_spatial_holdout"] = {"in_distribution": b, "shifted": s,
                                  "rel_degradation": rel(s, b)}

    OUT.write_text(json.dumps({"model": "frozen timesfm-2.5 (zero-shot)",
                               "dataset": DS, "cell": cell}, indent=2))
    print(f"\n=== TimesFM {DS}: avg rel degradation (shifted vs in-dist) ===")
    for shift, c in cell.items():
        d = c["rel_degradation"]
        print(f"  {shift:24s} avg {d['avg']:+.1%}   h12 {d['h12']:+.1%}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
