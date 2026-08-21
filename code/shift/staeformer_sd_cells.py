#!/usr/bin/env python3
"""D38 T7 assembly: STAEformer's four LargeST-SD Table II cells.

Runs AFTER the D38 training + its --save-preds output exist. Re-slices
the saved predictions on the frozen test windows with the exact tbig
conventions (same pool predicates, same masking, same common-prefix
rule) to produce Std / S1 / S2 / S3 pool MAEs, plus the supplementary
M=7 tau for the RELEASED ARTIFACTS ONLY (paper keeps pre-registered
M=6; decisions.md D38).

Parity (loud): the Std cell must reproduce the run's own
test_metrics.json average MAE; every other SD model's Std cell must
reproduce tbig_grid.json.

Output: results/shift/staeformer_sd_cells.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scipy.stats import kendalltau  # noqa: E402

from data_loaders.windows import frozen_anchors, load_raw  # noqa: E402
from eval.metrics import mae as metrics_mae  # noqa: E402
from scripts.verification.frozen_degradation_matrix import truth  # noqa: E402
from shift.splits import (classify_targets, is_offpeak, is_peak,  # noqa: E402
                          is_weekday, is_weekend, s3_split)
from shift.t3_rankings import MODELS, load_preds  # noqa: E402

DS = "largest_sd"
SH = REPO / "results/shift"
STAE_DIR = REPO / "results/baselines/staeformer/largest_sd"


def pool_mae(yt, yp):
    return float(metrics_mae(yt.astype(np.float64), yp.astype(np.float64),
                             mask_zero=True))


def main():
    yt = truth(DS).astype(np.float32)
    n_test, _, N = yt.shape
    anc = frozen_anchors(DS)["test"]
    _, ts, _ = load_raw(DS)

    stae = np.load(STAE_DIR / "y_pred.npy").astype(np.float32)
    assert stae.shape[1:] == (12, N), stae.shape
    models = list(MODELS[DS])
    loaded = {m: load_preds(m, DS, n_test, N, yt) for m in models}
    n_use = min(min(yp.shape[0] for yp, _ in loaded.values()),
                stae.shape[0], n_test)
    yt, anc = yt[:n_use], np.asarray(anc)[:n_use]
    preds = {m: loaded[m][0][:n_use] for m in models}
    preds["staeformer"] = stae[:n_use]

    sp3 = s3_split(DS)
    bm_s1 = classify_targets(anc, np.asarray(is_weekday(ts)))
    sm_s1 = classify_targets(anc, np.asarray(is_weekend(ts)))
    bm_s2 = classify_targets(anc, np.asarray(is_peak(ts)))
    sm_s2 = classify_targets(anc, np.asarray(is_offpeak(ts)))

    def cells(yp):
        return {
            "std": pool_mae(yt, yp),
            "s1": pool_mae(yt[sm_s1], yp[sm_s1]),
            "s2": pool_mae(yt[sm_s2], yp[sm_s2]),
            "s3": pool_mae(yt[:, :, sp3.test_sensors],
                           yp[:, :, sp3.test_sensors]),
        }

    tbig = json.loads((SH / "tbig_grid.json").read_text())
    banked = tbig["datasets"][DS]["models"]
    grid = {}
    for m, yp in preds.items():
        grid[m] = cells(yp)
        if m in banked:
            assert abs(grid[m]["std"] - banked[m]["std"]) < 5e-3, \
                (m, grid[m]["std"], banked[m]["std"])
    own = json.loads((STAE_DIR / "test_metrics.json").read_text())
    own_avg = own["average"]["mae"]
    # self-parity on the run's own FULL test set: test_metrics.json
    # averages every STAEformer window, while the grid's std cell uses
    # the all-models common prefix (n_use), so the two are different
    # populations and must not be compared directly.
    stae_yt = np.load(STAE_DIR / "y_true.npy").astype(np.float32)
    full_mae = pool_mae(stae_yt, np.load(STAE_DIR / "y_pred.npy"
                                         ).astype(np.float32))
    assert abs(full_mae - own_avg) < 5e-3, (full_mae, own_avg)
    # alignment: the run's truth equals the frozen truth on the prefix
    assert stae_yt.shape[0] >= n_use, (stae_yt.shape, n_use)
    align = float(np.max(np.abs(stae_yt[:n_use] - yt)))
    assert align < 1e-3, f"truth misalignment: max|diff|={align}"
    print(f"[t7] parity OK; staeformer cells: {grid['staeformer']}",
          flush=True)

    # supplementary M=7 tau (weekday vs weekend orderings), artifacts only
    m7 = models + ["staeformer"]
    id_mae = {m: pool_mae(yt[bm_s1], preds[m][bm_s1]) for m in m7}
    sh_mae = {m: pool_mae(yt[sm_s1], preds[m][sm_s1]) for m in m7}
    t7, _ = kendalltau([id_mae[m] for m in m7], [sh_mae[m] for m in m7])
    t6, _ = kendalltau([id_mae[m] for m in models],
                       [sh_mae[m] for m in models])
    print(f"[t7] tau M=6 (pre-registered) parity check = {t6:.3f}; "
          f"supplementary tau M=7 = {t7:.3f}", flush=True)
    assert abs(t6 - 0.333) < 1e-2, f"M=6 tau parity broke: {t6}"

    out = {"spec": "decisions.md D38 T7", "grid": grid,
           "n_windows_used": int(n_use),
           "tau_m6_preregistered": float(t6),
           "tau_m7_supplementary_artifacts_only": float(t7),
           "s1_id_mae": id_mae, "s1_shift_mae": sh_mae,
           "job": os.environ.get("SLURM_JOB_ID")}
    (SH / "staeformer_sd_cells.json").write_text(json.dumps(out, indent=1))
    print(f"[t7] wrote {SH/'staeformer_sd_cells.json'}", flush=True)


if __name__ == "__main__":
    main()
