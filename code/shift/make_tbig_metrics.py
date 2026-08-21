#!/usr/bin/env python3
"""Full-metric T-big grid (decisions.md D35) — makes the Table III release
promise true: RMSE, MAPE, sMAPE, R^2, and 95th-percentile error for every
model x dataset x {std, S1, S2, S3} cell, alongside MAE.

Assembly-only, zero new evaluation semantics: loading and pool construction
are copied line-for-line from the verified shift/make_tbig.py (T3 re-slice
convention on the frozen test anchors, D8 all-12-target-steps rule,
common-window-prefix truncation, y_true drift assert); the per-cell metric
call is eval.metrics.all_metrics instead of eval.metrics.mae.

Parity (loud, per D35): every cell's MAE here must reproduce the banked
tbig_grid.json value to within 5e-4 (its stored rounding). Any mismatch is
a hard assert naming the cell. This subsumes the D30 single-cell anchor.

Output: results/shift/tbig_metrics_full.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_loaders.windows import frozen_anchors, load_raw  # noqa: E402
from eval.metrics import all_metrics  # noqa: E402
from scripts.verification.frozen_degradation_matrix import truth  # noqa: E402
from shift.splits import (classify_targets, is_offpeak,  # noqa: E402
                          is_weekend, s3_split)
from shift.t3_rankings import MODELS, load_preds  # noqa: E402


def main():
    out = {"spec": "decisions.md D35; identical pools to tbig_grid.json "
                   "(D30/D8 conventions); metrics = eval.metrics.all_metrics "
                   "(mae, rmse, mape, smape, r2, p95; mask_zero, "
                   "_ZERO_TOL=0.001); per-cell MAE parity vs tbig_grid.json "
                   "asserted at 5e-4",
           "datasets": {}}
    banked = json.loads(
        (REPO / "results/shift/tbig_grid.json").read_text())["datasets"]
    n_parity = 0

    for ds, models in MODELS.items():
        print(f"=== {ds} ===", flush=True)
        yt = truth(ds).astype(np.float32)          # (n_test, 12, N)
        n_test, _, N = yt.shape
        anc = frozen_anchors(ds)["test"]
        assert len(anc) == n_test
        _, ts, _ = load_raw(ds)

        loaded = {m: load_preds(m, ds, n_test, N, yt) for m in models}
        n_use = min(min(yp.shape[0] for yp, _ in loaded.values()), n_test)
        if n_use < n_test:
            print(f"  [note] common window prefix {n_use}/{n_test}", flush=True)
        yt, anc = yt[:n_use], anc[:n_use]
        preds = {}
        for m, (yp, yl) in loaded.items():
            if yl is not None:
                d = float(np.abs(yl[:n_use] - yt).mean())
                assert d < 0.05, f"{m}/{ds}: y_true drift vs raw truth = {d}"
            preds[m] = yp[:n_use]

        wknd = classify_targets(anc, np.asarray(is_weekend(ts)))
        offp = classify_targets(anc, np.asarray(is_offpeak(ts)))
        hold = np.asarray(s3_split(ds).test_sensors)
        assert wknd.sum() > 0 and offp.sum() > 0 and len(hold) > 0

        block = {"n_windows": int(n_use), "n_windows_s1": int(wknd.sum()),
                 "n_windows_s2": int(offp.sum()),
                 "n_holdout_sensors": int(len(hold)), "models": {}}
        for m in models:
            yp = preds[m]
            cell = {"std": all_metrics(yt, yp),
                    "s1": all_metrics(yt[wknd], yp[wknd]),
                    "s2": all_metrics(yt[offp], yp[offp]),
                    "s3": all_metrics(yt[:, :, hold], yp[:, :, hold])}
            for k, mets in cell.items():
                want = banked[ds]["models"][m][k]
                got = mets["mae"]
                assert abs(got - want) < 5e-4, (
                    f"MAE parity FAILED at {ds}/{m}/{k}: {got} vs banked {want}")
                n_parity += 1
            block["models"][m] = {
                k: {name: round(v, 6) for name, v in mets.items()}
                for k, mets in cell.items()}
            print(f"  {m:10s} " + " ".join(
                f"{k}:mae={cell[k]['mae']:.2f},rmse={cell[k]['rmse']:.2f}"
                for k in ("std", "s1", "s2", "s3")), flush=True)
        out["datasets"][ds] = block

    print(f"parity OK: all {n_parity} cells reproduce tbig_grid.json MAE")
    out["parity"] = {"cells_checked": n_parity, "tolerance": 5e-4}
    out["host"] = os.uname().nodename
    out["job_id"] = os.environ.get("SLURM_JOB_ID")
    o = REPO / "results/shift/tbig_metrics_full.json"
    o.write_text(json.dumps(out, indent=2))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
