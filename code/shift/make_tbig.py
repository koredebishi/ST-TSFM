#!/usr/bin/env python3
"""T-big (decisions.md D30) — unified MAE grid for the IEEE paper.

Every model x dataset x {standard test, S1 weekend pool, S2 off-peak pool,
S3 holdout sensors}, average masked MAE over h1..h12. Assembly-only: no
training, no new evaluation semantics. Reuses the verified T3 loading path
(shift.t3_rankings.load_preds / MODELS) and eval.metrics.mae; pools come
from shift.splits with the D8 all-12-target-steps rule on the frozen test
anchors. Cell convention matches the banked matrix exactly (pooled masked
MAE over windows x horizons x sensors — see
scripts/verification/frozen_degradation_matrix.py mae_block).

Parity anchors (loud):
  1. Chronos-2 SD-S1 pool MAE must reproduce the banked
     frozen_degradation_matrix.json shifted average (Table IV's 26.14).
  2. Per-model standard-test MAE is reported against the banked
     t1_grid.json average as t1_delta (advisory only: T-big restricts all
     models to the common window prefix; T1 did not).

Output: results/shift/tbig_grid.json + a booktabs LaTeX body printed to
stdout and written to results/shift/tbig_table.tex.
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
from eval.metrics import mae as metrics_mae  # noqa: E402
from scripts.verification.frozen_degradation_matrix import truth  # noqa: E402
from shift.splits import (classify_targets, is_offpeak,  # noqa: E402
                          is_weekend, s3_split)
from shift.t3_rankings import MODELS, load_preds  # noqa: E402

ROW_ORDER = ["ha", "arima", "dcrnn", "gwnet", "staeformer", "chronos2", "timesfm"]
ROW_LABEL = {"ha": "HA", "arima": "ARIMA", "dcrnn": "DCRNN",
             "gwnet": "Graph WaveNet", "staeformer": "STAEformer",
             "chronos2": "Chronos-2 (frozen)", "timesfm": "TimesFM (frozen)"}
# D30 choice (b): bold ONLY the frozen-model cells of the two failing
# settings — LargeST-SD S1 and METR-LA S3.
BOLD = {("largest_sd", "s1", "chronos2"), ("largest_sd", "s1", "timesfm"),
        ("metr_la", "s3", "chronos2"), ("metr_la", "s3", "timesfm")}


def main():
    out = {"spec": "decisions.md D30; T3 re-slice convention (standard-split "
                   "models on the frozen test anchors, D8 all-12-target-steps "
                   "rule); metric = eval.metrics.mae (mask_zero), pooled over "
                   "windows x h1..h12 x sensors",
           "datasets": {}}
    t1 = json.loads((REPO / "results/shift/t1_grid.json").read_text())
    fdm = json.loads(
        (REPO / "results/shift/frozen_degradation_matrix.json").read_text())

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
            cell = {"std": metrics_mae(yt, yp),
                    "s1": metrics_mae(yt[wknd], yp[wknd]),
                    "s2": metrics_mae(yt[offp], yp[offp]),
                    "s3": metrics_mae(yt[:, :, hold], yp[:, :, hold])}
            t1cell = (t1.get(m) or {}).get(ds)
            cell = {k: round(v, 4) for k, v in cell.items()}
            cell["t1_avg_banked"] = None if not t1cell else t1cell["avg"]
            cell["t1_delta"] = (None if not t1cell
                                else round(cell["std"] - t1cell["avg"], 4))
            block["models"][m] = cell
            print(f"  {m:10s} std={cell['std']:.2f} s1={cell['s1']:.2f} "
                  f"s2={cell['s2']:.2f} s3={cell['s3']:.2f} "
                  f"(t1_delta={cell['t1_delta']})", flush=True)
        out["datasets"][ds] = block

    # Parity anchor 1 (loud): same pipeline as the banked matrix / Table IV.
    want = fdm["matrix"]["largest_sd"]["S1_weekday_to_weekend"]["shifted"]["avg"]
    got = out["datasets"]["largest_sd"]["models"]["chronos2"]["s1"]
    assert abs(got - want) < 0.02, f"parity anchor FAILED: {got} vs {want}"
    print(f"parity anchor OK: chronos2 SD-S1 {got} vs banked {want}")
    out["parity_anchor"] = {"chronos2_sd_s1_here": got,
                            "banked_matrix_avg": want}
    out["host"] = os.uname().nodename
    out["job_id"] = os.environ.get("SLURM_JOB_ID")
    o = REPO / "results/shift/tbig_grid.json"
    o.write_text(json.dumps(out, indent=2))
    print(f"wrote {o}")

    # ---- LaTeX body (booktabs, 12 numeric columns) ----
    def fmt(ds, key, m):
        c = out["datasets"][ds]["models"].get(m)
        if c is None:
            return "--"
        v = f"{c[key]:.2f}"
        return rf"\textbf{{{v}}}" if (ds, key, m) in BOLD else v

    lines = []
    for m in ROW_ORDER:
        cells = [fmt(ds, k, m) for ds in ("metr_la", "pems_bay", "largest_sd")
                 for k in ("std", "s1", "s2", "s3")]
        lines.append(ROW_LABEL[m] + " & " + " & ".join(cells) + r" \\")
    tex = "\n".join(lines)
    (REPO / "results/shift/tbig_table.tex").write_text(tex + "\n")
    print("---- LaTeX body ----")
    print(tex)


if __name__ == "__main__":
    main()
