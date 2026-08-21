#!/usr/bin/env python3
"""D37 E5b: per-ruler severity correlations under the unified estimators.

For each E1b variant and each ruler (w1, ks, dmean): descriptive
Spearman over the 18 model-setting evaluations with the self-tested
permutation p (shift.make_figures.spearman_perm, the same helper the
original figure used), plus the exact 9! setting-level test from E5's
machinery. Feeds Finding 1 prose and the Fig. 2 re-render.

Parity: under variant per_sensor_s3 the S1/S2 severities equal the
banked severity_rulers.json values, so any change vs the banked
correlations is attributable to the S3 unification alone.
Output: results/shift/e5b_ruler_correlations.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shift.e125_review_response import exact_perm9  # noqa: E402
from shift.make_figures import spearman_perm  # noqa: E402

SH = REPO / "results/shift"
DATASETS = ("metr_la", "pems_bay", "largest_sd")
SKEY = {"S1": "S1_weekday_to_weekend", "S2": "S2_peak_to_offpeak",
        "S3": "S3_spatial_holdout"}
TF_FILE = {"metr_la": "timesfm_metr_la_degradation.json",
           "pems_bay": "timesfm_bay_degradation.json",
           "largest_sd": "timesfm_largest_sd_degradation.json"}


def main():
    e1b = json.loads((SH / "e1b_unified_rulers.json").read_text())["variants"]
    fdm = json.loads((SH / "frozen_degradation_matrix.json").read_text())["matrix"]
    tf = {ds: json.loads((SH / TF_FILE[ds]).read_text())["cell"]
          for ds in DATASETS}
    deg = {}
    for ds in DATASETS:
        for s in ("S1", "S2", "S3"):
            deg[(ds, s, "chronos2")] = fdm[ds][SKEY[s]]["rel_degradation"]["avg"]
            deg[(ds, s, "timesfm")] = tf[ds][SKEY[s]]["rel_degradation"]["avg"]

    out = {"spec": "decisions.md D37 E5b", "variants": {}}
    for vname, sev in e1b.items():
        vent = {}
        for ruler in ("w1", "ks", "dmean"):
            norm = {}
            for ds in DATASETS:
                mx = max(sev[ds][s][ruler] for s in ("S1", "S2", "S3"))
                norm[ds] = {s: sev[ds][s][ruler] / mx
                            for s in ("S1", "S2", "S3")}
            xs, ys = [], []
            for ds in DATASETS:
                for s in ("S1", "S2", "S3"):
                    for m in ("chronos2", "timesfm"):
                        xs.append(norm[ds][s])
                        ys.append(100.0 * deg[(ds, s, m)])
            rho, p = spearman_perm(np.array(xs), np.array(ys))
            x9 = np.array([norm[ds][s] for ds in DATASETS
                           for s in ("S1", "S2", "S3")])
            y9 = np.array([np.mean([deg[(ds, s, "chronos2")],
                                    deg[(ds, s, "timesfm")]])
                           for ds in DATASETS for s in ("S1", "S2", "S3")])
            rho9, p9 = exact_perm9(x9, y9)
            vent[ruler] = {"rho_18": float(rho), "perm_p_18": float(p),
                           "rho_9": float(rho9),
                           "p_9_exact_twosided": float(p9)}
            print(f"[e5b] {vname}/{ruler}: rho18={rho:+.3f} p={p:.4f} "
                  f"rho9={rho9:+.3f} p9={p9:.4f}", flush=True)
        out["variants"][vname] = vent
    out["job"] = os.environ.get("SLURM_JOB_ID")
    (SH / "e5b_ruler_correlations.json").write_text(json.dumps(out, indent=1))
    print(f"[e5b] wrote {SH/'e5b_ruler_correlations.json'}", flush=True)


if __name__ == "__main__":
    main()
