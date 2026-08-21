#!/usr/bin/env python3
"""Multi-ruler severity (decisions.md D34): KS and mean-difference beside W1.

Audit question: is the severity blindness a W1 artifact, or does
the whole marginal family fail? Chen (chenwass) compared rulers
(Euclidean, DTW); we add the marginal analogues on OUR banked pools:
  - ks:    two-sample Kolmogorov-Smirnov statistic (shape-sensitive)
  - dmean: |mean(a) - mean(b)| (Euclidean-of-means analog; the pure
           affine/level component — pre-registered as the MOST
           anti-predictive, since Prop 1 proves the model class cannot
           feel exactly this component)
Estimator conventions IDENTICAL to shift/severity.py (reused directly):
per-sensor for S1/S2 reported as the sensor mean; pooled for S3; the
same _target_rows / _clean(_ZERO_TOL) machinery. W1 is recomputed through
the same loop and asserted against the banked severity_summary.json.

Correlation: for each ruler, severity normalized within dataset (by the
dataset's max across S1/S2/S3), Spearman rho + permutation p against the
banked frozen degradations (18 model-setting evaluations), via the
self-tested spearman_perm of shift/make_figures.py (n_perm=20000,
seed 2023) — the same statistic as the paper's F5.

Output: results/shift/severity_rulers.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_loaders.windows import frozen_anchors, load_raw  # noqa: E402
from shift.make_figures import spearman_perm  # noqa: E402
from shift.severity import _clean, _target_rows, w1  # noqa: E402
from shift.splits import s1_split, s2_split, s3_split  # noqa: E402

DATASETS = ("metr_la", "pems_bay", "largest_sd")
SKEY = {"S1": "S1_weekday_to_weekend", "S2": "S2_peak_to_offpeak",
        "S3": "S3_spatial_holdout"}
TF_FILE = {"metr_la": "timesfm_metr_la_degradation.json",
           "pems_bay": "timesfm_bay_degradation.json",
           "largest_sd": "timesfm_largest_sd_degradation.json"}


def ks(a: np.ndarray, b: np.ndarray) -> float:
    return float(ks_2samp(a, b, method="asymp").statistic)


def dmean(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(a.mean() - b.mean()))


RULERS = {"w1": w1, "ks": ks, "dmean": dmean}


def temporal(dataset: str, split) -> dict:
    raw, _, _ = load_raw(dataset)
    ra = _target_rows(split.train_anchors)
    rb = _target_rows(split.test_anchors)
    acc = {k: [] for k in RULERS}
    skipped = 0
    for j in range(raw.shape[1]):
        a, b = _clean(raw[ra, j]), _clean(raw[rb, j])
        if len(a) < 10 or len(b) < 10:
            skipped += 1
            continue
        for k, fn in RULERS.items():
            acc[k].append(fn(a, b))
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["n_sensors_skipped"] = skipped
    return out


def spatial(dataset: str, split) -> dict:
    raw, _, _ = load_raw(dataset)
    rows = _target_rows(frozen_anchors(dataset)["test"])
    a = _clean(raw[np.ix_(rows, split.train_sensors)].ravel())
    b = _clean(raw[np.ix_(rows, split.test_sensors)].ravel())
    return {k: fn(a, b) for k, fn in RULERS.items()}


def main():
    sev = {}
    for ds in DATASETS:
        sev[ds] = {"S1": temporal(ds, s1_split(ds)),
                   "S2": temporal(ds, s2_split(ds)),
                   "S3": spatial(ds, s3_split(ds))}
        print(ds, json.dumps({s: {k: round(v, 3) for k, v in c.items()}
                              for s, c in sev[ds].items()}), flush=True)

    # ---- parity: recomputed W1 must match the banked summary ----
    banked = json.loads(
        (REPO / "results/shift/severity_summary.json").read_text())["severities"]
    for ds in DATASETS:
        for s in ("S1", "S2"):
            want = banked[ds][s]["severity_w1_mean"]
            got = sev[ds][s]["w1"]
            assert abs(got - want) < 0.01, (ds, s, got, want)
        want = banked[ds]["S3"]["severity_w1_pooled"]
        got = sev[ds]["S3"]["w1"]
        assert abs(got - want) < 0.01, (ds, "S3", got, want)
    print("W1 parity vs banked severity_summary.json: OK", flush=True)

    # ---- degradations (banked) ----
    fdm = json.loads((REPO / "results/shift/frozen_degradation_matrix.json"
                      ).read_text())["matrix"]
    degr = {}
    for ds in DATASETS:
        tf = json.loads((REPO / "results/shift" / TF_FILE[ds]).read_text())["cell"]
        for s in ("S1", "S2", "S3"):
            degr[("chronos2", ds, s)] = fdm[ds][SKEY[s]]["rel_degradation"]["avg"]
            degr[("timesfm", ds, s)] = tf[SKEY[s]]["rel_degradation"]["avg"]

    # ---- per-ruler correlation over the 18 evaluations ----
    corr = {}
    for k in RULERS:
        xs, ys = [], []
        for ds in DATASETS:
            mx = max(sev[ds][s][k] for s in ("S1", "S2", "S3"))
            for s in ("S1", "S2", "S3"):
                nrm = sev[ds][s][k] / mx
                for backbone in ("chronos2", "timesfm"):
                    xs.append(nrm)
                    ys.append(degr[(backbone, ds, s)])
        rho, p = spearman_perm(np.array(xs), np.array(ys))
        corr[k] = {"spearman_rho": round(float(rho), 4),
                   "perm_p": round(float(p), 5), "n": len(xs)}
        print(f"ruler={k}: rho={rho:.3f} p={p:.4f}", flush=True)

    out = {"spec": "decisions.md D34; conventions of shift/severity.py; "
                   "correlation statistic identical to F5 (spearman_perm, "
                   "n_perm=20000, seed 2023)",
           "severities": sev, "correlations": corr,
           "host": os.uname().nodename,
           "job_id": os.environ.get("SLURM_JOB_ID")}
    o = REPO / "results/shift/severity_rulers.json"
    o.write_text(json.dumps(out, indent=2))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
