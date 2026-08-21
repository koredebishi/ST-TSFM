#!/usr/bin/env python3
"""D37 E1b: unify KS and mean-difference for S3 the same way as W-1.

The Fig. 2 scatter shows three severity measures; E1 unified only W-1.
This computes, for BOTH E1 variants, all three rulers per setting so a
re-rendered scatter uses one estimator family per variant:
- per_sensor_s3: S1/S2 = banked per-sensor means (severity_rulers.json,
  parity-asserted); S3 = per-held-out-sensor ruler against the pooled
  retained marginal, sensor mean (the E1 functional).
- pooled_all: every setting = ruler on pooled marginals (S3 equals the
  banked pooled values, parity-asserted where available).

Reuses shift.severity_rulers.RULERS (w1/ks/dmean) and shift.severity
row machinery verbatim. Output: results/shift/e1b_unified_rulers.json.
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
from shift.severity import _clean, _target_rows  # noqa: E402
from shift.severity_rulers import RULERS  # noqa: E402
from shift.splits import s1_split, s2_split, s3_split  # noqa: E402

SH = REPO / "results/shift"
DATASETS = ("metr_la", "pems_bay", "largest_sd")


def main():
    banked = json.loads((SH / "severity_rulers.json").read_text())["severities"]
    e1 = json.loads((SH / "e1_unified_severity.json").read_text())
    out = {"spec": "decisions.md D37 E1b", "variants": {}}
    per_sensor_s3, pooled_all = {}, {}
    for ds in DATASETS:
        raw, _, _ = load_raw(ds)
        s1, s2, s3 = s1_split(ds), s2_split(ds), s3_split(ds)
        rows_s3 = _target_rows(frozen_anchors(ds)["test"])
        ref = _clean(raw[np.ix_(rows_s3, s3.train_sensors)].ravel())
        pooled_hold = _clean(raw[np.ix_(rows_s3, s3.test_sensors)].ravel())

        ps = {"S1": dict(banked[ds]["S1"]), "S2": dict(banked[ds]["S2"])}
        s3_cell = {}
        for rname, rfn in RULERS.items():
            vals = []
            for j in s3.test_sensors:
                v = _clean(raw[rows_s3, j])
                if len(v) >= 10:
                    vals.append(rfn(v, ref))
            s3_cell[rname] = float(np.mean(vals))
        ps["S3"] = s3_cell
        # parity: E1b's per-sensor S3 W1 must reproduce E1's value
        e1_val = e1["variants"]["per_sensor_s3"][ds]["raw"]["S3"]
        assert abs(s3_cell["w1"] - e1_val) / e1_val < 1e-6, \
            (ds, s3_cell["w1"], e1_val)
        per_sensor_s3[ds] = ps

        pa = {}
        for sname, sp in (("S1", s1), ("S2", s2)):
            ra = _clean(raw[_target_rows(sp.train_anchors), :].ravel())
            rb = _clean(raw[_target_rows(sp.test_anchors), :].ravel())
            pa[sname] = {rn: float(rf(ra, rb)) for rn, rf in RULERS.items()}
        pa["S3"] = {rn: float(rf(ref, pooled_hold))
                    for rn, rf in RULERS.items()}
        # parity: pooled S3 W1 must reproduce the banked pooled value
        bk = banked[ds]["S3"]["w1"]
        assert abs(pa["S3"]["w1"] - bk) / bk < 5e-3, (ds, pa["S3"]["w1"], bk)
        # parity: pooled S1/S2 W1 must reproduce E1's pooled_all values
        for sname in ("S1", "S2"):
            e1p = e1["variants"]["pooled_all"][ds]["raw"][sname]
            assert abs(pa[sname]["w1"] - e1p) / e1p < 1e-6, \
                (ds, sname, pa[sname]["w1"], e1p)
        pooled_all[ds] = pa
        print(f"[e1b] {ds}: per_sensor_S3={s3_cell} pooled_S1={pa['S1']}",
              flush=True)

    out["variants"] = {"per_sensor_s3": per_sensor_s3,
                       "pooled_all": pooled_all}
    out["job"] = os.environ.get("SLURM_JOB_ID")
    (SH / "e1b_unified_rulers.json").write_text(json.dumps(out, indent=1))
    print(f"[e1b] wrote {SH/'e1b_unified_rulers.json'}", flush=True)


if __name__ == "__main__":
    main()
