#!/usr/bin/env python3
"""D11 exhaustive remedy: sweep k-means k for BAY S3 holdout candidates.

Pre-stated decision rule (written before results, per governance):
- Candidate = any cluster holding 10-30% of sensors (sane holdout size).
- If any candidate's pooled-W1 severity > tripwire (0.5 x S1 = 2.117):
  the highest-severity PASSING candidate with the smallest k becomes the
  BAY S3 holdout (k disclosed per dataset; LA/SD stay at k=5, which pass).
- If no candidate passes across the whole sweep: option B holds, with this
  table as the demonstration that BAY's max achievable spatial severity
  is below threshold at every probed granularity.

Severity is model-free (raw data only; no model outcome involved), so this
selection is stratification under a pre-registered rule, not tuning.
Output: results/shift/bay_s3_sweep.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_loaders.windows import frozen_anchors, load_raw, sensor_locations  # noqa: E402
from shift.severity import _clean, _target_rows, w1  # noqa: E402

S1_SEV = 4.234          # job 46164821, severity_summary.json
TRIPWIRE = 0.5 * S1_SEV
KS = [5, 8, 10, 15, 20]
SIZE_LO, SIZE_HI = 0.10, 0.30


def main():
    from sklearn.cluster import KMeans

    ds = "pems_bay"
    raw, _, _ = load_raw(ds)
    coords = sensor_locations(ds)
    n = len(coords)
    rows = _target_rows(frozen_anchors(ds)["test"])

    results = []
    for k in KS:
        km = KMeans(n_clusters=k, random_state=2023, n_init=10).fit(coords)
        for c in range(k):
            in_c = np.where(km.labels_ == c)[0]
            frac = len(in_c) / n
            out_c = np.where(km.labels_ != c)[0]
            a = _clean(raw[np.ix_(rows, out_c)].ravel())
            b = _clean(raw[np.ix_(rows, in_c)].ravel())
            sev = w1(a, b)
            tight = float(np.linalg.norm(
                coords[in_c] - km.cluster_centers_[c], axis=1).mean())
            results.append({
                "k": k, "cluster": c, "n_sensors": int(len(in_c)),
                "fraction": round(frac, 4),
                "size_ok": bool(SIZE_LO <= frac <= SIZE_HI),
                "tightness_deg": round(tight, 6),
                "severity_w1": round(sev, 4),
                "passes": bool(sev > TRIPWIRE
                               and SIZE_LO <= frac <= SIZE_HI),
            })
            print(f"[sweep] k={k} c={c}: n={len(in_c)} ({frac:.0%}) "
                  f"W1={sev:.3f} {'PASS' if results[-1]['passes'] else ''}",
                  flush=True)

    candidates = [r for r in results if r["size_ok"]]
    passing = [r for r in results if r["passes"]]
    best = max(candidates, key=lambda r: r["severity_w1"]) if candidates else None
    verdict = ("PASSING candidate exists -> use highest-severity passing "
               "cluster (smallest k tiebreak)" if passing else
               "NO candidate passes at any k -> option B holds, demonstrated")
    out = {
        "host": os.uname().nodename,
        "s1_severity": S1_SEV, "tripwire": round(TRIPWIRE, 4),
        "ks_swept": KS, "size_window": [SIZE_LO, SIZE_HI],
        "n_candidates_size_ok": len(candidates),
        "n_passing": len(passing),
        "best_size_ok_candidate": best,
        "passing": sorted(passing, key=lambda r: (-r["severity_w1"], r["k"])),
        "verdict": verdict,
        "all_results": results,
    }
    (REPO / "results/shift/bay_s3_sweep.json").write_text(
        json.dumps(out, indent=2))
    print(f"[sweep] verdict: {verdict}")
    print(f"[sweep] best size-ok candidate: {best}")


if __name__ == "__main__":
    main()
