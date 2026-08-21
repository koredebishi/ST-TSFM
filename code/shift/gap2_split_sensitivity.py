#!/usr/bin/env python3
"""GAP-2 — split-sensitivity ablations (paper_plan FINAL PAPER SCOPE).

Do the S2 and S3 findings survive the framework's own knobs?
- S2: vary the peak-window definition around D8's [07,10)∪[16,19).
- S3: vary k-means k around D8's k=5 (seed fixed at 2023).

Pure re-slicing of banked predictions — reuses t3_rankings' verified
loaders (alignment-proven), per_sensor_mae (parity-anchored), s3_tau, and
shift.ranking.ranking_stability (W2-verified). No GPU, no new predictions.

Reported per variant × dataset: frozen-TSFM rel degradation (Chronos-2,
TimesFM; nanmean-pooled per-sensor MAE, shifted/id − 1) and Kendall τ of
the full-model ranking. Findings to check: S2 degradation stays negative
everywhere; τ stays high everywhere except SD-S1 (S1 has no definitional
knob and is not varied here).
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
from scripts.verification.frozen_degradation_matrix import truth  # noqa: E402
from shift.ranking import ranking_stability  # noqa: E402
from shift.splits import classify_targets, s3_split  # noqa: E402
from shift.t3_rankings import (MODELS, load_preds, per_sensor_mae,  # noqa: E402
                               s3_tau)

SEED, N_BOOT = 2023, 1000
TSFMS = ("chronos2", "timesfm")

# S2 peak-window variants (base = D8, shift/splits.py is_peak)
PEAK_VARIANTS = {
    "base_[7,10)+[16,19)": lambda h: ((h >= 7) & (h < 10)) | ((h >= 16) & (h < 19)),
    "wide_[6,11)+[15,20)": lambda h: ((h >= 6) & (h < 11)) | ((h >= 15) & (h < 20)),
    "narrow_[7,9)+[16,18)": lambda h: ((h >= 7) & (h < 9)) | ((h >= 16) & (h < 18)),
    "shifted_[8,11)+[17,20)": lambda h: ((h >= 8) & (h < 11)) | ((h >= 17) & (h < 20)),
}
S3_KS = (4, 5, 6, 8)  # base = 5 (D8)


def pooled(ps: np.ndarray) -> float:
    return float(np.nanmean(ps))


def main():
    out = {"spec": "paper_plan FINAL PAPER SCOPE GAP-2; base defs = D8",
           "S2_peak_variants": {}, "S3_k_variants": {},
           "host": os.uname().nodename}

    data = {}
    for ds, models in MODELS.items():
        yt = truth(ds).astype(np.float32)
        n_test, _, N = yt.shape
        anc = frozen_anchors(ds)["test"]
        _, ts, _ = load_raw(ds)
        loaded = {m: load_preds(m, ds, n_test, N, yt) for m in models}
        n_use = min(min(yp.shape[0] for yp, _ in loaded.values()), n_test)
        yt, anc = yt[:n_use], anc[:n_use]
        preds = {}
        for m, (yp, yl) in loaded.items():
            if yl is not None:
                d = float(np.abs(yl[:n_use] - yt).mean())
                assert d < 0.05, f"{m}/{ds}: y_true drift {d}"
            preds[m] = yp[:n_use]
        data[ds] = (yt, anc, ts, preds, models)
        print(f"[gap2] {ds}: {n_use} windows, {len(models)} models", flush=True)

    # ---- S2 peak-definition sensitivity ----
    for vname, peak_fn in PEAK_VARIANTS.items():
        out["S2_peak_variants"][vname] = {}
        for ds, (yt, anc, ts, preds, models) in data.items():
            h = np.asarray(ts.hour)
            pk = peak_fn(h)
            bm = classify_targets(anc, pk)
            sm = classify_targets(anc, ~pk)
            if bm.sum() == 0 or sm.sum() == 0:
                out["S2_peak_variants"][vname][ds] = {"error": "empty slice"}
                continue
            id_mae = {m: per_sensor_mae(yt[bm], preds[m][bm]) for m in models}
            sh_mae = {m: per_sensor_mae(yt[sm], preds[m][sm]) for m in models}
            r = ranking_stability(id_mae, sh_mae, n_boot=N_BOOT, seed=SEED)
            degr = {m: round(pooled(sh_mae[m]) / pooled(id_mae[m]) - 1, 4)
                    for m in TSFMS}
            out["S2_peak_variants"][vname][ds] = {
                "tau": round(r["tau"], 3),
                "ci": [round(r["ci_low"], 3), round(r["ci_high"], 3)],
                "tsfm_rel_degradation": degr,
                "n_id": int(bm.sum()), "n_shift": int(sm.sum())}
            print(f"[gap2] S2 {vname} {ds}: tau={r['tau']:.3f} degr={degr}",
                  flush=True)

    # ---- S3 k sensitivity ----
    for k in S3_KS:
        out["S3_k_variants"][f"k={k}"] = {}
        for ds, (yt, anc, ts, preds, models) in data.items():
            try:
                sp = s3_split(ds, k=k)
            except Exception as e:  # degenerate clustering for this k
                out["S3_k_variants"][f"k={k}"][ds] = {"error": str(e)[:120]}
                continue
            all_mae = {m: per_sensor_mae(yt, preds[m]) for m in models}
            r = s3_tau(all_mae, sp)
            degr = {m: round(r["rel_degradation_per_model"][m], 4)
                    for m in TSFMS}
            out["S3_k_variants"][f"k={k}"][ds] = {
                "tau": round(r["tau"], 3),
                "ci": [round(r["ci_low"], 3), round(r["ci_high"], 3)],
                "tsfm_rel_degradation": degr,
                "n_heldout_sensors": r["n_sensors_shift"]}
            print(f"[gap2] S3 k={k} {ds}: tau={r['tau']:.3f} degr={degr} "
                  f"held-out={r['n_sensors_shift']}", flush=True)

    o = REPO / "results/shift/gap2_split_sensitivity.json"
    o.write_text(json.dumps(out, indent=2))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
