#!/usr/bin/env python3
"""T3 — Kendall-τ ranking stability per shift × dataset (plan §5.2) + P1–P3.

For every model with banked full-test per-window predictions, compute
per-sensor MAE (mask_zero, same _ZERO_TOL convention as eval.metrics) on the
in-distribution and shifted slices of the STANDARD test split, rank models in
each setting, and compute τ with bootstrap CI via shift.ranking (W2-verified,
unit-tested). S1/S2 slice windows (same sensor axis → ranking_stability
as-is); S3 slices sensors (disjoint axes → point τ via ranking.py primitives
+ an independent-resample bootstrap, documented in the output).

Model set: HA, ARIMA, DCRNN, GWNet, STAEformer (LA/BAY only), Chronos-2,
TimesFM. Trained models = T1 standard-split checkpoints (this measures
leaderboard reordering under shift; the T2 retrained variants are a separate
question answered in t2_table.md).

Alignment: every model's preds must be (n_test, 12, N) on the frozen test
anchors; models that dump y_true are asserted against the raw-reconstructed
truth (catches windowing drift loudly).

P-verdicts (pre-registered, plan §7):
  P1: τ(S1) < τ(S2) per dataset.
  P2: on S3, graph models (DCRNN, GWNet) degrade more than per-sensor /
      sequence models (HA, ARIMA, Chronos-2, TimesFM). CAVEAT recorded: for
      standard-trained models the S3 re-slice is a hardness-transfer proxy
      (they saw all sensors at train time); it is exact for zero-shot TSFMs.
  P3: STAEformer holds rank 1–2 under S2 but drops ≥2 positions under S1
      (shifted-setting rankings, LA/BAY).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_loaders.windows import frozen_anchors, load_raw  # noqa: E402
from eval.metrics import _ZERO_TOL, mae as metrics_mae  # noqa: E402
from scripts.verification.frozen_degradation_matrix import truth  # noqa: E402
from shift.ranking import (_aggregate_rank, kendall_tau,  # noqa: E402
                           ranking_stability)
from shift.splits import (classify_targets, is_offpeak, is_peak,  # noqa: E402
                          is_weekday, is_weekend, s3_split)

B = REPO / "results/baselines"
SEED, N_BOOT = 2023, 1000
GRAPH_MODELS = ("dcrnn", "gwnet")
NONGRAPH_MODELS = ("ha", "arima", "chronos2", "timesfm")

# SD trained baselines: T1 eval_dump.pt (preds+labels), banked 2026-06-10.
SD_DUMPS = {
    "dcrnn": B / "dcrnn/largest_sd/run_46133071/experiments/dcrnn/SD/eval_dump.pt",
    "gwnet": B / "gwnet/largest_sd/run_46132703/experiments/gwnet/SD-doubletransition-1/eval_dump.pt",
}

MODELS = {
    "metr_la": ["ha", "arima", "dcrnn", "gwnet", "staeformer", "chronos2", "timesfm"],
    "pems_bay": ["ha", "arima", "dcrnn", "gwnet", "staeformer", "chronos2", "timesfm"],
    "largest_sd": ["ha", "arima", "dcrnn", "gwnet", "chronos2", "timesfm"],
}


def per_sensor_mae(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """(W, H, N) -> (N,) MAE per sensor, mask_zero convention of eval.metrics."""
    m = np.isfinite(yt) & (np.abs(yt) > _ZERO_TOL)
    err = np.where(m, np.abs(yt - yp), 0.0)
    cnt = m.sum(axis=(0, 1))
    # float64 accumulation: float32 pairwise-sum order differences vs
    # eval.metrics otherwise show up at ~1e-5 on MAE ~3.6.
    # Sensors with zero valid targets in the slice (SD has all-zero-flow
    # sensors) -> NaN; ranking.py aggregates with nanmean, so dead sensors
    # drop out of ranks and bootstrap identically for every model.
    out = err.sum(axis=(0, 1), dtype=np.float64) / np.maximum(cnt, 1)
    return np.where(cnt > 0, out, np.nan)


def load_preds(model: str, ds: str, n_test: int, N: int,
               yt_ref: np.ndarray) -> np.ndarray:
    if ds == "largest_sd" and model in SD_DUMPS:
        d = torch.load(SD_DUMPS[model], map_location="cpu")
        yp = d["preds"].numpy().astype(np.float32)
        yl = d["labels"].numpy().astype(np.float32)
    else:
        yp = np.load(B / model / ds / "y_pred.npy").astype(np.float32)
        yt_path = B / model / ds / "y_true.npy"
        yl = np.load(yt_path).astype(np.float32) if yt_path.exists() else None
    if yp.shape[0] > n_test:
        # DCRNN's upstream test loader pads the final batch to a multiple of
        # batch_size (LA: 6912 vs 6850). Order is sequential; truncate the
        # padding. The y_true-vs-raw-truth assert in main PROVES the first
        # n_test rows are the real windows. (The LargeST loader instead
        # DROPS the last partial batch — 6976 = 109x64 < 7004 — handled in
        # main by restricting every model to the common prefix.)
        assert yl is not None, f"{model}/{ds}: padded preds without y_true"
        yp, yl = yp[:n_test], yl[:n_test]
    assert yp.shape[1:] == (12, N), f"{model}/{ds}: {yp.shape}"
    assert yp.shape[0] <= n_test, f"{model}/{ds}: {yp.shape[0]} > {n_test}"
    return yp, yl


def s3_tau(per_model_mae_all: dict[str, np.ndarray], sp) -> dict:
    """Point τ + CI for the sensor-disjoint S3 setting.

    Uses ranking.py primitives; bootstrap resamples the two disjoint sensor
    sets INDEPENDENTLY (ranking_stability requires a shared axis, which S3
    does not have by construction)."""
    models = list(per_model_mae_all)
    A = np.stack([per_model_mae_all[m][sp.train_sensors] for m in models])
    Bm = np.stack([per_model_mae_all[m][sp.test_sensors] for m in models])
    tau_point = kendall_tau(_aggregate_rank(A), _aggregate_rank(Bm))
    rng = np.random.default_rng(SEED)
    taus = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ia = rng.integers(0, A.shape[1], size=A.shape[1])
        ib = rng.integers(0, Bm.shape[1], size=Bm.shape[1])
        taus[i] = kendall_tau(_aggregate_rank(A[:, ia]),
                              _aggregate_rank(Bm[:, ib]))
    degr = {m: float(np.nanmean(per_model_mae_all[m][sp.test_sensors])
                     / np.nanmean(per_model_mae_all[m][sp.train_sensors]) - 1)
            for m in models}
    return {
        "tau": tau_point,
        "ci_low": float(np.percentile(taus, 2.5)),
        "ci_high": float(np.percentile(taus, 97.5)),
        "n_models": len(models), "n_boot": N_BOOT, "seed": SEED,
        "n_sensors_id": int(len(sp.train_sensors)),
        "n_sensors_shift": int(len(sp.test_sensors)),
        "models_ranked_id": [models[i] for i in np.argsort([np.nanmean(
            per_model_mae_all[m][sp.train_sensors]) for m in models])],
        "models_ranked_shift": [models[i] for i in np.argsort([np.nanmean(
            per_model_mae_all[m][sp.test_sensors]) for m in models])],
        "rel_degradation_per_model": degr,
        "bootstrap_note": "disjoint sensor sets resampled independently "
                          "(ranking_stability requires a shared axis)",
    }


def main():
    out = {"spec": "paper_plan §5.2 T3; predictions = T1 standard-split "
                   "models re-sliced on the frozen test anchors",
           "mask": f"mask_zero, _ZERO_TOL={_ZERO_TOL}",
           "datasets": {}}
    for ds, models in MODELS.items():
        print(f"=== {ds} ===", flush=True)
        yt = truth(ds).astype(np.float32)          # (n_test, 12, N)
        n_test, _, N = yt.shape
        anc = frozen_anchors(ds)["test"]
        assert len(anc) == n_test
        _, ts, _ = load_raw(ds)

        loaded = {m: load_preds(m, ds, n_test, N, yt) for m in models}
        # common prefix across models (LargeST loader drop_last; sequential
        # order proven per-model by the y_true assert below)
        n_use = min(min(yp.shape[0] for yp, _ in loaded.values()), n_test)
        if n_use < n_test:
            print(f"  [note] common window prefix {n_use}/{n_test} "
                  f"(drop_last loader)", flush=True)
        yt, anc = yt[:n_use], anc[:n_use]
        preds = {}
        for m, (yp, yl) in loaded.items():
            if yl is not None:
                d = float(np.abs(yl[:n_use] - yt).mean())
                assert d < 0.05, f"{m}/{ds}: y_true drift vs raw truth = {d}"
            preds[m] = yp[:n_use]
        # parity anchor: vectorized per-sensor MAE == eval.metrics.mae (s0)
        m0 = models[0]
        ps = per_sensor_mae(yt, preds[m0])
        j0 = int(np.flatnonzero(np.isfinite(ps))[0])
        ref = metrics_mae(yt[:, :, j0].astype(np.float64),
                          preds[m0][:, :, j0].astype(np.float64),
                          mask_zero=True)
        assert abs(ps[j0] - ref) < 1e-9, (ps[j0], ref)
        n_dead = int(np.isnan(ps).sum())
        if n_dead:
            print(f"  [note] {n_dead} dead sensors (zero valid targets) -> "
                  f"NaN, excluded via nanmean", flush=True)

        block = {"n_dead_sensors": n_dead, "n_windows_used": int(len(anc))}
        # ---- S1 / S2: window slices, shared sensor axis ----
        for shift, base_p, shift_p in (("S1", is_weekday, is_weekend),
                                       ("S2", is_peak, is_offpeak)):
            bm = classify_targets(anc, np.asarray(base_p(ts)))
            sm = classify_targets(anc, np.asarray(shift_p(ts)))
            id_mae = {m: per_sensor_mae(yt[bm], preds[m][bm]) for m in models}
            sh_mae = {m: per_sensor_mae(yt[sm], preds[m][sm]) for m in models}
            r = ranking_stability(id_mae, sh_mae, n_boot=N_BOOT, seed=SEED)
            r["n_windows_id"], r["n_windows_shift"] = int(bm.sum()), int(sm.sum())
            block[shift] = r
            print(f"  {shift}: tau={r['tau']:.3f} "
                  f"[{r['ci_low']:.3f},{r['ci_high']:.3f}] "
                  f"shift-rank: {r['models_ranked_shift']}", flush=True)

        # ---- S3: sensor slice ----
        sp = s3_split(ds)
        all_mae = {m: per_sensor_mae(yt, preds[m]) for m in models}
        block["S3"] = s3_tau(all_mae, sp)
        print(f"  S3: tau={block['S3']['tau']:.3f} "
              f"degr={ {k: round(v, 3) for k, v in block['S3']['rel_degradation_per_model'].items()} }",
              flush=True)
        out["datasets"][ds] = block

    # ---- pre-registered verdicts ----
    p1 = {ds: bool(b["S1"]["tau"] < b["S2"]["tau"])
          for ds, b in out["datasets"].items()}
    p2 = {}
    for ds, b in out["datasets"].items():
        d = b["S3"]["rel_degradation_per_model"]
        g = float(np.mean([d[m] for m in GRAPH_MODELS if m in d]))
        ng = float(np.mean([d[m] for m in NONGRAPH_MODELS if m in d]))
        p2[ds] = {"graph_mean_degr": round(g, 4),
                  "nongraph_mean_degr": round(ng, 4),
                  "graph_degrades_more": bool(g > ng)}
    p3 = {}
    for ds in ("metr_la", "pems_bay"):
        b = out["datasets"][ds]
        r2 = b["S2"]["models_ranked_shift"].index("staeformer") + 1
        r1 = b["S1"]["models_ranked_shift"].index("staeformer") + 1
        p3[ds] = {"rank_S2_shifted": r2, "rank_S1_shifted": r1,
                  "holds": bool(r2 <= 2 and (r1 - r2) >= 2)}
    out["verdicts"] = {
        "P1_tau_S1_lt_tau_S2": p1,
        "P2_graph_degrades_more_on_S3": p2,
        "P2_caveat": "S3 re-slice is exact for zero-shot TSFMs; a hardness-"
                     "transfer proxy for standard-trained models (they saw "
                     "all sensors at train time). Recorded before reading "
                     "results.",
        "P3_staeformer_S2_top2_S1_drop2": p3,
    }
    out["host"] = os.uname().nodename
    o = REPO / "results/shift/t3_rankings.json"
    o.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["verdicts"], indent=2))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
