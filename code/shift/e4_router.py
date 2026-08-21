#!/usr/bin/env python3
"""D37 E4: timestamp router on LargeST-SD, leak-free selection.

Selection rule (fixed in decisions.md D37 BEFORE any test result was
seen): for each regime (weekday, weekend), pick the candidate with the
best masked MAE on that regime's VALIDATION windows. Candidates: HA,
Chronos-2, TimesFM. HA validation predictions come from its train-only
time-of-week lookup (reused from baselines/historical_average/run.py);
frozen-model validation predictions are one new inference pass (this
job). Test evaluation then uses ONLY the banked test arrays: each test
window is served by the selected model for its regime (a window whose
targets are not all-weekend routes to the weekday model, disclosed).

Outputs: results/shift/e4_val_preds/{model}.npy and
results/shift/e4_router.json with parity asserts against tbig_grid.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from baselines.historical_average.run import (build_ha_lookup,  # noqa: E402
                                              time_of_week_key)
from baselines.tsfm_zeroshot_eval import (CONTEXT_LEN, SEQ_OUT,  # noqa: E402
                                          load_chronos2, load_dataset,
                                          load_timesfm)
from data_loaders.windows import frozen_anchors, load_raw  # noqa: E402
from eval.metrics import mae as metrics_mae  # noqa: E402
from shift.splits import classify_targets, is_weekday, is_weekend  # noqa: E402
from shift.t3_rankings import load_preds  # noqa: E402
from scripts.verification.frozen_degradation_matrix import truth  # noqa: E402

DS = "largest_sd"
SH = REPO / "results/shift"
VP = SH / "e4_val_preds"
CAND = ("ha", "chronos2", "timesfm")


def pool_mae(yt, yp):
    return float(metrics_mae(yt.astype(np.float64), yp.astype(np.float64),
                             mask_zero=True))


def ha_predict(raw, ts, anchors, idx_train):
    """HA forecasts for the given anchors, replicating run_largest_sd
    step by step: train_steps = every timestep any training window
    covers; time-of-week lookup; the dead-bucket fallback block copied
    verbatim from baselines/historical_average/run.py."""
    key = time_of_week_key(ts)
    from data_loaders.windows import SEQ_IN
    train_steps = np.unique(np.concatenate(
        [np.arange(i - SEQ_IN + 1, i + 1 + SEQ_OUT) for i in
         np.asarray(idx_train, dtype=int)]))
    lookup = build_ha_lookup(raw[train_steps].astype(np.float64),
                             key[train_steps])
    N = raw.shape[1]
    out = np.zeros((len(anchors), SEQ_OUT, N))
    for j, a in enumerate(np.asarray(anchors, dtype=int)):
        out[j] = lookup[key[a + 1: a + 1 + SEQ_OUT]]
    nan_mask = np.isnan(out)
    if nan_mask.any():
        train_vals = raw[train_steps].astype(np.float64)
        train_valid = np.abs(train_vals) > 1e-3
        per_sensor = np.where(
            train_valid.any(0),
            np.where(train_valid, train_vals, 0.0).sum(0)
            / np.maximum(train_valid.sum(0), 1),
            0.0,
        )
        out = np.where(nan_mask, np.broadcast_to(per_sensor, out.shape), out)
    return out.astype(np.float32)


def window_targets(raw, anchors):
    N = raw.shape[1]
    out = np.zeros((len(anchors), SEQ_OUT, N), dtype=np.float32)
    for j, a in enumerate(np.asarray(anchors, dtype=int)):
        out[j] = raw[a + 1: a + 1 + SEQ_OUT]
    return out


def infer_shard(model_name, ws, we):
    """One frozen-model shard over val_anchors[ws:we] -> partial file.

    Same shard pattern as the banked SD test run (tsfm_zeroshot_eval
    --window-start/--window-end); chunk 256 matches its convention."""
    raw, _, _, _ = load_dataset(DS)
    val_anchors = np.asarray(frozen_anchors(DS)["val"], dtype=int)
    we = min(we, len(val_anchors))
    part = VP / "partials"
    part.mkdir(parents=True, exist_ok=True)
    f = part / f"{model_name}_{ws}_{we}.npy"
    if f.exists():
        print(f"[e4] shard {f.name} already done", flush=True)
        return
    assert torch.cuda.is_available(), "No CUDA device visible."
    fn, meta = (load_chronos2() if model_name == "chronos2"
                else load_timesfm())
    print(f"[e4] shard {model_name} [{ws},{we}) meta={meta}", flush=True)
    N = raw.shape[1]
    yp = np.zeros((we - ws, SEQ_OUT, N), dtype=np.float32)
    for j, a in enumerate(val_anchors[ws:we]):
        ctx = raw[a - CONTEXT_LEN + 1: a + 1].T
        outs = [fn(np.ascontiguousarray(ctx[s:s + 256]))
                for s in range(0, N, 256)]
        yp[j] = np.concatenate(outs).T
        if j % 50 == 0:
            print(f"[e4] {model_name} shard {ws}: {j}/{we - ws}", flush=True)
    np.save(f, yp)
    print(f"[e4] wrote {f}", flush=True)


def merge_partials(model_name, n_val):
    part = VP / "partials"
    shards = sorted(part.glob(f"{model_name}_*.npy"))
    assert shards, f"no partials for {model_name}"
    out = None
    covered = np.zeros(n_val, dtype=bool)
    for p in shards:
        ws, we = map(int, p.stem.split("_")[1:3])
        a = np.load(p)
        if out is None:
            out = np.zeros((n_val,) + a.shape[1:], dtype=np.float32)
        out[ws:we] = a
        covered[ws:we] = True
    assert covered.all(), f"{model_name}: gaps in shard coverage"
    np.save(VP / f"{model_name}.npy", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfm-start", type=int, default=None)
    ap.add_argument("--tfm-end", type=int, default=None)
    args = ap.parse_args()
    VP.mkdir(exist_ok=True)
    if args.tfm_start is not None:
        infer_shard("timesfm", args.tfm_start, args.tfm_end)
        return

    raw, test_anchors, _, _ = load_dataset(DS)
    _, ts, _ = load_raw(DS)
    anc = frozen_anchors(DS)
    val_anchors = anc["val"]
    train_rows_end = int(np.asarray(anc["train"]).max()) + 1  # context rows only
    print(f"[e4] val windows={len(val_anchors)} test windows={len(test_anchors)}",
          flush=True)

    # ---------- stage 1: validation predictions ----------
    # HA generator parity anchor: rebuilt HA on the TEST anchors must
    # reproduce the banked HA test predictions before its validation
    # numbers are trusted.
    ha_test_rebuilt = ha_predict(raw, ts, anc["test"], anc["train"])
    banked_ha = np.load(
        REPO / "results/baselines/ha/largest_sd/y_pred.npy")
    d = float(np.abs(ha_test_rebuilt[:len(banked_ha)] - banked_ha).max())
    assert d < 1e-3, f"HA generator drift vs banked test preds: {d}"
    print(f"[e4] HA generator parity vs banked test preds: max|diff|={d:.2e}",
          flush=True)

    y_val = window_targets(raw, val_anchors)
    val_preds = {"ha": ha_predict(raw, ts, val_anchors, anc["train"])}
    for name in ("chronos2", "timesfm"):
        f = VP / f"{name}.npy"
        if f.exists():
            val_preds[name] = np.load(f)
            print(f"[e4] {name}: reusing cached val preds", flush=True)
        else:
            val_preds[name] = merge_partials(name, len(val_anchors))
            print(f"[e4] {name}: merged shards", flush=True)

    # ---------- stage 2: leak-free selection on validation ----------
    wd_mask = classify_targets(val_anchors, np.asarray(is_weekday(ts)))
    we_mask = classify_targets(val_anchors, np.asarray(is_weekend(ts)))
    print(f"[e4] val pools: weekday={int(wd_mask.sum())} "
          f"weekend={int(we_mask.sum())}", flush=True)
    sel, val_table = {}, {}
    for regime, mask in (("weekday", wd_mask), ("weekend", we_mask)):
        assert mask.sum() > 0, f"empty {regime} validation pool"
        maes = {m: pool_mae(y_val[mask], val_preds[m][mask]) for m in CAND}
        assert all(np.isfinite(v) for v in maes.values()), \
            f"non-finite validation MAE: {maes}"
        sel[regime] = min(maes, key=maes.get)
        val_table[regime] = maes
        print(f"[e4] val {regime}: {maes} -> {sel[regime]}", flush=True)

    # ---------- stage 3: routed evaluation on banked test arrays ----------
    # tbig's Table II values are pooled over the common window prefix of
    # ALL SD models (drop_last loaders); evaluate on exactly that prefix
    # so the router row is apples-to-apples with every monolithic row.
    from shift.t3_rankings import MODELS as ALL_MODELS
    yt = truth(DS).astype(np.float32)
    n_test, _, N = yt.shape
    loaded_all = {m: load_preds(m, DS, n_test, N, yt)
                  for m in ALL_MODELS[DS]}
    n_use = min(min(yp.shape[0] for yp, _ in loaded_all.values()), n_test)
    yt = yt[:n_use]
    t_anchors = np.asarray(anc["test"])[:n_use]
    preds = {m: loaded_all[m][0][:n_use] for m in CAND}

    te_we = classify_targets(t_anchors, np.asarray(is_weekend(ts)))
    routed = np.where(te_we[:, None, None],
                      preds[sel["weekend"]], preds[sel["weekday"]])

    tbig = json.loads((SH / "tbig_grid.json").read_text())
    banked = tbig["datasets"][DS]["models"]
    report = {}
    for m in CAND:
        full = pool_mae(yt, preds[m])
        assert abs(full - banked[m]["std"]) < 5e-3, (m, full, banked[m]["std"])
        report[m] = {"full_test": full,
                     "weekend_pool": pool_mae(yt[te_we], preds[m][te_we]),
                     "weekday_pool": pool_mae(yt[~te_we], preds[m][~te_we])}
    report["router"] = {
        "full_test": pool_mae(yt, routed),
        "weekend_pool": pool_mae(yt[te_we], routed[te_we]),
        "weekday_pool": pool_mae(yt[~te_we], routed[~te_we])}
    print(f"[e4] TEST: {json.dumps(report, indent=1)}", flush=True)

    out = {"spec": "decisions.md D37 E4",
           "selection_rule": "best masked MAE per regime on validation",
           "candidates": list(CAND), "selected": sel,
           "validation_mae": val_table,
           "n_val_windows": int(len(val_anchors)),
           "mixed_windows_note": "test windows whose targets are not "
                                 "all-weekend route to the weekday model",
           "test": report,
           "monolithic_banked_std": {m: banked[m]["std"] for m in
                                     ("ha", "arima", "dcrnn", "gwnet",
                                      "chronos2", "timesfm")},
           "job": os.environ.get("SLURM_JOB_ID")}
    (SH / "e4_router.json").write_text(json.dumps(out, indent=1))
    print(f"[e4] wrote {SH/'e4_router.json'}", flush=True)


if __name__ == "__main__":
    main()
