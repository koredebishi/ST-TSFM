#!/usr/bin/env python3
"""Build the LargeST-SD training sets for the weekend retraining runs (Table V).

Same layout as build_s1_pools.build_sd (his.npz link plus filtered
idx_{train,val,test}.npy). Every run is tested on the S1 weekend pool, so
its test MAE is the weekend-pool MAE.

Training sets:
  weekend        all weekend-target train/val windows
  std_match      random subset of the standard windows, same size as the
                 weekday pool (14717 / 4873)
  weekday_match  random subset of the weekday pool, same size as the
                 weekend pool

Checks: every kept window's targets are on the right day type, the
weekday pool matches splits_summary.json, and subset sizes are exact.
Output: datasets/shift_pools/d39/<set>/sd/2019/ and a summary JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data_loaders.windows import frozen_anchors, load_raw  # noqa: E402
from shift.splits import classify_targets, is_weekday, is_weekend  # noqa: E402

OUT = REPO / "datasets/shift_pools/d39"
SRC = REPO / "datasets/LargeST-SD/2019"
DRAW_SEED = 2023
WEEKDAY_TRAIN_N, WEEKDAY_VAL_N = 14717, 4873  # splits_summary parity


def pool(anchors, ts, pred):
    m = classify_targets(np.asarray(anchors, dtype=int), np.asarray(pred(ts)))
    return np.asarray(anchors, dtype=int)[m]


def subset(a, n, rng):
    assert len(a) >= n, (len(a), n)
    return np.sort(rng.choice(a, size=n, replace=False))


def write_arm(name, idx_train, idx_val, idx_test):
    root = OUT / name / "sd" / "2019"
    root.mkdir(parents=True, exist_ok=True)
    link = root / "his.npz"
    if not link.exists():
        link.symlink_to(SRC / "his.npz")
    # LargeST main.py resolves the adjacency relative to the dataroot
    # (data/sd/sd_rn_adj.npy), same layout as shift_pools/s1/sd_dataroot
    adj = root.parent / "sd_rn_adj.npy"
    if not adj.exists():
        adj.symlink_to(SRC.parent / "sd_rn_adj.npy")
    np.save(root / "idx_train.npy", idx_train.astype(np.int64))
    np.save(root / "idx_val.npy", idx_val.astype(np.int64))
    np.save(root / "idx_test.npy", idx_test.astype(np.int64))
    print(f"[d39] {name}: train={len(idx_train)} val={len(idx_val)} "
          f"test={len(idx_test)}", flush=True)
    return {"train": int(len(idx_train)), "val": int(len(idx_val)),
            "test": int(len(idx_test))}


def main():
    _, ts, _ = load_raw("largest_sd")
    anc = frozen_anchors("largest_sd")
    rng = np.random.default_rng(DRAW_SEED)

    wd_train = pool(anc["train"], ts, is_weekday)
    wd_val = pool(anc["val"], ts, is_weekday)
    we_train = pool(anc["train"], ts, is_weekend)
    we_val = pool(anc["val"], ts, is_weekend)
    we_test = pool(anc["test"], ts, is_weekend)
    assert len(wd_train) == WEEKDAY_TRAIN_N, len(wd_train)
    assert len(wd_val) == WEEKDAY_VAL_N, len(wd_val)

    summary = {"spec": "decisions.md D39", "draw_seed": DRAW_SEED,
               "weekday_pool_parity": [len(wd_train), len(wd_val)],
               "arms": {}}
    summary["arms"]["weekend"] = write_arm(
        "weekend", we_train, we_val, we_test)
    summary["arms"]["std_match"] = write_arm(
        "std_match",
        subset(np.asarray(anc["train"], dtype=int), WEEKDAY_TRAIN_N, rng),
        subset(np.asarray(anc["val"], dtype=int), WEEKDAY_VAL_N, rng),
        we_test)
    summary["arms"]["weekday_match"] = write_arm(
        "weekday_match",
        subset(wd_train, len(we_train), rng),
        subset(wd_val, len(we_val), rng),
        we_test)

    # purity: every kept window's targets satisfy its arm's predicate
    for name, idx, pred in (("weekend", summary and np.load(
            OUT / "weekend/sd/2019/idx_train.npy"), is_weekend),
            ("weekday_match", np.load(
             OUT / "weekday_match/sd/2019/idx_train.npy"), is_weekday)):
        m = classify_targets(idx, np.asarray(pred(ts)))
        assert m.all(), f"purity violation in {name}"
    print("[d39] purity OK", flush=True)

    (OUT / "d39_pools_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[d39] wrote {OUT/'d39_pools_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
