#!/usr/bin/env python3
"""Build the S1 (weekday to weekend) training pools for the retrained models.

Writes, under datasets/shift_pools/s1/:
- sd_dataroot/sd/2019/: his.npz (link) and filtered idx_{train,val,test}.npy,
  the layout the LargeST data loader reads.
- {metr_la,pems_bay}/: filtered {train,val,test}.npz in the DCRNN /
  Graph WaveNet format.

Train and val keep weekday-target windows; test keeps weekend-target
windows (shift/splits.s1_split on the fixed anchors).

Checks before anything is written:
1. counts match results/shift/splits_summary.json;
2. every kept target row has the right day type;
3. METR-LA / PEMS-BAY: filtered y equals the raw slice at each anchor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data_loaders.windows import SEQ_IN, SEQ_OUT, frozen_anchors, load_raw  # noqa: E402
from shift.splits import (classify_targets, is_offpeak, is_peak,  # noqa: E402
                          is_weekday, is_weekend, s1_split, s2_split)

SHIFT = sys.argv[1] if len(sys.argv) > 1 else "s1"
CFG = {"s1": (s1_split, is_weekday, is_weekend, "S1"),
       "s2": (s2_split, is_peak, is_offpeak, "S2")}
SPLIT_FN, TRAIN_PRED, TEST_PRED, SUMKEY = CFG[SHIFT]
OUT = REPO / f"datasets/shift_pools/{SHIFT}"
SUMMARY = json.loads((REPO / "results/shift/splits_summary.json").read_text())


def check_counts(ds, sp):
    ref = SUMMARY["summaries"][ds][SUMKEY]
    assert len(sp.train_anchors) == ref["n_train"], (ds, "train", len(sp.train_anchors), ref["n_train"])
    assert len(sp.val_anchors) == ref["n_val"]
    assert len(sp.test_anchors) == ref["n_test"]


def check_purity(ds, sp, ts):
    tr, te = np.asarray(TRAIN_PRED(ts)), np.asarray(TEST_PRED(ts))
    assert classify_targets(sp.train_anchors, tr).all(), f"{ds}: train pool impure"
    assert classify_targets(sp.val_anchors, tr).all(), f"{ds}: val pool impure"
    assert classify_targets(sp.test_anchors, te).all(), f"{ds}: test pool impure"


def build_sd():
    ds = "largest_sd"
    sp = SPLIT_FN(ds)
    _, ts, _ = load_raw(ds)
    check_counts(ds, sp)
    check_purity(ds, sp, ts)
    root = OUT / "sd_dataroot/sd/2019"
    root.mkdir(parents=True, exist_ok=True)
    src = REPO / "datasets/LargeST-SD/2019/his.npz"
    link = root / "his.npz"
    if not link.exists():
        link.symlink_to(src)
    np.save(root / "idx_train.npy", sp.train_anchors.astype(np.int64))
    np.save(root / "idx_val.npy", sp.val_anchors.astype(np.int64))
    np.save(root / "idx_test.npy", sp.test_anchors.astype(np.int64))
    print(f"[s1-pools] SD: {len(sp.train_anchors)}/{len(sp.val_anchors)}/"
          f"{len(sp.test_anchors)} -> {root}", flush=True)


def build_la_bay(ds: str, proc: Path):
    sp = SPLIT_FN(ds)
    raw, ts, _ = load_raw(ds)
    check_counts(ds, sp)
    check_purity(ds, sp, ts)
    pools = frozen_anchors(ds)
    first = SEQ_IN - 1
    n_train, n_val = len(pools["train"]), len(pools["val"])
    offs = {"train": first, "val": first + n_train,
            "test": first + n_train + n_val}
    kept = {"train": sp.train_anchors, "val": sp.val_anchors,
            "test": sp.test_anchors}
    dst = OUT / ds
    dst.mkdir(parents=True, exist_ok=True)
    if all((dst / f"{c}.npz").exists() for c in ("train", "val", "test")):
        # already built (asserted at build time); do NOT rewrite — a training
        # job may be reading these files right now (race safety)
        print(f"[s1-pools] {ds}: exists, skipping", flush=True)
        return
    for cat in ("train", "val", "test"):
        z = np.load(proc / f"{cat}.npz")
        pos = kept[cat] - offs[cat]
        assert pos.min() >= 0 and pos.max() < len(z["x"]), f"{ds}/{cat}: pos out of range"
        out = {k: (z[k][pos] if z[k].shape[:1] == z["x"].shape[:1] else z[k])
               for k in z.files}
        # alignment assert: filtered y ch0 equals raw slice at kept anchors
        for j in (0, len(pos) // 2, len(pos) - 1):
            a = kept[cat][j]
            assert np.allclose(out["y"][j, ..., 0],
                               raw[a + 1: a + 1 + SEQ_OUT], atol=1e-3), \
                f"{ds}/{cat}: misalignment at kept window {j}"
        np.savez(dst / f"{cat}.npz", **out)
        print(f"[s1-pools] {ds}/{cat}: {len(pos)} windows -> {dst}/{cat}.npz",
              flush=True)


def main():
    build_sd()
    build_la_bay("metr_la", REPO / "datasets/METR-LA/processed")
    build_la_bay("pems_bay", REPO / "datasets/PEMS-BAY/processed")
    print("[s1-pools] ALL ASSERTS PASSED")


if __name__ == "__main__":
    main()
