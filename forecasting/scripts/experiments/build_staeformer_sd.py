#!/usr/bin/env python3
"""Build LargeST-SD in the data format of the STAEformer code.

Writes datasets/LargeST-SD/staeformer_fmt/{data.npz,index.npz}:
- data.npz: (T, N, 3) float64. Channel 0 is raw flow (his.npz values
  un-normalized with the stored mean/std), channel 1 is time of day
  (96 steps/day), channel 2 is day of week.
- index.npz: train/val/test triplets [a-11, a+1, a+13] from the fixed
  anchors, so every window matches the other models' windows.

Check: the test targets read through index.npz must equal
results/baselines/ha/largest_sd/y_true.npy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data_loaders.windows import SEQ_IN, SEQ_OUT  # noqa: E402

DATA = REPO / "datasets/LargeST-SD/2019"
OUT = REPO / "datasets/LargeST-SD/staeformer_fmt"


def main():
    npz = np.load(DATA / "his.npz")
    flow = (npz["data"][..., 0].astype(np.float64) * float(npz["std"])
            + float(npz["mean"]))
    T, N = flow.shape
    ts = pd.date_range("2019-01-01", periods=T, freq="15min")
    tod = ((ts.hour * 60 + ts.minute) / (24 * 60)).to_numpy()
    dow = ts.dayofweek.to_numpy().astype(np.float64)

    data = np.stack(
        [flow,
         np.broadcast_to(tod[:, None], (T, N)),
         np.broadcast_to(dow[:, None], (T, N))], axis=-1)

    idx = {}
    for split in ("train", "val", "test"):
        a = np.load(DATA / f"idx_{split}.npy").astype(np.int64)
        idx[split] = np.stack(
            [a - SEQ_IN + 1, a + 1, a + 1 + SEQ_OUT], axis=1)
        assert idx[split][:, 0].min() >= 0
        assert idx[split][:, 2].max() <= T

    banked = np.load(REPO / "results/baselines/ha/largest_sd/y_true.npy")
    test = idx["test"]
    assert len(test) == len(banked), (len(test), len(banked))
    for k in (0, len(test) // 2, len(test) - 1):
        s, m, e = test[k]
        assert np.allclose(data[m:e, :, 0], banked[k], atol=1e-3), \
            f"target mismatch at test window {k}"
    print(f"[build] parity OK: {len(test)} test windows match banked truth",
          flush=True)

    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "data.npz", data=data)
    np.savez_compressed(OUT / "index.npz", **idx)
    print(f"[build] wrote {OUT}: data {data.shape}, "
          f"index {[f'{k}:{v.shape}' for k, v in idx.items()]}", flush=True)


if __name__ == "__main__":
    main()
