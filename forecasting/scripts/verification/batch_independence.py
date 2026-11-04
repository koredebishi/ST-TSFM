#!/usr/bin/env python3
"""Check that each frozen model forecasts every sensor independently.

baselines/tsfm_zeroshot_eval.py passes a (chunk, C) array of sensor
contexts to the model. If a row's forecast depended on other rows, the
S3 holdout would not be a real holdout.

Per model, on real test windows:
  A. alone   = fn(ctx[i:i+1])            one series, batch of 1
  B. inbatch = fn(ctx[0:CHUNK])[i]       same series inside a full chunk
  C. perm    = fn(ctx[perm])[pos_of_i]   same chunk, rows shuffled

Independent means A == B == C. We report the largest absolute difference
and the difference relative to the series' forecast scale.

Uses the loaders of baselines/tsfm_zeroshot_eval.py, so the tested code is
the code that produced the forecasts.
Output: results/verification/batch_independence.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "baselines"))

from tsfm_zeroshot_eval import (CHUNK, CONTEXT_LEN, load_chronos2,  # noqa: E402
                                load_dataset, load_timesfm)

SEED = 2023
PROBE_SENSORS = (0, 7, 42, 128, 255)   # rows inside the first chunk
DATASET = "largest_sd"                  # the S3 dataset; 716 sensors


def probe(fn, ctx, rng):
    """ctx: (N, C). Returns per-probe-sensor deviation stats."""
    n_batch = min(CHUNK, ctx.shape[0])
    batch = np.ascontiguousarray(ctx[:n_batch])
    perm = rng.permutation(n_batch)
    inv = np.argsort(perm)

    y_batch = fn(batch)                              # (n_batch, 12)
    y_perm = fn(np.ascontiguousarray(batch[perm]))   # (n_batch, 12)
    y_perm_unsorted = y_perm[inv]                    # back to original order

    out = {}
    for i in PROBE_SENSORS:
        if i >= n_batch:
            continue
        y_alone = fn(np.ascontiguousarray(ctx[i:i + 1]))[0]
        scale = float(np.abs(y_batch[i]).mean()) or 1.0
        out[str(i)] = {
            "max_abs_alone_vs_batch": float(np.abs(y_alone - y_batch[i]).max()),
            "max_abs_batch_vs_perm": float(
                np.abs(y_batch[i] - y_perm_unsorted[i]).max()),
            "rel_alone_vs_batch": float(
                np.abs(y_alone - y_batch[i]).max() / scale),
            "rel_batch_vs_perm": float(
                np.abs(y_batch[i] - y_perm_unsorted[i]).max() / scale),
            "forecast_scale": scale,
        }
    return out


def main():
    assert torch.cuda.is_available(), "No CUDA device visible."
    rng = np.random.default_rng(SEED)
    raw, anchors, _, source = load_dataset(DATASET)
    a = int(anchors[0])
    ctx = raw[a - CONTEXT_LEN + 1 : a + 1].T   # (N, C), exactly as the audit
    assert ctx.shape[1] == CONTEXT_LEN, ctx.shape

    result = {
        "spec": "batch-independence check for the frozen backbones",
        "dataset": DATASET, "source": source,
        "anchor": a, "context_len": CONTEXT_LEN, "chunk": CHUNK,
        "n_sensors": int(ctx.shape[0]), "seed": SEED,
        "probe_sensors": list(PROBE_SENSORS),
        "job": os.environ.get("SLURM_JOB_ID"),
        "host": os.uname().nodename,
        "models": {},
    }

    for name, loader in (("chronos2", load_chronos2),
                         ("timesfm", load_timesfm)):
        fn, meta = loader()
        print(f"[bi] {name}: {meta}", flush=True)
        stats = probe(fn, ctx, np.random.default_rng(SEED))
        worst_ab = max(v["max_abs_alone_vs_batch"] for v in stats.values())
        worst_bp = max(v["max_abs_batch_vs_perm"] for v in stats.values())
        result["models"][name] = {
            "meta": meta, "per_sensor": stats,
            "worst_max_abs_alone_vs_batch": worst_ab,
            "worst_max_abs_batch_vs_perm": worst_bp,
        }
        print(f"[bi] {name}: worst |alone-batch| = {worst_ab:.3e}, "
              f"worst |batch-perm| = {worst_bp:.3e}", flush=True)
        del fn
        torch.cuda.empty_cache()

    out = REPO / "results/verification/batch_independence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"[bi] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
