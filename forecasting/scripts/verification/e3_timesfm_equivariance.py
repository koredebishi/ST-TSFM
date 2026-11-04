#!/usr/bin/env python3
"""TimesFM affine-equivariance check (same test as prop1_equivariance.py).

1. Records the normalization source code of the installed timesfm
   package, so the paper's claim about its form points to the code.
2. Loads TimesFM the same way as baselines/tsfm_zeroshot_eval.py and
   compares f(a*x+b) with a*f(x)+b on real LargeST-SD test windows, for
   the same transforms as prop1_equivariance.py.

Output: results/shift/e3_timesfm_equivariance.json
"""
from __future__ import annotations

import inspect
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from baselines.tsfm_zeroshot_eval import (CONTEXT_LEN, SEQ_OUT,  # noqa: E402
                                          load_dataset, load_timesfm)

OUT = REPO / "results/shift/e3_timesfm_equivariance.json"
TRANSFORMS = [(1.0, 10.0), (1.0, 100.0), (1.0, 1000.0), (1.0, -100.0),
              (2.0, 0.0), (10.0, 0.0), (2.0, 500.0)]
N_WINDOWS = 4
N_SERIES = 32  # sensors per window, keeps the run light


def inspect_normalization() -> dict:
    """Capture the installed package's normalization source verbatim."""
    import timesfm
    pkg = Path(inspect.getfile(timesfm)).parent
    hits = []
    for p in sorted(pkg.rglob("*.py")):
        text = p.read_text(errors="replace")
        for m in re.finditer(r"def [a-z_]*normal[a-z_]*\(", text):
            start = text.rfind("\n", 0, m.start()) + 1
            snippet = text[start:start + 1200]
            hits.append({"file": str(p.relative_to(pkg)),
                         "snippet": snippet})
    return {"package_dir": str(pkg), "n_hits": len(hits), "hits": hits[:6]}


def main():
    assert torch.cuda.is_available(), "No CUDA device visible."
    torch.manual_seed(2023)

    banked_meta = json.loads(
        (REPO / "results/baselines/timesfm/largest_sd/test_metrics.json"
         ).read_text())
    fn, meta = load_timesfm()
    assert meta["model_id"] == banked_meta["model_id"], \
        (meta["model_id"], banked_meta["model_id"])
    assert meta["api"] == banked_meta["api"], (meta["api"], banked_meta["api"])
    print(f"[e3] loaded {meta} == banked run", flush=True)

    raw, anchors, _, _ = load_dataset("largest_sd")
    sel = anchors[:: max(1, len(anchors) // N_WINDOWS)][:N_WINDOWS]

    rows = []
    worst_ratio_dev = 0.0
    for anc in sel:
        ctx = raw[int(anc) - CONTEXT_LEN + 1: int(anc) + 1].T[:N_SERIES]
        ctx = np.ascontiguousarray(ctx, dtype=np.float32)
        base = fn(ctx.copy())                       # (N_SERIES, 12)
        for a, b in TRANSFORMS:
            out = fn(np.ascontiguousarray(a * ctx + b))
            expect = a * base + b
            scale = float(np.abs(expect).mean()) + 1e-9
            abs_dev = float(np.abs(out - expect).mean())
            rel_dev = float(abs_dev / scale)
            # MAE-ratio framing as in the Chronos-2 verification: the
            # transformed forecast error vs a * base error, target a*y+b.
            rows.append({"anchor": int(anc), "a": float(a), "b": float(b),
                         "mean_abs_dev": abs_dev, "rel_dev": rel_dev})
            worst_ratio_dev = float(max(worst_ratio_dev, rel_dev))
            print(f"[e3] anc={anc} a={a} b={b}: rel_dev={rel_dev:.5f}",
                  flush=True)

    out = {"spec": "decisions.md D37 E3",
           "meta": meta, "banked_meta_model": banked_meta["model_id"],
           "context_len": CONTEXT_LEN, "seq_out": SEQ_OUT,
           "n_windows": len(sel), "n_series": N_SERIES,
           "transforms": TRANSFORMS, "rows": rows,
           "worst_rel_dev": worst_ratio_dev,
           "normalization_inspection": inspect_normalization(),
           "job": os.environ.get("SLURM_JOB_ID")}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"[e3] worst relative deviation {worst_ratio_dev:.5f}; wrote {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
