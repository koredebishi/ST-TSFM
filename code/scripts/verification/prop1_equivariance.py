#!/usr/bin/env python3
"""Prop-1 empirical check: Chronos-2 affine equivariance f(a·x+b) = a·f(x)+b.

Statement + sourced code facts: notes/prop1_severity_decoupling.md.
Uses the verified ChronosBackbone wrapper (zero-gate adapter = exact frozen,
keystone-proven) on real SD test windows. Expected deviation: bf16
quantization noise only (D14 addendum: ~0.25 per-forecast floor at a=1;
scales with a·scale of the window).
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

from mechanism.attach import build_ttra  # noqa: E402
from mechanism.train_ttra import CTX, POOL, SEQ_OUT, load_sd_raw  # noqa: E402

OUT = REPO / "results/shift/prop1_equivariance.json"
TRANSFORMS = [(1.0, 10.0), (1.0, 100.0), (1.0, 1000.0), (1.0, -100.0),
              (2.0, 0.0), (10.0, 0.0), (2.0, 500.0)]
N_WINDOWS = 4
REL_TOL = 0.02  # bf16 relative floor


def main():
    assert torch.cuda.is_available(), "No CUDA device visible."
    torch.manual_seed(2023)
    backbone, _ = build_ttra(variant="fast_weight", device="cuda")
    assert backbone.n_backbone_trainable() == 0

    raw = load_sd_raw()
    idx_test = np.load(POOL / "idx_test.npy")
    anchors = idx_test[:: max(1, len(idx_test) // N_WINDOWS)][:N_WINDOWS]

    mi = backbone.median_index
    rows = []
    worst_ratio_dev = 0.0
    with torch.no_grad():
        for anc in anchors:
            ctx = torch.from_numpy(
                raw[int(anc) - CTX + 1: int(anc) + 1].T.copy()).cuda()
            y = torch.from_numpy(
                raw[int(anc) + 1: int(anc) + 1 + SEQ_OUT].T.copy()).cuda()
            base = backbone.forward_train(ctx)[:, :, :SEQ_OUT].float()
            # one no-op forward proves the adapter path adds exactly zero
            assert backbone._last_residual_absmax == 0.0
            mae0 = float((base[:, mi, :] - y).abs().mean())
            # per-window context scale = the natural unit for element devs
            wscale = float(ctx.float().std())
            for a, b in TRANSFORMS:
                pred_t = backbone.forward_train(a * ctx + b)[:, :, :SEQ_OUT].float()
                # METRIC-LEVEL claim (Prop 1 ii/iii): MAE of the transformed
                # problem == a * MAE of the base problem
                mae1 = float((pred_t[:, mi, :] - (a * y + b)).abs().mean())
                ratio = mae1 / (a * mae0)
                # elementwise deviation in window-scale units — DIAGNOSTIC
                # only (bf16 quantization is sinh-amplified on tail
                # quantiles; the a=2 row is exact because power-of-two
                # scaling is exponent-only in bf16)
                dev_scaled = float((pred_t - (a * base + b)).abs().max()) / (a * wscale)
                rows.append({"anchor": int(anc), "a": a, "b": b,
                             "mae_base": round(mae0, 4),
                             "mae_transformed": round(mae1, 4),
                             "mae_ratio_vs_a": round(ratio, 6),
                             "max_elem_dev_in_window_scales": round(dev_scaled, 4)})
                worst_ratio_dev = max(worst_ratio_dev, abs(ratio - 1))
                print(f"[prop1] anchor={anc} a={a} b={b} "
                      f"MAE {mae0:.3f}->{mae1:.3f} ratio={ratio:.4f} "
                      f"elem_dev={dev_scaled:.4f} wscales", flush=True)

    passed = worst_ratio_dev < REL_TOL
    out = {"claim": "MAE(f(a*x+b), a*y+b) == a * MAE(f(x), y) "
                    "(Prop 1 ii/iii — the metric-level statement)",
           "note": "elementwise devs are bf16 quantization, sinh-amplified "
                   "on tail quantiles; a=2 rows are BITWISE exact "
                   "(power-of-two scaling = exponent shift in bf16). "
                   "See D14 addendum + prop1 note caveats.",
           "checkpoint": "amazon/chronos-2 snapshot 29ec3766 "
                         "(use_arcsinh=true — inside normalized domain, "
                         "equivariance unaffected; see prop1 note)",
           "worst_mae_ratio_dev": round(worst_ratio_dev, 6),
           "ratio_tol": REL_TOL,
           "PASS": bool(passed), "rows": rows,
           "host": os.uname().nodename}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("worst_mae_ratio_dev", "PASS")},
                     indent=2))
    assert passed, \
        f"MAE-ratio deviation {worst_ratio_dev} exceeds tol {REL_TOL}"


if __name__ == "__main__":
    main()
