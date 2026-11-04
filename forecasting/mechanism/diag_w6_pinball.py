#!/usr/bin/env python3
"""Pinball loss of the frozen and adapted model, on validation and test.

The adapter raised test MAE although training lowered the pinball loss
on validation. This measures the training loss (pinball) on both pools
for both models, to tell the two explanations apart. Rule, written before
the run:
  adapted pinball < frozen pinball on test  -> the loss and the metric
                                               disagree
  otherwise                                 -> what the adapter learned on
                                               weekdays does not transfer
                                               to weekends

Uses pinball_loss, load_sd_raw and the pool paths of
mechanism/train_ttra.py (job 46749604). Frozen = adapter with gate 0
(exact no-op); adapted = adapter_best.pt. Check: the recomputed median
MAE must match w6_gate.json (adapted 27.7994, frozen 26.1387).
Output: results/ttra/sd_s1/w6_diag_pinball.json
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

from eval.metrics import all_metrics  # noqa: E402
from mechanism.attach import build_ttra  # noqa: E402
from mechanism.train_ttra import (CTX, POOL, SEQ_OUT, FROZEN_PRED,  # noqa: E402
                                  FROZEN_TEST_IDX, git_commit, load_sd_raw,
                                  pinball_loss)

OUT = REPO / "results/ttra/sd_s1"
ADAPTER = OUT / "adapter_best.pt"
SEED = 2023
# Values from w6_gate.json (job 46749604) for the sanity check
GATE_ADAPTED_AVG, GATE_FROZEN_AVG = 27.7994, 26.1387
MAE_TOL = 0.5  # bf16/cuBLAS per-forecast noise floor is about 0.25


def pinball_elem(qpred: torch.Tensor, target: torch.Tensor,
                 quantiles: torch.Tensor) -> torch.Tensor:
    """Elementwise pinball, same expression as pinball_loss (model.py:531-548)
    before its H-mean/Q-sum/batch-mean reduction. Returns (N, Q, H)."""
    y = target.unsqueeze(1)
    tau = quantiles.view(1, -1, 1)
    return 2 * torch.abs((y - qpred) * ((y <= qpred).float() - tau))


def main():
    assert torch.cuda.is_available(), "No CUDA device visible."
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    raw = load_sd_raw()                                    # (T, N)
    N = raw.shape[1]
    idx_val = np.load(POOL / "idx_val.npy")
    idx_test = np.load(POOL / "idx_test.npy")
    # identical draw to training's val_sub: same seed, first rng consumption
    val_sub = rng.choice(idx_val, min(512, len(idx_val)), replace=False)
    assert idx_test.min() >= CTX - 1

    backbone, adapter = build_ttra(variant="fast_weight", device="cuda")
    assert backbone.n_backbone_trainable() == 0
    quantiles = backbone.model.quantiles.float().cuda()
    mi = backbone.median_index

    def window_forward(a: int) -> torch.Tensor:
        ctx = torch.from_numpy(raw[a - CTX + 1: a + 1].T.copy()).cuda()
        return backbone.forward_train(ctx)[:, :, :SEQ_OUT].float()

    def target_of(a: int) -> torch.Tensor:
        return torch.from_numpy(raw[a + 1: a + 1 + SEQ_OUT].T.copy()).cuda()

    def eval_pool(anchors, collect_median=False):
        tot, ph = 0.0, {3: 0.0, 6: 0.0, 12: 0.0}
        med = np.zeros((len(anchors), SEQ_OUT, N), np.float32) if collect_median else None
        checked = False
        with torch.no_grad():
            for k, a in enumerate(anchors):
                q, y = window_forward(int(a)), target_of(int(a))
                elem = pinball_elem(q, y, quantiles)            # (N, Q, H)
                if not checked:  # one-time parity with the verified reduction
                    ref = pinball_loss(q, y, quantiles).item()
                    got = elem.mean(-1).sum(-1).mean().item()
                    assert abs(ref - got) < 1e-4 * max(1.0, abs(ref)), (ref, got)
                    checked = True
                tot += elem.mean(-1).sum(-1).mean().item()
                for h in ph:
                    ph[h] += elem[:, :, h - 1].sum(1).mean().item()
                if collect_median:
                    med[k] = q[:, mi, :].cpu().numpy().T        # (12, N)
        n = len(anchors)
        return ({"pinball": round(tot / n, 4),
                 "per_horizon": {h: round(v / n, 4) for h, v in ph.items()}},
                med)

    def test_mae(med):
        yt = np.stack([raw[int(a) + 1: int(a) + 1 + SEQ_OUT]
                       for a in idx_test]).astype(np.float64)
        return all_metrics(yt.reshape(-1, N), med.reshape(-1, N).astype(np.float64),
                           mask_zero=True)["mae"]

    results = {}
    # ---- frozen: adapter with gate 0 is an exact no-op ----
    with torch.no_grad():
        _ = window_forward(int(idx_test[0]))
    assert backbone._last_residual_absmax == 0.0, \
        f"zero-gate residual not exact 0: {backbone._last_residual_absmax}"
    fr_val, _ = eval_pool(val_sub)
    fr_test, fr_med = eval_pool(idx_test, collect_median=True)
    fr_mae = test_mae(fr_med)
    assert abs(fr_mae - GATE_FROZEN_AVG) < MAE_TOL, \
        f"frozen MAE sanity: {fr_mae} vs gate {GATE_FROZEN_AVG}"
    results["frozen"] = {"val_weekday": fr_val, "test_weekend": fr_test,
                         "test_median_mae": round(fr_mae, 4)}
    print(f"[diag] frozen: val {fr_val['pinball']} test {fr_test['pinball']} "
          f"mae {fr_mae:.4f}", flush=True)

    # ---- adapted: best checkpoint from job 46749604 ----
    adapter.load_state_dict(torch.load(ADAPTER, map_location="cuda"))
    ad_val, _ = eval_pool(val_sub)
    ad_test, ad_med = eval_pool(idx_test, collect_median=True)
    ad_mae = test_mae(ad_med)
    assert abs(ad_mae - GATE_ADAPTED_AVG) < MAE_TOL, \
        f"adapted MAE sanity: {ad_mae} vs gate {GATE_ADAPTED_AVG}"
    banked = np.load(OUT / "y_pred_adapted.npy")
    drift = float(np.abs(banked - ad_med).mean())
    results["adapted"] = {"val_weekday": ad_val, "test_weekend": ad_test,
                          "test_median_mae": round(ad_mae, 4),
                          "banked_pred_mean_absdiff": round(drift, 4)}
    print(f"[diag] adapted: val {ad_val['pinball']} test {ad_test['pinball']} "
          f"mae {ad_mae:.4f} (banked drift {drift:.4f})", flush=True)

    # ---- decision rule stated before the run ----
    mismatch = ad_test["pinball"] < fr_test["pinball"]
    val_gain = fr_val["pinball"] - ad_val["pinball"]
    test_gain = fr_test["pinball"] - ad_test["pinball"]
    out = {
        "diag": "D19 Tier-1: 2x2 pinball (train objective) on S1 pools",
        **results,
        "adapted_minus_frozen": {
            "val_pinball_gain": round(val_gain, 4),
            "test_pinball_gain": round(test_gain, 4)},
        "VERDICT_objective_mismatch": bool(mismatch),
        "TIER2_AUTHORIZED_mae_loss_retrain": bool(mismatch),
        "rule": "pre-registered D19: adapted<frozen pinball on TEST -> "
                "Tier 2 GO; else transfer failure -> STOP, SCEF reframe",
        "n_val": int(len(val_sub)), "n_test": int(len(idx_test)),
        "adapter_ckpt": str(ADAPTER.relative_to(REPO)),
        "host": os.uname().nodename, "git_commit": git_commit(),
    }
    (OUT / "w6_diag_pinball.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
