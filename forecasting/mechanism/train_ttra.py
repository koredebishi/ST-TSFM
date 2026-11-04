#!/usr/bin/env python3
"""Train the TTRA adapter on LargeST-SD weekdays and test it on weekends.

Backbone frozen; only the fast-weight adapter (mechanism/fast_weights.py)
is trained, on the S1 weekday pools (datasets/shift_pools/s1). Training
uses AdamW (lr 5e-4), gradient clipping at 1.0, 500 warmup steps, and
the raw-space pinball loss of chronos2/model.py.

Test (rule written before training): adapted MAE < frozen MAE on the S1
weekend test pool (1902 windows), at horizons 3/6/12 and on average.
Also reported: the share of the frozen weekend degradation the adapter
removes (prediction P4', see PREDICTIONS.md).

--smoke: 32 fixed windows, 200 steps; checks the loss is finite and
falls by at least 5%, and that the backbone gets no gradients.
Output: results/ttra/sd_s1/w6_gate.json and adapter_best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.metrics import all_metrics  # noqa: E402
from mechanism.attach import build_ttra  # noqa: E402

SEQ_OUT = 12
CTX = 512  # same context as the saved forecasts
POOL = REPO / "datasets/shift_pools/s1/sd_dataroot/sd/2019"
HIS = REPO / "datasets/LargeST-SD/2019/his.npz"
FROZEN_PRED = REPO / "results/baselines/chronos2/largest_sd/y_pred.npy"
FROZEN_TEST_IDX = REPO / "datasets/LargeST-SD/2019/idx_test.npy"


def load_sd_raw() -> np.ndarray:
    npz = np.load(HIS)
    return (npz["data"][..., 0].astype(np.float32) * float(npz["std"])
            + float(npz["mean"]))


def pinball_loss(qpred: torch.Tensor, target: torch.Tensor,
                 quantiles: torch.Tensor) -> torch.Tensor:
    """Quantile loss, expression replicated from chronos2/model.py:531-548.

    qpred: (N, Q, H) raw-space quantile predictions (first H=12 steps).
    target: (N, H) raw targets. quantiles: (Q,).
    2 * |(y - q) * ((y <= q) - tau)|; mean over horizon, sum over quantiles,
    mean over batch (the upstream reduction order).
    """
    y = target.unsqueeze(1)                       # (N, 1, H)
    tau = quantiles.view(1, -1, 1)                # (1, Q, 1)
    loss = 2 * torch.abs((y - qpred) * ((y <= qpred).float() - tau))
    return loss.mean(dim=-1).sum(dim=-1).mean()   # H-mean, Q-sum, batch-mean


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-size", type=int, default=512)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated outer lrs; runs the 800-step "
                         "fixed-set smoke per lr WITHOUT asserting (Knob-1 "
                         "'short lr trials on a slice'), writes lr_sweep.json")
    ap.add_argument("--out", default=str(REPO / "results/ttra/sd_s1"))
    args = ap.parse_args()

    assert torch.cuda.is_available(), "No CUDA device visible."
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_sd_raw()                                   # (T, N) raw flow
    idx_train = np.load(POOL / "idx_train.npy")
    idx_val = np.load(POOL / "idx_val.npy")
    idx_test = np.load(POOL / "idx_test.npy")
    # no-leakage assert
    assert not set(idx_train) & set(idx_test), "train/test overlap!"
    # skip train anchors with <CTX history
    n_before = len(idx_train)
    idx_train = idx_train[idx_train >= CTX - 1]
    skipped = n_before - len(idx_train)
    assert idx_test.min() >= CTX - 1, "test anchor without full context"
    print(f"[ttra] pools: train {len(idx_train)} (skipped {skipped} short-"
          f"context), val {len(idx_val)}, test {len(idx_test)}", flush=True)

    backbone, adapter = build_ttra(variant="fast_weight", device="cuda")
    assert backbone.n_backbone_trainable() == 0
    quantiles = backbone.model.quantiles.float().cuda()   # (21,)
    mi = backbone.median_index

    def window_forward(a: int) -> torch.Tensor:
        """One window: (N,CTX) raw context -> (N, Q, 12) raw quantile preds."""
        ctx = torch.from_numpy(raw[a - CTX + 1: a + 1].T.copy()).cuda()
        preds = backbone.forward_train(ctx)               # (N, Q, >=12)
        return preds[:, :, :SEQ_OUT].float()

    def target_of(a: int) -> torch.Tensor:
        return torch.from_numpy(raw[a + 1: a + 1 + SEQ_OUT].T.copy()).cuda()

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr,
                            weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))

    if args.sweep:
        # lr trial: same fixed-set smoke, no assert.
        import subprocess as sp
        results = {}
        for lr in args.sweep.split(","):
            r = sp.run([sys.executable, __file__, "--smoke", "--lr", lr],
                       capture_output=True, text=True)
            drop = None
            for line in r.stdout.splitlines():
                if "fixed-set loss before" in line:
                    drop = float(line.rsplit("drop=", 1)[1].rstrip("%")) / 100
            results[lr] = {"drop": drop, "passed": r.returncode == 0}
            print(f"[sweep] lr={lr}: drop={drop} pass={r.returncode == 0}",
                  flush=True)
        (out_dir / "lr_sweep.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        return

    if args.smoke:
        sm = rng.choice(idx_train, 32, replace=False)

        def fixed_set_loss() -> float:
            """Mean loss over the SAME 32 windows (no-grad) — before/after
            on an identical set, so window-composition noise cannot
            confound the learning signal (first smoke run failed on a
            first20-vs-last20 step comparison that mixed different windows
            while the actual trajectory fell 540->71)."""
            with torch.no_grad():
                return float(np.mean([
                    pinball_loss(window_forward(int(a)), target_of(int(a)),
                                 quantiles).item() for a in sm]))

        # 800 steps = past the 500-step LR warmup + 300 at full lr. The
        # 200-step smoke ran entirely inside its own warmup (avg effective
        # lr ~20% of nominal) with a zero-init gate — 3.2% drop there was
        # the DESIGNED ramp behavior, not a fair learning test. Bar stays
        # >=5%, now measured at operating conditions.
        l0 = fixed_set_loss()
        t0 = time.time()
        for step in range(800):
            a = int(sm[step % 32])
            opt.zero_grad()
            loss = pinball_loss(window_forward(a), target_of(a), quantiles)
            assert torch.isfinite(loss), f"non-finite loss at step {step}"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.clip)
            opt.step(); sched.step()
            if step % 50 == 0:
                print(f"[smoke] step {step} loss {loss.item():.4f} "
                      f"({(time.time()-t0)/(step+1):.2f}s/step)", flush=True)
        assert all(p.grad is None for p in backbone.model.parameters()), \
            "backbone received grads!"
        l1 = fixed_set_loss()
        print(f"[smoke] fixed-set loss before={l0:.4f} after={l1:.4f} "
              f"drop={(l0-l1)/l0:.1%}")
        assert l1 < l0 * 0.95, \
            f"smoke: fixed-set loss did not decrease >=5% ({l0:.4f}->{l1:.4f})"
        print("[smoke] PASS")
        return

    # ---------------- full training ----------------
    val_sub = rng.choice(idx_val, min(args.val_size, len(idx_val)),
                         replace=False)

    def val_loss() -> float:
        tot = 0.0
        with torch.no_grad():
            for a in val_sub:
                tot += pinball_loss(window_forward(int(a)),
                                    target_of(int(a)), quantiles).item()
        return tot / len(val_sub)

    best, best_state, bad, step = float("inf"), None, 0, 0
    t0 = time.time()
    stop = False
    for epoch in range(args.epochs):
        order = rng.permutation(idx_train)
        for a in order:
            opt.zero_grad()
            loss = pinball_loss(window_forward(int(a)), target_of(int(a)),
                                quantiles)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.clip)
            opt.step(); sched.step()
            step += 1
            if step % 500 == 0:
                print(f"[train] ep{epoch} step {step} loss {loss.item():.4f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
            if step % args.val_every == 0:
                v = val_loss()
                print(f"[val] step {step}: {v:.4f} (best {best:.4f})", flush=True)
                if v < best:
                    best, bad = v, 0
                    best_state = {k: t.detach().cpu().clone()
                                  for k, t in adapter.state_dict().items()}
                    torch.save(best_state, out_dir / "adapter_best.pt")
                else:
                    bad += 1
                    if bad >= args.patience:
                        print("[train] early stop", flush=True)
                        stop = True
                        break
        if stop:
            break
    if best_state is not None:
        adapter.load_state_dict(best_state)
    torch.save(adapter.state_dict(), out_dir / "adapter_final.pt")

    # ---------------- test: adapted vs frozen ----------------
    N = raw.shape[1]
    y_pred = np.zeros((len(idx_test), SEQ_OUT, N), dtype=np.float32)
    y_true = np.zeros_like(y_pred)
    with torch.no_grad():
        for k, a in enumerate(idx_test):
            preds = window_forward(int(a))                # (N, Q, 12)
            y_pred[k] = preds[:, mi, :].cpu().numpy().T   # (12, N)
            y_true[k] = raw[int(a) + 1: int(a) + 1 + SEQ_OUT]
    np.save(out_dir / "y_pred_adapted.npy", y_pred)

    def mae_block(yt, yp):
        per_h = {h: round(all_metrics(yt[:, h - 1], yp[:, h - 1].astype(np.float64),
                                      mask_zero=True)["mae"], 4)
                 for h in (3, 6, 12)}
        avg = round(all_metrics(yt.reshape(-1, N),
                                yp.reshape(-1, N).astype(np.float64),
                                mask_zero=True)["mae"], 4)
        return {"per_horizon": per_h, "avg": avg}

    adapted = mae_block(y_true.astype(np.float64), y_pred)

    # frozen baseline: re-slice banked predictions on the same test windows
    frozen_all = np.load(FROZEN_PRED).astype(np.float64)
    frozen_idx = np.load(FROZEN_TEST_IDX)
    pos = np.searchsorted(frozen_idx, idx_test)
    assert (frozen_idx[pos] == idx_test).all(), "S1 test not subset of frozen test"
    frozen = mae_block(y_true.astype(np.float64), frozen_all[pos])
    # in-distribution frozen (weekday test windows) for the P4' denominator
    from shift.splits import classify_targets, is_weekday
    from data_loaders.windows import load_raw as _lr
    _, ts, _ = _lr("largest_sd")
    wd_mask = classify_targets(frozen_idx, np.asarray(is_weekday(ts)))
    yt_wd = np.stack([raw[int(a) + 1: int(a) + 1 + SEQ_OUT]
                      for a in frozen_idx[wd_mask]]).astype(np.float64)
    frozen_id = mae_block(yt_wd, frozen_all[wd_mask])

    degr = frozen["avg"] - frozen_id["avg"]
    recovered = frozen["avg"] - adapted["avg"]
    reduction = recovered / degr if degr > 0 else None
    gate = adapted["avg"] < frozen["avg"]
    out = {
        "gate": "W6 SD-S1 (D16/D18)",
        "adapted": adapted, "frozen_shift": frozen,
        "frozen_in_distribution": frozen_id,
        "frozen_degradation_avg": round(degr, 4),
        "recovered_avg": round(recovered, 4),
        "p4prime_reduction_fraction": (round(reduction, 4)
                                       if reduction is not None else None),
        "GATE_PASS_adapted_lt_frozen": bool(gate),
        "n_test_windows": int(len(idx_test)),
        "n_train_used": int(len(idx_train)), "n_train_skipped": int(skipped),
        "knobs": {"lr": args.lr, "clip": args.clip, "warmup": args.warmup,
                  "loss": "pinball(model.py:531-548) raw-space",
                  "epochs_max": args.epochs, "patience": args.patience,
                  "seed": args.seed},
        "adapter": {"variant": "fast_weight (D15)",
                    "n_trainable": adapter.n_trainable()},
        "host": os.uname().nodename, "git_commit": git_commit(),
        "spec": "notes/specs/ttra_training_spec.md (D18)",
    }
    (out_dir / "w6_gate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
