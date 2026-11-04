#!/usr/bin/env python3
"""Zero-shot Chronos-2 and TimesFM 2.5 forecasts on the fixed test windows.

One script for both models and all three datasets. Every model sees the
same 12-step target windows and is scored with eval/metrics.py
(zero readings masked). The context is 512 steps unless --context-len is
given.

Before any forecasting, the target windows are checked against the
dataset files: for METR-LA and PEMS-BAY the test anchors are rebuilt from
the DCRNN windowing and compared with test.npz; for LargeST-SD the
anchors are idx_test.npy.

Usage:
    python baselines/tsfm_zeroshot_eval.py --model chronos2 --dataset metr_la \
        --output results/baselines/chronos2/metr_la/test_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from eval.metrics import all_metrics  # noqa: E402

SEQ_IN = 12
SEQ_OUT = 12
CONTEXT_LEN = 512  # both models, all datasets
CHUNK = 256        # series per forward pass

DATASETS = {
    "metr_la": dict(h5=REPO / "datasets/METR-LA/metr-la.h5",
                    npz=REPO / "datasets/METR-LA/processed/test.npz",
                    units="mph", cadence="5min"),
    "pems_bay": dict(h5=REPO / "datasets/PEMS-BAY/pems-bay.h5",
                     npz=REPO / "datasets/PEMS-BAY/processed/test.npz",
                     units="mph", cadence="5min"),
    "largest_sd": dict(dir=REPO / "datasets/LargeST-SD/2019",
                       units="vehicles_per_15min", cadence="15min"),
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        return "unknown"


def load_dataset(name: str):
    """Returns (raw (T,N) float32, anchors (B,), y_test (B,12,N) float64, source str)."""
    cfg = DATASETS[name]
    if name == "largest_sd":
        npz = np.load(cfg["dir"] / "his.npz")
        raw = (npz["data"][..., 0].astype(np.float64) * float(npz["std"])
               + float(npz["mean"]))
        idx_test = np.load(cfg["dir"] / "idx_test.npy").astype(int)
        B, N = len(idx_test), raw.shape[1]
        y = np.zeros((B, SEQ_OUT, N))
        for k, i in enumerate(idx_test):
            y[k] = raw[i + 1 : i + 1 + SEQ_OUT]
        return raw.astype(np.float32), idx_test, y, "LargeST-SD/2019 (his.npz + idx_test.npy)"

    df = pd.read_hdf(cfg["h5"])
    raw = df.values.astype(np.float64)
    T = len(raw)
    test = np.load(cfg["npz"])
    y = test["y"][..., 0].astype(np.float64)  # (B, 12, N)
    B = y.shape[0]

    # Reconstruct test-window anchors from the DCRNN windowing arithmetic
    # (generate_training_data.py: 70/10/20 chrono split over T-23 windows).
    num_samples = T - SEQ_IN - SEQ_OUT + 1
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_train - num_test
    assert B == num_test, f"test.npz has {B} windows, arithmetic says {num_test}"
    anchors = (SEQ_IN - 1) + num_train + num_val + np.arange(B)

    # Alignment assertion: raw targets must equal the frozen npz targets.
    for k in (0, B // 2, B - 1):
        a = anchors[k]
        assert np.allclose(raw[a + 1 : a + 1 + SEQ_OUT], y[k], atol=1e-3), \
            f"anchor misalignment at test window {k}"
    return raw.astype(np.float32), anchors, y, f"{cfg['h5'].name} + processed/test.npz"


# ---------------------------------------------------------------- models

def load_chronos2():
    from chronos import BaseChronosPipeline
    model_id = "amazon/chronos-2"
    try:
        pipe = BaseChronosPipeline.from_pretrained(
            model_id, device_map="cuda", dtype=torch.bfloat16)
    except TypeError:
        pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cuda")

    def fn(ctx: np.ndarray) -> np.ndarray:  # ctx (n, C) -> (n, 12)
        t = torch.from_numpy(ctx)
        try:
            q, _ = pipe.predict_quantiles(
                t, prediction_length=SEQ_OUT, quantile_levels=[0.5])
        except (ValueError, AssertionError):
            q, _ = pipe.predict_quantiles(
                t.unsqueeze(1), prediction_length=SEQ_OUT, quantile_levels=[0.5])
        if isinstance(q, list):
            q = torch.stack([torch.as_tensor(x) for x in q])
        q = torch.as_tensor(q).squeeze(-1)
        while q.dim() > 2:
            q = q.squeeze(1)
        return q.cpu().float().numpy()

    meta = {"model_id": model_id, "api": type(pipe).__name__,
            "package": f"chronos-forecasting=={pkg_version('chronos-forecasting')}"}
    return fn, meta


def load_timesfm(context_len: int = CONTEXT_LEN):
    import timesfm
    errors = []
    try:  # 2.x API
        model_id = "google/timesfm-2.5-200m-pytorch"
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        m.compile(timesfm.ForecastConfig(
            max_context=context_len, max_horizon=SEQ_OUT,
            normalize_inputs=True))

        def fn(ctx: np.ndarray) -> np.ndarray:
            point, _ = m.forecast(horizon=SEQ_OUT,
                                  inputs=[ctx[j] for j in range(len(ctx))])
            return np.asarray(point)[:, :SEQ_OUT]

        api = "timesfm-2.x"
    except Exception as e:
        errors.append(f"2.x API: {e}")
        model_id = "google/timesfm-2.0-500m-pytorch"
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu", per_core_batch_size=32,
                context_len=context_len, horizon_len=SEQ_OUT,
                num_layers=50, model_dims=1280,
                use_positional_embedding=False),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_id))

        def fn(ctx: np.ndarray) -> np.ndarray:
            point, _ = tfm.forecast([ctx[j] for j in range(len(ctx))],
                                    freq=[0] * len(ctx))
            return np.asarray(point)[:, :SEQ_OUT]

        api = "timesfm-1.x"

    meta = {"model_id": model_id, "api": api,
            "package": f"timesfm=={pkg_version('timesfm')}",
            "load_fallback_errors": errors or None}
    return fn, meta


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["chronos2", "timesfm"])
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--output", required=True)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    # Sharding: a window range saves partial predictions;
    # --merge assembles partials and computes metrics. Predictions are
    # byte-identical to a single-job run — compute scheduling only.
    ap.add_argument("--window-start", type=int, default=None)
    ap.add_argument("--window-end", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    # Save full predictions next to the JSON. Shift-test pools (S1/S2) are
    # subsets of the frozen test windows, so with saved predictions the shift
    # pools are scored by re-slicing, with no new inference.
    ap.add_argument("--save-preds", action="store_true")
    # Context length. The default is 512, so runs
    # without the flag behave exactly as before.
    ap.add_argument("--context-len", type=int, default=CONTEXT_LEN)
    args = ap.parse_args()
    C = args.context_len
    assert C > 0, f"bad --context-len {C}"
    import hashlib
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    raw, anchors, y_test, source = load_dataset(args.dataset)
    B, N = len(anchors), raw.shape[1]
    partial_dir = Path(args.output).parent / "partials"
    shard_mode = args.window_start is not None or args.window_end is not None

    if args.merge:
        shards = sorted(partial_dir.glob("pred_*.npy"))
        assert shards, f"no partials in {partial_dir}"
        y_pred = np.full((B, SEQ_OUT, N), np.nan, dtype=np.float32)
        wall = 0.0
        for p in shards:
            ws, we = map(int, p.stem.split("_")[1:3])
            y_pred[ws:we] = np.load(p)
            m = json.loads((partial_dir / f"meta_{ws}_{we}.json").read_text())
            assert m["context_len"] == C, \
                f"shard {p.name} has context_len {m['context_len']}, merge asked {C}"
            wall += m["predict_wall_seconds"]
        assert not np.isnan(y_pred).any(), "incomplete shard coverage"
        meta = {k: v for k, v in m.items()
                if k not in ("predict_wall_seconds", "window_start", "window_end")}
        meta["shards"] = [p.name for p in shards]
        print(f"[tsfm] merged {len(shards)} shards covering {B} windows", flush=True)
    else:
        assert torch.cuda.is_available(), "No CUDA device visible."
        print(f"[tsfm] model={args.model} dataset={args.dataset} "
              f"context_len={C} chunk={args.chunk} "
              f"gpu={torch.cuda.get_device_name(0)}", flush=True)
        print(f"[tsfm] windows={B} sensors={N} raw_T={raw.shape[0]}", flush=True)
        assert anchors.min() - C + 1 >= 0, \
            "first test anchor has insufficient history for context_len"

        fn, meta = (load_chronos2() if args.model == "chronos2"
                    else load_timesfm(C))
        print(f"[tsfm] loaded {meta}", flush=True)

        # Warmup (compile paths, HF cache) — excluded from timing.
        fn(raw[anchors[0] - C + 1 : anchors[0] + 1].T[: min(8, N)].copy())

        ws = args.window_start or 0
        we = args.window_end if args.window_end is not None else B
        assert 0 <= ws < we <= B, f"bad window range [{ws}, {we}) for B={B}"
        y_pred = np.zeros((we - ws, SEQ_OUT, N), dtype=np.float32)
        t0 = time.time()
        for j, k in enumerate(range(ws, we)):
            a = anchors[k]
            ctx = raw[a - C + 1 : a + 1].T  # (N, C)
            outs = [fn(np.ascontiguousarray(ctx[s : s + args.chunk]))
                    for s in range(0, N, args.chunk)]
            y_pred[j] = np.concatenate(outs).T  # (12, N)
            if j % 100 == 0 or k == we - 1:
                el = time.time() - t0
                rate = (j + 1) / el
                print(f"[tsfm] window {k + 1}/{we} (shard [{ws},{we}))  "
                      f"{el:.0f}s elapsed  {rate:.2f} win/s  "
                      f"ETA {(we - k - 1) / rate / 3600:.2f}h", flush=True)
        wall = time.time() - t0

        if shard_mode:
            partial_dir.mkdir(parents=True, exist_ok=True)
            np.save(partial_dir / f"pred_{ws}_{we}.npy", y_pred)
            meta.update(predict_wall_seconds=round(wall, 1),
                        window_start=ws, window_end=we,
                        context_len=C, chunk=args.chunk,
                        host=os.uname().nodename,
                        script_sha256=script_sha256)
            (partial_dir / f"meta_{ws}_{we}.json").write_text(
                json.dumps(meta, indent=2))
            print(f"[tsfm] wrote shard [{ws},{we}) to {partial_dir}")
            return

    if args.save_preds:
        pred_path = Path(args.output).with_name("y_pred.npy")
        np.save(pred_path, y_pred)
        print(f"[tsfm] saved predictions to {pred_path}")

    horizons = {str(h + 1): all_metrics(y_test[:, h], y_pred[:, h].astype(np.float64),
                                        mask_zero=True)
                for h in range(SEQ_OUT)}
    avg = all_metrics(y_test.reshape(-1, N),
                      y_pred.reshape(-1, N).astype(np.float64), mask_zero=True)

    cfg = DATASETS[args.dataset]
    out = {
        "model": f"{args.model}-zeroshot",
        **meta,
        "dataset": args.dataset,
        "units": cfg["units"],
        "cadence": cfg["cadence"],
        "context_len": C,
        "chunk": args.chunk,
        "script_sha256": script_sha256,
        "spec": "decisions.md D7 (2026-06-10)",
        "seed": None,
        "source": source,
        "host": os.uname().nodename,
        "git_commit": git_commit(),
        "mask_zero": True,
        "n_windows": int(B),
        "predict_wall_seconds": round(wall, 1),
        "horizons": horizons,
        "average": avg,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[tsfm] avg: {json.dumps(avg)}")
    print(f"[tsfm] wrote {args.output}")


if __name__ == "__main__":
    main()
