"""Per-sensor ARIMA(3,0,1) baseline (DCRNN paper convention).

Fits a SARIMAX(3,0,1) state-space model per sensor on the training portion,
then for each test window: applies the fitted model to the 12-step input
history and forecasts the next 12 steps. Failed fits fall back to the
training-portion per-sensor mean. Sensors are processed in parallel via joblib.

For METR-LA / PEMS-BAY we reuse the existing windowed test.npz so the test
predictions align step-for-step with DCRNN / GWNet. For LargeST-SD we
materialise windows from his.npz + idx_test.npy in the same way.

Usage:
    python baselines/arima/run.py --dataset metr_la  --n_jobs 16 --output <path>.json
    python baselines/arima/run.py --dataset pems_bay --n_jobs 16 --output <path>.json
    python baselines/arima/run.py --dataset largest_sd --n_jobs 16 --output <path>.json
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from eval.metrics import all_metrics  # noqa: E402

SEQ_IN = 12
SEQ_OUT = 12
ARIMA_ORDER = (3, 0, 1)  # as in the DCRNN paper


def load_metr_pems(h5_path: Path, test_npz_path: Path):
    df = pd.read_hdf(h5_path)
    T = len(df)
    n_windows = T - SEQ_IN - SEQ_OUT + 1
    n_train = round(n_windows * 0.7)
    train_end = n_train + SEQ_IN
    train_full = df.values[:train_end].astype(np.float64)
    test = np.load(test_npz_path)
    x_test = test["x"][..., 0].astype(np.float64)  # (B, 12, N)
    y_test = test["y"][..., 0].astype(np.float64)
    return train_full, x_test, y_test


def load_largest_sd(data_dir: Path):
    npz = np.load(data_dir / "his.npz")
    # LargeST-SD channel 0 is z-scored TRAFFIC FLOW (vehicles per 15-min interval).
    # Inverse-transform back to raw counts so metrics are in interpretable units
    # and zero-masking correctly identifies stalled sensors (raw 0, not z 0).
    speed = (npz["data"][..., 0].astype(np.float64) * float(npz["std"])
             + float(npz["mean"]))
    idx_train = np.load(data_dir / "idx_train.npy")
    idx_test = np.load(data_dir / "idx_test.npy")
    # LargeST convention: idx is the anchor (last input step). Input window is
    # data[idx - SEQ_IN + 1 : idx + 1] and target is data[idx + 1 : idx + 1 + SEQ_OUT].
    train_end = int(idx_train.max()) + 1 + SEQ_OUT
    train_full = speed[:train_end]
    n_test = len(idx_test)
    N = speed.shape[1]
    x_test = np.zeros((n_test, SEQ_IN, N))
    y_test = np.zeros_like(x_test)
    for k, i in enumerate(idx_test):
        x_test[k] = speed[i - SEQ_IN + 1 : i + 1]
        y_test[k] = speed[i + 1 : i + 1 + SEQ_OUT]
    return train_full, x_test, y_test


def fit_and_forecast_sensor(j: int, train_col: np.ndarray, x_test_col: np.ndarray):
    """Returns (j, predictions of shape (n_test, SEQ_OUT))."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    valid_train = train_col[np.abs(train_col) > 1e-3]
    fallback_mean = float(valid_train.mean()) if valid_train.size else 0.0
    n_test = x_test_col.shape[0]

    try:
        # statsmodels default stationarity checks kept on:
        # unconstrained MLE produced explosive MA roots on PEMS-BAY.
        fit = SARIMAX(
            train_col,
            order=ARIMA_ORDER,
        ).fit(disp=False, maxiter=50)
    except Exception:
        return j, np.full((n_test, SEQ_OUT), fallback_mean)

    out = np.empty((n_test, SEQ_OUT))
    for k in range(n_test):
        try:
            applied = fit.apply(x_test_col[k], refit=False)
            out[k] = np.asarray(applied.forecast(steps=SEQ_OUT))
        except Exception:
            out[k] = fallback_mean
    return j, out


def run(train_full: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
        dataset: str, source: str, output: Path, n_jobs: int,
        save_preds: bool = False):
    N = train_full.shape[1]
    print(f"[arima] dataset={dataset}  train_T={train_full.shape[0]}  "
          f"test_B={x_test.shape[0]}  sensors={N}  jobs={n_jobs}", flush=True)

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(fit_and_forecast_sensor)(j, train_full[:, j], x_test[:, :, j])
        for j in range(N)
    )
    print(f"[arima] fit+forecast done in {time.time() - t0:.1f}s", flush=True)

    y_pred = np.zeros_like(y_test)
    for j, preds in results:
        y_pred[:, :, j] = preds

    horizons = {
        str(h + 1): all_metrics(y_test[:, h], y_pred[:, h], mask_zero=True)
        for h in range(SEQ_OUT)
    }
    avg = all_metrics(
        y_test.reshape(-1, N), y_pred.reshape(-1, N), mask_zero=True
    )
    # Metric units differ across datasets: METR-LA / PEMS-BAY are speed (mph)
    # at 5-min cadence; LargeST-SD is traffic flow (vehicles per 15-min interval).
    units = "vehicles_per_15min" if dataset == "largest_sd" else "mph"
    cadence = "15min" if dataset == "largest_sd" else "5min"
    out = {
        "model": "arima",
        "order": list(ARIMA_ORDER),
        "enforce_stationarity": True,
        "enforce_invertibility": True,
        "spec": "decisions.md D1 (2026-06-10)",
        "dataset": dataset,
        "units": units,
        "cadence": cadence,
        "source": source,
        "host": os.uname().nodename,
        "mask_zero": True,
        "horizons": horizons,
        "average": avg,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(f"[arima] {dataset} avg MAE={avg['mae']:.4f} {units} RMSE={avg['rmse']:.4f} → {output}")
    if save_preds:
        # canonical (B, H, N) float32, matching the banked TSFM convention
        np.save(output.parent / "y_pred.npy", y_pred.astype(np.float32))
        np.save(output.parent / "y_true.npy", y_test.astype(np.float32))
        print(f"[arima] saved y_pred/y_true .npy to {output.parent}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["metr_la", "pems_bay", "largest_sd"])
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n_jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--save-preds", action="store_true",
                   help="also save y_pred.npy / y_true.npy (B, H, N) float32 "
                        "next to --output")
    args = p.parse_args()

    if args.dataset == "metr_la":
        train_full, x_test, y_test = load_metr_pems(
            REPO / "datasets/METR-LA/metr-la.h5",
            REPO / "datasets/METR-LA/processed/test.npz",
        )
        src = "metr-la.h5 + processed/test.npz"
    elif args.dataset == "pems_bay":
        train_full, x_test, y_test = load_metr_pems(
            REPO / "datasets/PEMS-BAY/pems-bay.h5",
            REPO / "datasets/PEMS-BAY/processed/test.npz",
        )
        src = "pems-bay.h5 + processed/test.npz"
    else:
        train_full, x_test, y_test = load_largest_sd(REPO / "datasets/LargeST-SD/2019")
        src = "LargeST-SD/2019 (his.npz + idx_test.npy)"

    run(train_full, x_test, y_test, args.dataset, src, args.output, args.n_jobs,
        args.save_preds)


if __name__ == "__main__":
    main()
