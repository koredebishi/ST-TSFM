"""Load the three traffic datasets, the fixed windows and saved forecasts.

A window is named by its anchor `a`, the last input step. The input is
rows a-11 .. a and the target is rows a+1 .. a+12.

METR-LA and PEMS-BAY use the DCRNN split (70 / 10 / 20, in time order).
LargeST-SD uses the split files that ship with LargeST (60 / 20 / 20).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("ST_TSFM_DATA", ROOT / "datasets"))
FORECAST_DIR = Path(os.environ.get("ST_TSFM_FORECASTS", ROOT / "forecasts"))
# Sensor coordinates for METR-LA and PEMS-BAY come with the DCRNN code
# (github.com/liyaguang/DCRNN, data/sensor_graph/).
SENSOR_GRAPH_DIR = Path(os.environ.get("DCRNN_SENSOR_GRAPH",
                                       DATA_DIR / "sensor_graph"))

INPUT_STEPS = 12
TARGET_STEPS = 12
NAMES = ("metr_la", "pems_bay", "largest_sd")
LABELS = {"metr_la": "METR-LA", "pems_bay": "PEMS-BAY", "largest_sd": "LargeST-SD"}


def _files(name):
    if name == "metr_la":
        return dict(h5=DATA_DIR / "METR-LA/metr-la.h5",
                    test=DATA_DIR / "METR-LA/processed/test.npz",
                    coords=SENSOR_GRAPH_DIR / "graph_sensor_locations.csv")
    if name == "pems_bay":
        return dict(h5=DATA_DIR / "PEMS-BAY/pems-bay.h5",
                    test=DATA_DIR / "PEMS-BAY/processed/test.npz",
                    coords=SENSOR_GRAPH_DIR / "graph_sensor_locations_bay.csv")
    if name == "largest_sd":
        return dict(h5=DATA_DIR / "LargeST-SD/sd_his_2019.h5",
                    dir=DATA_DIR / "LargeST-SD/2019",
                    coords=DATA_DIR / "LargeST-SD/sd_meta.csv")
    raise ValueError(f"unknown dataset {name!r}")


def load_series(name):
    """Raw readings (T, N), their timestamps, and the sensor ids."""
    df = pd.read_hdf(_files(name)["h5"])
    return df.values.astype(np.float64), df.index, list(df.columns)


def window_anchors(name):
    """Anchors of the train, val and test windows."""
    files = _files(name)
    if name == "largest_sd":
        return {part: np.load(files["dir"] / f"idx_{part}.npy").astype(int)
                for part in ("train", "val", "test")}

    # DCRNN cuts T - 23 windows into 70 / 10 / 20 in time order.
    n_steps = len(pd.read_hdf(files["h5"]))
    n_windows = n_steps - INPUT_STEPS - TARGET_STEPS + 1
    n_test = round(n_windows * 0.2)
    n_train = round(n_windows * 0.7)
    n_val = n_windows - n_train - n_test
    first = INPUT_STEPS - 1
    return {"train": first + np.arange(n_train),
            "val": first + n_train + np.arange(n_val),
            "test": first + n_train + n_val + np.arange(n_test)}


def largest_sd_flow():
    """LargeST-SD flow from his.npz, the file the models read (T, N).

    LargeST stores z-scores; the stored mean and std undo the scaling.
    """
    his = np.load(_files("largest_sd")["dir"] / "his.npz")
    return his["data"][..., 0].astype(np.float64) * float(his["std"]) + float(his["mean"])


def test_targets(name):
    """True values of every test window, shape (windows, 12, sensors)."""
    if name != "largest_sd":
        return np.load(_files(name)["test"])["y"][..., 0].astype(np.float64)
    anchors = window_anchors(name)["test"]
    rows = anchors[:, None] + 1 + np.arange(TARGET_STEPS)
    return largest_sd_flow()[rows]


def sensor_coords(name):
    """(lat, lon) of every sensor, in data-column order."""
    files = _files(name)
    _, _, ids = load_series(name)
    ids = [int(s) for s in ids]
    if name == "largest_sd":
        meta = pd.read_csv(files["coords"]).set_index("ID")
        coords = meta.loc[ids, ["Lat", "Lng"]].values
    elif name == "metr_la":
        table = pd.read_csv(files["coords"]).set_index("sensor_id")
        coords = table.loc[ids, ["latitude", "longitude"]].values
    else:  # the PEMS-BAY file has no header
        table = pd.read_csv(files["coords"], header=None,
                            names=["sensor_id", "latitude", "longitude"]).set_index("sensor_id")
        coords = table.loc[ids, ["latitude", "longitude"]].values
    coords = np.asarray(coords, dtype=np.float64)
    assert coords.shape == (len(ids), 2) and not np.isnan(coords).any(), name
    return coords


def load_forecast(model, name, part="test"):
    """Saved forecasts of one model, shape (windows, 12, sensors).

    part="val" loads the validation forecasts (used by the router).
    Some trained models pad their last test batch, so extra rows at the
    end are cut. LargeST-SD runs of the trained graph models drop the last
    partial batch instead, so they can be a little shorter than the test
    set; callers use the common prefix of windows.
    """
    file = "y_pred.npy" if part == "test" else f"y_pred_{part}.npy"
    pred = np.load(FORECAST_DIR / model / name / file).astype(np.float64)
    pred = pred[:len(window_anchors(name)[part])]
    assert pred.shape[1] == TARGET_STEPS, (model, name, pred.shape)
    return pred
