"""Shared access to the frozen evaluation windows (protocol §10a).

One source of truth for: raw series + timestamps, the train/val/test window
anchors, and sensor coordinates aligned to data columns. Both the shift
splits (D8) and future eval wrappers read from here so the frozen protocol
exists in exactly one place.

Anchor convention (both pipelines): `a` = the LAST input step. Input window
is rows [a-11, a]; target window is rows [a+1, a+12].

- METR-LA / PEMS-BAY: anchors reconstructed from the DCRNN
  generate_training_data arithmetic (70/10/20 chronological over T-23
  windows) and assert-aligned against processed/{train,val,test}.npz.
- LargeST-SD: idx_{train,val,test}.npy ARE the anchors (60/20/20).

NOTE: baselines/tsfm_zeroshot_eval.py currently duplicates the LA/BAY
anchor arithmetic; it is NOT refactored to import from here until the
in-flight sharded TimesFM jobs (D9) have finished running that file.
"""

from __future__ import annotations

from pathlib import Path

import os

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
# Sensor-coordinate CSVs ship with the DCRNN repository
# (github.com/liyaguang/DCRNN, data/sensor_graph/). Point DCRNN_DATA at
# a local clone; the default assumes it sits beside this release.
DCRNN_DATA = Path(os.environ.get("DCRNN_DATA",
                                 REPO.parent / "DCRNN/data/sensor_graph"))
SEQ_IN = 12
SEQ_OUT = 12

DATASETS = {
    "metr_la": dict(
        h5=REPO / "datasets/METR-LA/metr-la.h5",
        processed=REPO / "datasets/METR-LA/processed",
        locations=DCRNN_DATA / "graph_sensor_locations.csv",
        locations_has_header=True,
        units="mph", cadence="5min",
    ),
    "pems_bay": dict(
        h5=REPO / "datasets/PEMS-BAY/pems-bay.h5",
        processed=REPO / "datasets/PEMS-BAY/processed",
        locations=DCRNN_DATA / "graph_sensor_locations_bay.csv",
        locations_has_header=False,
        units="mph", cadence="5min",
    ),
    "largest_sd": dict(
        h5=REPO / "datasets/LargeST-SD/sd_his_2019.h5",
        dir=REPO / "datasets/LargeST-SD/2019",
        locations=REPO / "datasets/LargeST-SD/sd_meta.csv",
        units="vehicles_per_15min", cadence="15min",
    ),
}


def load_raw(name: str):
    """Returns (values (T, N) float64, timestamps DatetimeIndex, sensor_ids)."""
    cfg = DATASETS[name]
    df = pd.read_hdf(cfg["h5"])
    return df.values.astype(np.float64), df.index, list(df.columns)


def frozen_anchors(name: str) -> dict[str, np.ndarray]:
    """The protocol's train/val/test window anchors (last-input-step rows)."""
    cfg = DATASETS[name]
    if name == "largest_sd":
        return {split: np.load(cfg["dir"] / f"idx_{split}.npy").astype(int)
                for split in ("train", "val", "test")}

    df = pd.read_hdf(cfg["h5"])
    T = len(df)
    num_samples = T - SEQ_IN - SEQ_OUT + 1
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_train - num_test
    first = SEQ_IN - 1  # anchor of window 0
    return {
        "train": first + np.arange(num_train),
        "val": first + num_train + np.arange(num_val),
        "test": first + num_train + num_val + np.arange(num_test),
    }


def verify_alignment(name: str, split: str = "test", atol: float = 1e-3):
    """Assert reconstructed anchors reproduce the frozen npz targets.

    Compute-node only (loads real data). Raises AssertionError on mismatch.
    """
    cfg = DATASETS[name]
    raw, _, _ = load_raw(name)
    anchors = frozen_anchors(name)[split]
    if name == "largest_sd":
        npz = np.load(cfg["dir"] / "his.npz")
        ref = (npz["data"][..., 0].astype(np.float64) * float(npz["std"])
               + float(npz["mean"]))
        for k in (0, len(anchors) // 2, len(anchors) - 1):
            a = anchors[k]
            assert np.allclose(raw[a + 1 : a + 1 + SEQ_OUT],
                               ref[a + 1 : a + 1 + SEQ_OUT], atol=atol), \
                f"{name}/{split}: raw vs his.npz mismatch at anchor {a}"
        return
    y = np.load(cfg["processed"] / f"{split}.npz")["y"][..., 0]
    assert len(anchors) == len(y), \
        f"{name}/{split}: {len(anchors)} anchors vs {len(y)} npz windows"
    for k in (0, len(anchors) // 2, len(anchors) - 1):
        a = anchors[k]
        assert np.allclose(raw[a + 1 : a + 1 + SEQ_OUT], y[k], atol=atol), \
            f"{name}/{split}: anchor misalignment at window {k}"


def sensor_locations(name: str) -> np.ndarray:
    """(N, 2) array of (lat, lon), row-aligned to the data columns."""
    cfg = DATASETS[name]
    _, _, sensor_ids = load_raw(name)
    if name == "largest_sd":
        meta = pd.read_csv(cfg["locations"]).set_index("ID")
        coords = meta.loc[[int(s) for s in sensor_ids], ["Lat", "Lng"]].values
    else:
        if cfg["locations_has_header"]:
            loc = pd.read_csv(cfg["locations"]).set_index("sensor_id")
            loc = loc[["latitude", "longitude"]]
        else:
            loc = pd.read_csv(cfg["locations"], header=None,
                              names=["sensor_id", "latitude", "longitude"]
                              ).set_index("sensor_id")
        coords = loc.loc[[int(s) for s in sensor_ids]].values
    coords = np.asarray(coords, dtype=np.float64)
    assert coords.shape == (len(sensor_ids), 2), \
        f"{name}: coordinate lookup incomplete ({coords.shape})"
    assert not np.isnan(coords).any(), f"{name}: NaN coordinates"
    return coords
