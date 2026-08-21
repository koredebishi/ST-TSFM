#!/usr/bin/env python3
"""Mental-model figure: ALL distribution slices superposed on one canvas.

One panel per dataset, five outline densities over the same x-axis:
weekday / weekend (S1), peak / off-peak (S2), and the S3 held-out region
(a slice of SPACE, all times). Intuition artifact for Fig-1 design — the
paper's panel uses the split-pool version (plot_distributions.py).
Output: results/shift/figures/{dataset}_all_slices.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_loaders.windows import load_raw  # noqa: E402
from shift.severity import _clean  # noqa: E402
from shift.splits import (is_offpeak, is_peak, is_weekday, is_weekend,  # noqa: E402
                          s3_split)

FIGDIR = REPO / "results/shift/figures"
UNITS = {"metr_la": "speed (mph)", "pems_bay": "speed (mph)",
         "largest_sd": "flow (veh/15min)"}


def outline(ax, values, label, color, bins):
    h, edges = np.histogram(values, bins=bins, density=True)
    ax.stairs(h, edges, label=label, color=color, linewidth=2.0)


def main():
    (FIGDIR / "all_slices_mental_model").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for ds in ("metr_la", "pems_bay", "largest_sd"):
        raw, ts, _ = load_raw(ds)
        sp3 = s3_split(ds)
        slices = {
            "weekday (S1 train)": raw[np.asarray(is_weekday(ts))].ravel(),
            "weekend (S1 test)": raw[np.asarray(is_weekend(ts))].ravel(),
            "peak hours (S2 train)": raw[np.asarray(is_peak(ts))].ravel(),
            "off-peak (S2 test)": raw[np.asarray(is_offpeak(ts))].ravel(),
            "held-out REGION (S3 test)": raw[:, sp3.test_sensors].ravel(),
        }
        cleaned = {}
        for k, v in slices.items():
            v = _clean(v)
            cleaned[k] = rng.choice(v, min(len(v), 400_000), replace=False)
        hi = np.percentile(np.concatenate(list(cleaned.values())), 99.5)
        bins = np.linspace(0, hi, 90)

        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for (k, v), c in zip(cleaned.items(), colors):
            outline(ax, v, k, c, bins)
        ax.set_xlabel(UNITS[ds])
        ax.set_ylabel("density (how often the road is at this value)")
        ax.set_title(f"{ds} — every shift slice on one canvas", fontsize=11)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = FIGDIR / "all_slices_mental_model" / f"{ds}_all_slices.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
