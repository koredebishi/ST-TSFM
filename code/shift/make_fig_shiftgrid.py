#!/usr/bin/env python3
"""F1 — the audit grid as Fig. 1: all nine shift pools as the severity
instrument sees them (successor to D32/F6, which it replaces).

Design and revision history recorded in the decision log; the
2026-08-04 (D32 addendum 3): GPT-audit structure (pool cache with one
x-range per dataset row, severity strip anchored to the panel bottom in
axes coordinates, thin red top rule for failing cells, dot-on-whisker
summaries) with four corrections kept from the internal audit:
(1) KDE pairs are normalized JOINTLY per cell (per-curve peak
    normalization destroys the within-cell shape comparison);
(2) failing-cell tags state what fail means in this figure,
    degradation, with the Table II Chronos-2 values; "RANK FLIP" was
    factually wrong for METR-LA S3, where Table IV has tau = 1.000;
(3) one scale bar per row so W1 and the whiskers are physically
    interpretable without tick axes;
(4) row labels name the observable (speed/flow), matching III-A.
The y=0.5 guide line from the GPT draft is omitted: with peak-normalized
curves it references nothing.

3x3 grid, rows = datasets, cols = S1/S2/S3. Each cell overlays a smoothed
marginal of the in-distribution pool (P_ID, filled) with the shifted pool
(P_S, line; red in the two failing settings, which carry a thin red top
rule and a degradation tag), adds mean +/- 1 sigma dot-whiskers for both
pools, and shows severity twice: the Table I W1 value and, beside it, a
strip of the same W1 normalized within dataset -- the quantity every
correlation in Section IV consumes.

Pools come from the banked truth arrays re-sliced with the same D8 pool
rule as every result table; zeros masked by the eval convention.
Output: results/shift/figures/F1_shiftgrid.{pdf,png}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from shift.make_figures import (BLUE, DLABEL, FIG, INK, RED,  # noqa: E402
                                UNITS, plt)
from data_loaders.windows import frozen_anchors, load_raw  # noqa: E402
from eval.metrics import _ZERO_TOL  # noqa: E402
from scripts.verification.frozen_degradation_matrix import truth  # noqa: E402
from shift.splits import (classify_targets, is_offpeak, is_peak,  # noqa: E402
                          is_weekday, is_weekend, s3_split)

GRAY = "#b9b7ae"
TRACK, BARFILL = "#e8e6df", "#6f6d66"
FAIL = {("largest_sd", "S1"), ("metr_la", "S3")}
# Degradation tags: Table II, Chronos-2 column, one backbone shown
# consistently (TimesFM reads +8% and +21% on these cells; on METR-LA S3
# TimesFM is the worse backbone).
FAIL_TAG = {("largest_sd", "S1"): "fails (+29%)",
            ("metr_la", "S3"): "fails (+19%)"}
OBS = {"metr_la": "speed", "pems_bay": "speed", "largest_sd": "flow"}
SHIFT_TITLES = {"S1": r"S1   weekday $\to$ weekend",
                "S2": r"S2   peak $\to$ off-peak",
                "S3": "S3   spatial holdout"}
KDE_SUBSAMPLE = 60_000  # per pool; plenty for a marginal, keeps KDE fast

DATASETS = ("metr_la", "pems_bay", "largest_sd")
SHIFTS = ("S1", "S2", "S3")


def valid_values(yt: np.ndarray) -> np.ndarray:
    v = yt.ravel()
    return v[np.isfinite(v) & (np.abs(v) > _ZERO_TOL)]


def pools(ds: str, shift: str, yt: np.ndarray, anc, ts):
    if shift == "S1":
        a = classify_targets(anc, np.asarray(is_weekday(ts)))
        b = classify_targets(anc, np.asarray(is_weekend(ts)))
        return valid_values(yt[a]), valid_values(yt[b])
    if shift == "S2":
        a = classify_targets(anc, np.asarray(is_peak(ts)))
        b = classify_targets(anc, np.asarray(is_offpeak(ts)))
        return valid_values(yt[a]), valid_values(yt[b])
    sp = s3_split(ds)
    keep = np.asarray(sp.train_sensors)
    hold = np.asarray(sp.test_sensors)
    # sensor axis of the (windows, horizon, sensors) truth arrays
    return valid_values(yt[:, :, keep]), valid_values(yt[:, :, hold])


def kde_raw(vals: np.ndarray, grid: np.ndarray, rng) -> np.ndarray:
    """Raw (unnormalized) KDE so a cell's pair can be scaled jointly."""
    v = vals if len(vals) <= KDE_SUBSAMPLE else rng.choice(
        vals, KDE_SUBSAMPLE, replace=False)
    return gaussian_kde(v)(grid)


def sev_value(sev: dict, ds: str, shift: str) -> float:
    e = sev[ds][shift]
    return e.get("severity_w1_mean", e.get("severity_w1_pooled"))


def tidy_bar_length(hi: float) -> float:
    """A round scale-bar length near a quarter of the row's x-range."""
    target = 0.25 * hi
    for step in (100, 50, 25, 20, 10, 5, 2, 1):
        if target >= step:
            return float(round(target / step) * step)
    return float(f"{target:.1g}")


def main():
    rng = np.random.default_rng(2023)
    sev = json.loads(
        (REPO / "results/shift/severity_summary.json").read_text()
    )["severities"]
    # D37 W-fig1: unified per-sensor estimator (decisions.md D37, e1
    # variant per_sensor_s3). S3's printed value and strip must match
    # Table I, so each S3 cell's severity is replaced by the per-sensor
    # unification; S1/S2 are already the per-sensor means.
    e1 = json.loads(
        (REPO / "results/shift/e1_unified_severity.json").read_text()
    )["variants"]["per_sensor_s3"]
    for _ds in ("metr_la", "pems_bay", "largest_sd"):
        sev[_ds]["S3"] = {"severity_w1_mean": e1[_ds]["raw"]["S3"]}

    # ---- pool cache: one x-range per dataset row (GPT audit, item 1) ----
    pool_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    row_hi: dict[str, float] = {}
    for ds in DATASETS:
        yt = truth(ds).astype(np.float32)
        anc = frozen_anchors(ds)["test"]
        assert len(anc) == yt.shape[0]
        _, ts, _ = load_raw(ds)
        row_values = []
        for shift in SHIFTS:
            a, b = pools(ds, shift, yt, anc, ts)
            pool_cache[(ds, shift)] = (a, b)
            row_values.extend([a, b])
        row_hi[ds] = float(np.percentile(np.concatenate(row_values), 99.5))

    fig, axes = plt.subplots(3, 3, figsize=(7.15, 2.95), sharey=True)

    for r, ds in enumerate(DATASETS):
        w1_row = [sev_value(sev, ds, s) for s in SHIFTS]
        row_max = max(w1_row)
        hi = row_hi[ds]

        for c, shift in enumerate(SHIFTS):
            ax = axes[r, c]
            fail = (ds, shift) in FAIL
            a, b = pool_cache[(ds, shift)]
            grid = np.linspace(0, hi, 320)

            # ---- joint normalization per cell (correction 1) ----
            fa = kde_raw(a, grid, rng)
            fb = kde_raw(b, grid, rng)
            top = max(fa.max(), fb.max())
            f_id, f_shift = fa / top, fb / top

            shifted_color = RED if fail else BLUE

            # quiet structural baseline (no y=0.5 guide; see docstring)
            ax.axhline(0, color="0.86", lw=0.55, zorder=0)

            # ID distribution: fill plus faint outline
            ax.fill_between(grid, f_id, color=GRAY, alpha=0.48, lw=0,
                            zorder=1)
            ax.plot(grid, f_id, color="0.50", lw=0.65, zorder=2)

            # shifted distribution
            ax.plot(grid, f_shift, color=shifted_color, lw=1.45, zorder=3)

            # mean +/- std dot-whiskers (data x, axes y)
            for vals, col, y_ax in [(a, "0.42", 0.070),
                                    (b, shifted_color, 0.015)]:
                m, s = float(np.mean(vals)), float(np.std(vals))
                ax.plot([m - s, m + s], [y_ax, y_ax],
                        transform=ax.get_xaxis_transform(),
                        color=col, lw=0.95, solid_capstyle="round",
                        clip_on=False, zorder=4)
                ax.scatter([m], [y_ax],
                           transform=ax.get_xaxis_transform(),
                           s=8, color=col, edgecolors="white",
                           linewidths=0.35, clip_on=False, zorder=5)

            # ---- severity strip, bottom-anchored in axes coords ----
            sn = w1_row[c] / row_max
            track_y, track_x0, track_x1 = -0.17, 0.04, 0.59
            ax.plot([track_x0, track_x1], [track_y, track_y],
                    transform=ax.transAxes, color=TRACK, lw=3.0,
                    solid_capstyle="round", clip_on=False)
            ax.plot([track_x0, track_x0 + (track_x1 - track_x0) * sn],
                    [track_y, track_y],
                    transform=ax.transAxes, color=BARFILL, lw=3.0,
                    solid_capstyle="round", clip_on=False)
            ax.text(track_x1 + 0.04, track_y,
                    rf"$W_1={w1_row[c]:.1f}$",
                    transform=ax.transAxes, fontsize=5.9, ha="left",
                    va="center", color=INK, clip_on=False)

            # ---- restrained failure marker (correction 2) ----
            if fail:
                ax.plot([0.02, 0.98], [1.075, 1.075],
                        transform=ax.transAxes, color=RED, lw=1.45,
                        solid_capstyle="round", clip_on=False)
                ax.text(0.98, 1.105, FAIL_TAG[(ds, shift)],
                        transform=ax.transAxes, fontsize=5.4,
                        fontweight="bold", ha="right", va="bottom",
                        color=RED, clip_on=False)

            ax.set_xlim(0, hi)
            ax.set_ylim(-0.02, 1.08)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if r == 0:
                ax.set_title(SHIFT_TITLES[shift], fontsize=7.0, color=INK,
                             pad=12, fontweight="semibold")
            if c == 0:
                ax.text(-0.085, 0.52,
                        f"{DLABEL[ds]}\n{OBS[ds]}, {UNITS[ds]}",
                        transform=ax.transAxes, ha="right", va="center",
                        fontsize=6.5, color=INK, linespacing=1.45)

        # ---- one scale bar per row, rightmost cell (correction 3) ----
        sb = tidy_bar_length(hi)
        ax_last = axes[r, 2]
        x1 = hi * 0.97
        x0 = x1 - sb
        ybar = -0.31  # its own band below the severity strip (collision fix)
        ax_last.plot([x0, x1], [ybar, ybar],
                     transform=ax_last.get_xaxis_transform(),
                     color=INK, lw=0.8, solid_capstyle="butt",
                     clip_on=False)
        for xe in (x0, x1):
            ax_last.plot([xe, xe], [ybar - 0.025, ybar + 0.025],
                         transform=ax_last.get_xaxis_transform(),
                         color=INK, lw=0.8, clip_on=False)
        ax_last.text(x0 - 0.02 * hi, ybar,
                     f"{sb:g} {UNITS[ds]}",
                     transform=ax_last.get_xaxis_transform(),
                     ha="right", va="center", fontsize=5.3, color=INK,
                     clip_on=False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=GRAY, alpha=0.48, ec="0.50",
                      lw=0.5),
        plt.Line2D([], [], c=BLUE, lw=1.45),
        plt.Line2D([], [], c=RED, lw=1.45),
    ]
    fig.legend(handles,
               [r"in-distribution pool $\mathcal{P}_{\mathrm{ID}}$",
                r"shifted pool $\mathcal{P}_S$",
                "shifted pool, failing setting"],
               fontsize=6.0, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.56, 0.005), handletextpad=0.55,
               columnspacing=1.45)

    fig.subplots_adjust(left=0.14, right=0.992, top=0.85, bottom=0.19,
                        hspace=0.58, wspace=0.10)

    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"F1_shiftgrid.{ext}", dpi=400)
    print(f"wrote {FIG}/F1_shiftgrid.pdf  "
          f"job={os.environ.get('SLURM_JOB_ID')}")


if __name__ == "__main__":
    main()
