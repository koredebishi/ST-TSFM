#!/usr/bin/env python3
"""F5 replacement (D34): severity vs degradation under three marginal rulers.

Three panels (W1 | KS | mean difference), 18 model-setting evaluations
each: x = within-dataset normalized severity under that ruler, y =
average relative degradation (%). Failing settings ringed. Spearman rho
and permutation p printed per panel (values from
results/shift/severity_rulers.json, computed by the same self-tested
spearman_perm as the original F5).

Style: imported from shift.make_figures (single shared style module).
Restyle 2026-08-05 (decision log D36 + addenda, variant C):
safe evaluations keep their dataset color desaturated (alpha 0.35,
s=14); the four failing points are full-saturation fill with a red
ring and direct SD-S1 / LA-S3 labels; backbone dodge +/-0.008; one
shared x-axis label and a horizontal legend on the bottom line instead
of per-panel xlabels and an in-panel legend box.
Output: results/shift/figures/F5_rulers.{pdf,png}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from shift.make_figures import (DLABEL, DSET_COLOR, FIG, INK,  # noqa: E402
                                MUTED, RED, plt)

DATASETS = ("metr_la", "pems_bay", "largest_sd")
SKEY = {"S1": "S1_weekday_to_weekend", "S2": "S2_peak_to_offpeak",
        "S3": "S3_spatial_holdout"}
TF_FILE = {"metr_la": "timesfm_metr_la_degradation.json",
           "pems_bay": "timesfm_bay_degradation.json",
           "largest_sd": "timesfm_largest_sd_degradation.json"}
FAILING = {("largest_sd", "S1"), ("metr_la", "S3")}
MARKER = {"chronos2": "o", "timesfm": "^"}
# tiny plotting dodge only: the two backbones share each setting's x
# (same pool severity), so stacked markers read as one darker blob.
X_OFFSET = {"chronos2": -0.008, "timesfm": 0.008}
RLABEL = {"w1": "Wasserstein-1", "ks": "Kolmogorov–Smirnov",
          "dmean": "mean difference"}


def main():
    R = json.loads((REPO / "results/shift/severity_rulers.json").read_text())
    sev, _ = R["severities"], R["correlations"]
    # D37 W6: unified per-sensor estimator. S3 severities come from the
    # E1b unification (each held-out sensor vs pooled retained marginal,
    # sensor mean) so all three panels share one functional family;
    # panel correlations come from E5b under the same variant.
    e1b = json.loads((REPO / "results/shift/e1b_unified_rulers.json"
                      ).read_text())["variants"]["per_sensor_s3"]
    e5b = json.loads((REPO / "results/shift/e5b_ruler_correlations.json"
                      ).read_text())["variants"]["per_sensor_s3"]
    sev = {ds: {s: dict(sev[ds][s]) for s in ("S1", "S2", "S3")}
           for ds in DATASETS}
    for ds in DATASETS:
        sev[ds]["S3"] = dict(e1b[ds]["S3"])
    corr = {r: {"spearman_rho": e5b[r]["rho_18"],
                "perm_p": e5b[r]["perm_p_18"]} for r in ("w1", "ks", "dmean")}
    fdm = json.loads((REPO / "results/shift/frozen_degradation_matrix.json"
                      ).read_text())["matrix"]
    degr = {}
    for ds in DATASETS:
        tf = json.loads((REPO / "results/shift" / TF_FILE[ds]).read_text())["cell"]
        for s in ("S1", "S2", "S3"):
            degr[("chronos2", ds, s)] = fdm[ds][SKEY[s]]["rel_degradation"]["avg"]
            degr[("timesfm", ds, s)] = tf[SKEY[s]]["rel_degradation"]["avg"]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    for ax, ruler in zip(axes, ("w1", "ks", "dmean")):
        fail_pts = {}
        for ds in DATASETS:
            mx = max(sev[ds][s][ruler] for s in ("S1", "S2", "S3"))
            for s in ("S1", "S2", "S3"):
                x = sev[ds][s][ruler] / mx
                for bb in ("chronos2", "timesfm"):
                    y = 100.0 * degr[(bb, ds, s)]
                    ring = (ds, s) in FAILING
                    xp = float(np.clip(x + X_OFFSET[bb], 0.0, 1.0))
                    ax.scatter(xp, y,
                               s=25 if ring else 14,
                               marker=MARKER[bb],
                               facecolor=DSET_COLOR[ds],
                               edgecolor=RED if ring else "none",
                               linewidth=1.4 if ring else 0.0,
                               alpha=1.0 if ring else 0.35,
                               zorder=4 if ring else 2)
                    if ring:
                        fail_pts.setdefault((ds, s), []).append((xp, y))
        # one direct label per failing setting (both backbones share its x)
        for (ds, s), pts in fail_pts.items():
            ax.annotate(f"{DLABEL[ds].split('-')[-1]}-{s}",
                        (pts[0][0], max(p[1] for p in pts)),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=6.5, color=INK)
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        c = corr[ruler]
        ax.set_title(RLABEL[ruler], color=INK)
        # W21: descriptive rho only; the 18 evaluations are not
        # independent (Section V-B), so the setting-level exact test
        # lives in the caption instead of an anticonservative panel p.
        ax.text(0.02, 0.05,
                rf"$\rho$ = {c['spearman_rho']:.2f}",
                transform=ax.transAxes, ha="left", va="bottom", color=INK)
        ax.grid(True, axis="y", zorder=0)
    axes[0].set_ylabel("avg. degradation (%)")
    # one bottom line: shared x-label, then the legend lying horizontally
    fig.tight_layout(w_pad=0.8, rect=(0, 0.10, 1, 1))
    fig.text(0.145, 0.045, "normalized severity", ha="center", va="center",
             color=INK)
    handles = [plt.Line2D([], [], marker="o", ls="", color=DSET_COLOR[ds],
                          label=DLABEL[ds]) for ds in DATASETS]
    handles += [plt.Line2D([], [], marker=MARKER[bb], ls="",
                           markerfacecolor="none", color=INK,
                           label={"chronos2": "Chronos-2",
                                  "timesfm": "TimesFM"}[bb])
                for bb in ("chronos2", "timesfm")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.62, 0.045),
               ncol=5, frameon=False, fontsize=6, handletextpad=0.2,
               columnspacing=0.7, handlelength=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"F5_rulers.{ext}")
    print(f"wrote {FIG}/F5_rulers.pdf job={os.environ.get('SLURM_JOB_ID')}")


if __name__ == "__main__":
    main()
