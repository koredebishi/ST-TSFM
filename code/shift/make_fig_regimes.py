#!/usr/bin/env python3
"""F6 — S1 regime marginals (decisions.md D32): show the shift as physics.

One panel per dataset: the marginal distribution of the observable over
weekday-target test windows vs weekend-target test windows (the S1 shift,
the failing regime on LargeST-SD). Values come from the banked truth
arrays re-sliced with the same D8 pool rule as every result table; zeros
masked by the eval convention. Severity from severity_summary.json is
annotated directly (sensor-mean W1, Table I convention).

Style: imported from shift.make_figures (single shared style module).
Output: results/shift/figures/F6_s1_marginals.{pdf,png}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from shift.make_figures import (BLUE, DLABEL, FIG, INK, MUTED,  # noqa: E402
                                UNITS, plt)
from data_loaders.windows import frozen_anchors, load_raw  # noqa: E402
from eval.metrics import _ZERO_TOL  # noqa: E402
from scripts.verification.frozen_degradation_matrix import truth  # noqa: E402
from shift.splits import classify_targets, is_weekday, is_weekend  # noqa: E402

GRAY = "#9a988f"


def valid_values(yt: np.ndarray) -> np.ndarray:
    v = yt.ravel()
    return v[np.isfinite(v) & (np.abs(v) > _ZERO_TOL)]


def main():
    sev = json.loads(
        (REPO / "results/shift/severity_summary.json").read_text())["severities"]
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.6))
    for ax, ds in zip(axes, ("metr_la", "pems_bay", "largest_sd")):
        yt = truth(ds).astype(np.float32)
        anc = frozen_anchors(ds)["test"]
        assert len(anc) == yt.shape[0]
        _, ts, _ = load_raw(ds)
        wd = classify_targets(anc, np.asarray(is_weekday(ts)))
        we = classify_targets(anc, np.asarray(is_weekend(ts)))
        a = valid_values(yt[wd])
        b = valid_values(yt[we])
        hi = np.percentile(np.concatenate([a, b]), 99.5)
        bins = np.linspace(0, hi, 80)
        ax.hist(a, bins=bins, density=True, color=GRAY, alpha=0.55,
                label="weekday targets")
        ax.hist(b, bins=bins, density=True, histtype="step", lw=1.3,
                color=BLUE, label="weekend targets (S1)")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.32)
        w1 = sev[ds]["S1"]["severity_w1_mean"]
        ax.text(0.98, 0.94,
                f"{DLABEL[ds]}\n$W_1$ = {w1:.1f} {UNITS[ds]}",
                transform=ax.transAxes, ha="right", va="top", color=INK)
        ax.set_yticks([])
        ax.set_xlabel(f"{UNITS[ds]}", labelpad=1)
        for s in ("left",):
            ax.spines[s].set_visible(False)
    axes[0].legend(loc="upper left", handlelength=1.2)
    fig.tight_layout(h_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"F6_s1_marginals.{ext}")
    print(f"wrote {FIG}/F6_s1_marginals.pdf  job={os.environ.get('SLURM_JOB_ID')}")


if __name__ == "__main__":
    main()
