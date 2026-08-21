#!/usr/bin/env python3
"""Distribution-shift visualizations (Dish-TS/RevIN Fig-1 style) + D11 evidence.

Outputs to results/shift/figures/:
- {dataset}_s1_dist.png — weekday vs weekend marginal densities (observable)
- {dataset}_s2_dist.png — peak vs off-peak densities
- pems_bay_s3_clusters.png — per-cluster density vs rest (D11 evidence)
And results/shift/bay_cluster_severity.json — pooled W1 of EACH BAY k-means
cluster vs the rest, with the §15 tripwire (0.5 x S1) marked, so the D11
reselection (option A) is chosen from a full table, not a single draw.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_loaders.windows import frozen_anchors, load_raw, sensor_locations  # noqa: E402
from shift.severity import _clean, _target_rows, w1  # noqa: E402
from shift.splits import (KMEANS_K, KMEANS_SEED, is_offpeak, is_peak,  # noqa: E402
                          is_weekday, is_weekend, s1_split, s2_split)

FIGDIR = REPO / "results/shift/figures"
UNITS = {"metr_la": "speed (mph)", "pems_bay": "speed (mph)",
         "largest_sd": "flow (veh/15min)"}


def density_plot(a, b, labels, title, units, path, w1_val):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bins = np.histogram_bin_edges(np.concatenate([a, b]), bins=80)
    ax.hist(a, bins=bins, density=True, alpha=0.55, label=labels[0])
    ax.hist(b, bins=bins, density=True, alpha=0.55, label=labels[1])
    ax.set_xlabel(units)
    ax.set_ylabel("density")
    ax.set_title(f"{title}  (W1 = {w1_val:.2f})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {path}", flush=True)


def main():
    (FIGDIR / "by_shift_pairs").mkdir(parents=True, exist_ok=True)
    (FIGDIR / "d11_bay_s3_evidence").mkdir(parents=True, exist_ok=True)

    for ds in ("metr_la", "pems_bay", "largest_sd"):
        raw, ts, _ = load_raw(ds)
        for tag, split_fn, preds in (
            ("s1", s1_split, (is_weekday, is_weekend, "weekday", "weekend")),
            ("s2", s2_split, (is_peak, is_offpeak, "peak", "off-peak")),
        ):
            sp = split_fn(ds)
            a = _clean(raw[_target_rows(sp.train_anchors)].ravel())
            b = _clean(raw[_target_rows(sp.test_anchors)].ravel())
            # subsample for plotting speed; W1 on full arrays
            rng = np.random.default_rng(0)
            density_plot(
                rng.choice(a, min(len(a), 300_000), replace=False),
                rng.choice(b, min(len(b), 300_000), replace=False),
                (preds[2], preds[3]),
                f"{ds} — {preds[2]} vs {preds[3]}", UNITS[ds],
                FIGDIR / "by_shift_pairs" / f"{ds}_{tag}_dist.png",
                w1(a, b))

    # ---- D11 evidence: every BAY cluster vs rest --------------------------
    ds = "pems_bay"
    raw, _, _ = load_raw(ds)
    coords = sensor_locations(ds)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=KMEANS_K, random_state=KMEANS_SEED, n_init=10
                ).fit(coords)
    labels = km.labels_
    rows = _target_rows(frozen_anchors(ds)["test"])
    s1_sev = 4.234  # job 46164821, severity_summary.json
    tripwire = 0.5 * s1_sev

    table = []
    fig, axes = plt.subplots(1, KMEANS_K, figsize=(16, 3), sharey=True)
    for c in range(KMEANS_K):
        in_c = np.where(labels == c)[0]
        out_c = np.where(labels != c)[0]
        a = _clean(raw[np.ix_(rows, out_c)].ravel())
        b = _clean(raw[np.ix_(rows, in_c)].ravel())
        sev = w1(a, b)
        tight = float(np.linalg.norm(
            coords[in_c] - km.cluster_centers_[c], axis=1).mean())
        table.append({"cluster": c, "n_sensors": int(len(in_c)),
                      "tightness_deg": round(tight, 6),
                      "severity_w1_pooled": round(sev, 4),
                      "passes_tripwire": bool(sev > tripwire)})
        rng = np.random.default_rng(0)
        ax = axes[c]
        bins = np.linspace(0, 80, 60)
        ax.hist(rng.choice(a, 200_000, replace=False), bins=bins,
                density=True, alpha=0.55, label="rest")
        ax.hist(rng.choice(b, min(len(b), 200_000), replace=False), bins=bins,
                density=True, alpha=0.55, label=f"cluster {c}")
        ax.set_title(f"c{c}: n={len(in_c)}, W1={sev:.2f}"
                     f"{' PASS' if sev > tripwire else ' fail'}", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"PEMS-BAY S3 candidates — tripwire W1 > {tripwire:.2f} "
                 f"(= 0.5 x S1 {s1_sev})", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "d11_bay_s3_evidence" / "pems_bay_s3_clusters.png",
                dpi=150)
    plt.close(fig)

    out = {"host": os.uname().nodename, "s1_severity": s1_sev,
           "tripwire": round(tripwire, 4),
           "kmeans": {"k": KMEANS_K, "seed": KMEANS_SEED},
           "clusters": sorted(table, key=lambda r: r["tightness_deg"])}
    (REPO / "results/shift/bay_cluster_severity.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
