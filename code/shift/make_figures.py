#!/usr/bin/env python3
"""BigData paper figures, all from BANKED artifacts (no recompute).

v2 (2026-07-17): publication style pass.
  - One shared style block (palette, hairline grid, no in-figure conclusion
    titles -- captions carry conclusions; identity labels stay).
  - F1 heatmap RETIRED -> T1_severity.tex (booktabs table; the old viridis
    heatmap had 6/9 cells visually identical because SD dominates the scale).
  - F3 + F6 MERGED -> F3_sd_s1_case (they shared the same axes/cell; the
    merged figure is the SD-S1 case study: horizon panel + transfer-failure
    diagnostic bars).
  - F5: severity axis normalized WITHIN dataset (W1 / dataset max) because
    raw W1 units differ across datasets (mph vs veh/15min) and cannot share
    one axis; pooled Spearman rho + permutation p printed on the panel.
    Presentation-level stat from banked numbers only, seed pinned.
  - F2: cells with positive degradation (the finding cells) get an ink
    outline; diverging colormap = palette blue<->red with NEUTRAL midpoint.

Reads results/shift/*.json (+ t2_table.md numbers pinned here with their
job-ID provenance) and writes PDFs+PNGs (+ .tex) to results/shift/figures/.
CPU only, matplotlib Agg.

Outputs:
  T1_severity.tex           severity table (replaces F1)
  F2_degradation_matrix     2 backbones x 3 datasets x 3 shifts [MONEY]
  F3_sd_s1_case             SD-S1 case study: horizon + TTRA diagnostic
  F4_ranking_slopes         ID->shift slope charts + tau
  F5_severity_vs_degradation  the perp finding [INSIGHT]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]
SH = REPO / "results/shift"
FIG = SH / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DATASETS = ["metr_la", "pems_bay", "largest_sd"]
DLABEL = {"metr_la": "METR-LA", "pems_bay": "PEMS-BAY", "largest_sd": "LargeST-SD"}
UNITS = {"metr_la": "mph", "pems_bay": "mph", "largest_sd": "veh/15 min"}
SHIFTS = ["S1", "S2", "S3"]
SKEY = {"S1": "S1_weekday_to_weekend", "S2": "S2_peak_to_offpeak",
        "S3": "S3_spatial_holdout"}

# ---- shared style (dataviz-skill palette, print instance) -------------
INK, MUTED, GRIDC, AXISC = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, GREEN, ORANGE, RED = "#2a78d6", "#008300", "#eb6834", "#e34948"
NEUTRAL, BLUE_DEEP, RED_DEEP = "#f0efec", "#1c5cab", "#d03b3b"
DIVERGING = LinearSegmentedColormap.from_list("bd", [BLUE_DEEP, NEUTRAL, RED_DEEP])
DSET_COLOR = {"metr_la": BLUE, "pems_bay": GREEN, "largest_sd": ORANGE}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.labelcolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRIDC, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "legend.frameon": False,
    "savefig.dpi": 300,
})

SINGLE_COL, DOUBLE_COL = 3.5, 7.16  # IEEE column widths (inches)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[fig] wrote {name}.pdf/.png", flush=True)


def load(name):
    return json.loads((SH / name).read_text())


def sev_of(entry):
    """W1 severity of a shift entry. S1/S2 use per-sensor mean; S3 (spatial
    holdout) has no marginal split so it stores a single pooled distance."""
    return entry.get("severity_w1_mean", entry.get("severity_w1_pooled"))


# ---- Spearman with permutation p (self-tested; banked numbers only) ---
def _rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(len(a), float)
    r = np.arange(1, len(a) + 1, dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = r[i:j + 1].mean()
        i = j + 1
    return ranks


def spearman_perm(x, y, n_perm=20000, seed=2023):
    rx, ry = _rankdata(x), _rankdata(y)

    def _corr(u, v):
        u = u - u.mean(); v = v - v.mean()
        return float(u @ v / np.sqrt((u @ u) * (v @ v)))

    rho = _corr(rx, ry)
    rng = np.random.default_rng(seed)
    hits = sum(abs(_corr(rx, rng.permutation(ry))) >= abs(rho) - 1e-12
               for _ in range(n_perm))
    return rho, (hits + 1) / (n_perm + 1)


assert _rankdata([10., 20., 20., 30.]).tolist() == [1.0, 2.5, 2.5, 4.0]
assert abs(spearman_perm([1, 2, 3, 4], [10, 20, 30, 40], n_perm=200)[0] - 1.0) < 1e-12


# ---------------------------------------------------------------- T1 (was F1)
def t1_severity_table():
    s = load("severity_summary.json")["severities"]
    rows = []
    for d in DATASETS:
        vals = " & ".join(f"{sev_of(s[d][sh]):.1f}" for sh in SHIFTS)
        rows.append(f"{DLABEL[d]} & {UNITS[d]} & {vals} \\\\")
    body = "\n".join(rows)
    tex = (
        "% Auto-generated by shift/make_figures.py from severity_summary.json.\n"
        "% S1/S2 = per-sensor mean W1; S3 (spatial holdout) = pooled W1 --\n"
        "% different estimand, marked with \\dag in the header.\n"
        "\\begin{tabular}{llccc}\n\\toprule\n"
        "Dataset & Units & S1 & S2 & S3$^{\\dag}$ \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )
    (FIG / "T1_severity.tex").write_text(tex)
    print("[fig] wrote T1_severity.tex (replaces F1 heatmap)", flush=True)


# ---------------------------------------------------------------- F2
def _degr_matrix():
    """Return dict[backbone][dataset][shift] = rel avg degradation (fraction)."""
    c2 = load("frozen_degradation_matrix.json")["matrix"]
    out = {"Chronos-2": {}, "TimesFM": {}}
    for d in DATASETS:
        out["Chronos-2"][d] = {sh: c2[d][SKEY[sh]]["rel_degradation"]["avg"]
                               for sh in SHIFTS}
    tfm = {d: load(f"timesfm_{'bay' if d == 'pems_bay' else d}_degradation.json")
           for d in DATASETS}
    for d in DATASETS:
        out["TimesFM"][d] = {sh: tfm[d]["cell"][SKEY[sh]]["rel_degradation"]["avg"]
                             for sh in SHIFTS}
    return out


def f2_degradation():
    D = _degr_matrix()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.35), sharey=True)
    vmax = max(abs(v) for bb in D.values() for dd in bb.values()
               for v in dd.values()) * 100
    for ax, bb in zip(axes, ["Chronos-2", "TimesFM"]):
        M = np.array([[D[bb][d][sh] * 100 for sh in SHIFTS] for d in DATASETS])
        im = ax.imshow(M, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(SHIFTS)
        ax.set_yticks(range(3)); ax.set_yticklabels([DLABEL[d] for d in DATASETS])
        ax.set_title(f"{bb} (frozen, zero-shot)", fontsize=8, pad=4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        for i in range(3):
            for j in range(3):
                v = M[i, j]
                ax.text(j, i, f"{v:+.0f}%", ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(v) > 0.55 * vmax else INK)
                if v > 0:  # a finding cell: degradation under shift
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor=INK, lw=1.4))
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("$\\Delta$ MAE under shift (%)   ($+$ = degrades)", fontsize=7.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2, color=MUTED)
    _save(fig, "F2_degradation_matrix")


# ---------------------------------------------------------------- F3 (merged F3+F6)
# Panel a: SD-S1 MAE vs horizon. frozen/adapted/in-dist from banked
# results/ttra/sd_s1/w6_gate.json (job 46749604); GWNet retrained per-horizon
# pinned from t2_table.md (job IDs in that file).
# Panel b: 2x2 pinball diagnostic from w6_diag_pinball.json (job 46781042).
RETRAINED_H = {  # (h3, h6, h12) avg-MAE on the shift TEST pool, GWNet
    "S1": {"metr_la": (2.54, 3.08, 3.84), "pems_bay": (0.94, 1.14, 1.44),
           "largest_sd": (16.89, 23.04, 35.16)},
}


def f3_sd_s1_case():
    g = json.loads((REPO / "results/ttra/sd_s1/w6_gate.json").read_text())
    dia = json.loads((REPO / "results/ttra/sd_s1/w6_diag_pinball.json").read_text())
    hs = [3, 6, 12]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL, 2.55), gridspec_kw={"width_ratios": [1.5, 1]})

    series = [  # (label, values, color, ls, marker) -- legend order = visual
        ("TTRA-adapted",         # order at h=12, top to bottom
         [g["adapted"]["per_horizon"][str(h)] for h in hs],
         ORANGE, "-.", "^"),  # color law: adapted=trained-side -> orange (color law, 2026-07-22)
        ("frozen (weekend)",
         [g["frozen_shift"]["per_horizon"][str(h)] for h in hs],
         BLUE, "-", "o"),
        ("GWNet retrained",
         RETRAINED_H["S1"]["largest_sd"], GREEN, "--", "s"),
        ("frozen (weekday)",
         [g["frozen_in_distribution"]["per_horizon"][str(h)] for h in hs],
         MUTED, ":", "o"),
    ]
    for label, vals, color, ls, mk in series:
        ax1.plot(hs, vals, ls, marker=mk, color=color, lw=1.4, ms=4.5,
                 markerfacecolor=color, markeredgecolor="white",
                 markeredgewidth=0.5, label=label)
    ax1.set_xticks(hs)
    ax1.set_xlabel("horizon (steps: 45 / 90 / 180 min)")
    ax1.set_ylabel("MAE (veh / 15 min)")
    ax1.grid(axis="y")
    ax1.set_xlim(2.4, 12.6)
    ax1.legend(loc="upper left", handlelength=1.9, labelspacing=0.35,
               borderaxespad=0.1)
    ax1.text(0.0, 1.04, "(a)", transform=ax1.transAxes, fontsize=8,
             fontweight="bold", va="bottom")

    labels = ["val\n(weekday)", "test\n(weekend)"]
    frozen = [dia["frozen"]["val_weekday"]["pinball"],
              dia["frozen"]["test_weekend"]["pinball"]]
    adapt = [dia["adapted"]["val_weekday"]["pinball"],
             dia["adapted"]["test_weekend"]["pinball"]]
    x = np.arange(2); w = 0.34
    ax2.bar(x - w / 2 - 0.01, frozen, w, label="frozen", color=BLUE)
    ax2.bar(x + w / 2 + 0.01, adapt, w, label="TTRA-adapted", color=ORANGE)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("pinball loss (train objective)")
    ax2.grid(axis="y")
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper left", handlelength=1.0, handleheight=1.0)
    for xi, (fr, ad) in enumerate(zip(frozen, adapt)):
        d = (ad - fr) / fr * 100
        ax2.text(xi, max(fr, ad) * 1.02, f"{d:+.1f}%", ha="center",
                 fontsize=7.5, color=INK)
    ax2.set_ylim(0, max(max(frozen), max(adapt)) * 1.16)
    ax2.text(0.0, 1.04, "(b)", transform=ax2.transAxes, fontsize=8,
             fontweight="bold", va="bottom")
    fig.subplots_adjust(wspace=0.42)
    _save(fig, "F3_sd_s1_case")


# ---------------------------------------------------------------- F4
def f4_ranking():
    # D36 redesign: LargeST-SD S1 only, column width, no end markers.
    # The other datasets' tau lives in the ranking-stability table and
    # the METR-LA single-pair inversions in Finding 2 prose. Labels stay
    # on both sides: the two orderings are the figure's content and the
    # weekday ordering appears in no table.
    mlabel = {"ha": "HA", "arima": "ARIMA", "dcrnn": "DCRNN",
              "gwnet": "GWNet", "staeformer": "STAEformer",
              "chronos2": "Chronos-2", "timesfm": "TimesFM"}
    b = load("t3_rankings.json")["datasets"]["largest_sd"]["S1"]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    idr = b["models_ranked_id"]; shr = b["models_ranked_shift"]
    y0 = {m: i for i, m in enumerate(idr)}
    y1 = {m: i for i, m in enumerate(shr)}
    for m in idr:
        moved = y0[m] != y1[m]
        ax.plot([0, 1], [y0[m], y1[m]], "-",
                lw=1.8 if moved else 0.9,
                color=RED if moved else AXISC,
                zorder=3 if moved else 2)
        ax.text(-0.06, y0[m], mlabel.get(m, m), ha="right", va="center",
                fontsize=7, color=INK if moved else "#52514e")
        ax.text(1.06, y1[m], mlabel.get(m, m), ha="left", va="center",
                fontsize=7, color=INK if moved else "#52514e")
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(len(idr) - 0.5, -0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["weekday", "weekend"])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    lo, hi = b["ci_low"], b["ci_high"]
    tau_line = f"$\\tau$ = {b['tau']:.3f}"
    if hi - lo > 5e-4:  # bootstrap CI informative; degenerate CIs omitted
        tau_line += f"  [{lo:.2f}, {hi:.2f}]"
    ax.set_title(f"LargeST-SD weekday$\\to$weekend\n{tau_line}",
                 fontsize=8, pad=3)
    _save(fig, "F4_ranking_slopes")


# ---------------------------------------------------------------- F5
def f5_scatter():
    sev = load("severity_summary.json")["severities"]
    D = _degr_matrix()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.9))
    marks = {"Chronos-2": "o", "TimesFM": "^"}
    xs, ys = [], []
    # severity normalized within dataset: raw W1 units differ across
    # datasets (mph vs veh/15min) and cannot share one axis.
    smax = {d: max(sev_of(sev[d][sh]) for sh in SHIFTS) for d in DATASETS}
    for bb in ["Chronos-2", "TimesFM"]:
        for d in DATASETS:
            for sh in SHIFTS:
                x = sev_of(sev[d][sh]) / smax[d]
                y = D[bb][d][sh] * 100
                xs.append(x); ys.append(y)
                ax.scatter(x, y, marker=marks[bb], color=DSET_COLOR[d],
                           s=32, edgecolor="white", lw=0.6, zorder=3)
                if y > 0 and bb == "Chronos-2":  # label the finding cells once
                    ax.annotate(f"{DLABEL[d].split('-')[-1]}-{sh}", (x, y),
                                xytext=(7, -2), textcoords="offset points",
                                fontsize=6.5, color="#52514e")
    rho, p = spearman_perm(np.array(xs), np.array(ys))
    ax.axhline(0, color=AXISC, lw=0.8, zorder=1)
    ax.set_xlabel("shift severity, normalized ($W_1$ / dataset max)")
    ax.set_ylabel("$\\Delta$ MAE under shift (%)")
    ax.grid(axis="y")
    if p < 0.001:
        p_txt = "p < 0.001"
    elif p < 0.01:
        p_txt = f"p = {p:.3f}"
    else:
        p_txt = f"p = {p:.2f}"
    ax.text(0.97, 0.52,
            f"Spearman $\\rho$ = {rho:.2f}, {p_txt} (n = {len(xs)})",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color=INK)
    handles = ([Line2D([], [], marker="o", color=DSET_COLOR[d], ls="",
                       ms=5, label=DLABEL[d]) for d in DATASETS] +
               [Line2D([], [], marker=marks[bb], markerfacecolor="none",
                       markeredgecolor=INK, ls="", ms=5, label=bb)
                for bb in ["Chronos-2", "TimesFM"]])
    ax.legend(handles=handles, loc="lower left", fontsize=6.5,
              handletextpad=0.3, borderaxespad=0.2, labelspacing=0.3)
    _save(fig, "F5_severity_vs_degradation")


# ================================================================ G1
# In-distribution MAE dot chart (replaces Table II / tab:t1).
# Data: results/shift/t1_grid.json (job 47153165). Per model x dataset:
# h3/h6/h12/avg MAE. Trained graph = orange, classical = gray, frozen = blue.
GRAPH = {"dcrnn", "gwnet", "staeformer"}
CLASSICAL = {"ha", "arima"}
FROZEN = {"chronos2", "timesfm"}
MLABEL = {"ha": "HA", "arima": "ARIMA", "dcrnn": "DCRNN", "gwnet": "Graph WaveNet",
          "staeformer": "STAEformer", "chronos2": "Chronos-2", "timesfm": "TimesFM"}


def _mcolor(m):
    if m in GRAPH:
        return ORANGE
    if m in FROZEN:
        return BLUE
    return MUTED


def g1_indist():
    g = load("t1_grid.json")
    models = ["ha", "arima", "dcrnn", "gwnet", "staeformer", "chronos2", "timesfm"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.55))
    for ax, d in zip(axes, DATASETS):
        # order by avg MAE (best at top); scoped-out (None) sink to bottom
        present = [(m, g[m][d]) for m in models if g[m][d] is not None]
        present.sort(key=lambda kv: kv[1]["avg"])
        absent = [m for m in models if g[m][d] is None]
        order = [m for m, _ in present] + absent
        ymap = {m: len(order) - 1 - i for i, m in enumerate(order)}  # best on top
        xmax = max(c["h12"] for _, c in present)
        for m in order:
            c = g[m][d]
            y = ymap[m]
            if c is None:
                ax.text(0.02, y, "scoped out", transform=ax.get_yaxis_transform(),
                        va="center", ha="left", fontsize=6.5, style="italic",
                        color=MUTED)
                continue
            col = _mcolor(m)
            lo, hi, av = c["h3"], c["h12"], c["avg"]
            ax.plot([lo, hi], [y, y], "-", color=col, lw=1.3, alpha=0.5,
                    solid_capstyle="round", zorder=2)
            ax.scatter(av, y, color=col, s=26, edgecolor="white", lw=0.6, zorder=3)
            # SD/veh values are integers-ish; keep 2 decimals for mph, 1 for veh
            fmt = f"{av:.1f}" if d == "largest_sd" else f"{av:.2f}"
            ax.annotate(fmt, (av, y), xytext=(0, 4.5),
                        textcoords="offset points", ha="center", fontsize=6,
                        color=INK)
        ax.set_yticks([ymap[m] for m in order])
        ax.set_yticklabels([MLABEL[m] for m in order], fontsize=6.8)
        ax.set_title(DLABEL[d], fontsize=8, pad=3)
        ax.set_xlabel(f"MAE ({UNITS[d]})", fontsize=7)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_xlim(0, xmax * 1.16)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.tick_params(length=0)
    _save(fig, "G1_indist_mae")


# ================================================================ G2
# Degradation vs horizon: the failing curves rising out of the safe band.
# Data: frozen_degradation_matrix.json + timesfm_*_degradation.json,
# per_horizon rel_degradation (h3/h6/h12) x 18 model-setting curves.
def _perhz(bb, d, sh):
    if bb == "Chronos-2":
        c = load("frozen_degradation_matrix.json")["matrix"][d][SKEY[sh]]
    else:
        tfm = load(f"timesfm_{'bay' if d == 'pems_bay' else d}_degradation.json")
        c = tfm["cell"][SKEY[sh]]
    r = c["rel_degradation"]
    return [r["h3"] * 100, r["h6"] * 100, r["h12"] * 100]


def g2_deg_horizon():
    hs = [3, 6, 12]
    failing = [("largest_sd", "S1"), ("metr_la", "S3")]  # the 2 failing settings
    fail_keys = set(failing)
    safe = []
    for bb in ["Chronos-2", "TimesFM"]:
        for d in DATASETS:
            for sh in SHIFTS:
                if (d, sh) not in fail_keys:
                    safe.append(_perhz(bb, d, sh))
    safe = np.array(safe)  # 14 x 3
    assert safe.shape[0] == 14, f"expected 14 safe curves, got {safe.shape[0]}"
    lo, hi = safe.min(axis=0), safe.max(axis=0)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.7))
    ax.fill_between(hs, lo, hi, color=MUTED, alpha=0.18, lw=0, zorder=1)
    ax.plot(hs, [0, 0, 0], "-", color=AXISC, lw=0.8, zorder=1)
    ax.text(12, (lo[-1] + hi[-1]) / 2, "  safe band\n  (14 curves)", fontsize=6.3,
            color=MUTED, va="center", ha="left")
    hue = {"largest_sd": ORANGE, "metr_la": BLUE}
    ls = {"Chronos-2": "-", "TimesFM": "--"}
    mk = {"Chronos-2": "o", "TimesFM": "^"}
    for d, sh in failing:
        for bb in ["Chronos-2", "TimesFM"]:
            ys = _perhz(bb, d, sh)
            ax.plot(hs, ys, ls[bb], marker=mk[bb], color=hue[d], lw=1.5, ms=4.5,
                    markerfacecolor=hue[d], markeredgecolor="white",
                    markeredgewidth=0.5, zorder=3)
        yend = _perhz("Chronos-2", d, sh)[-1]
        lab = f"{DLABEL[d].split('-')[-1] if '-' in DLABEL[d] else DLABEL[d]}-{sh}"
        ax.annotate(f"{lab}", (12, yend), xytext=(6, 2),
                    textcoords="offset points", fontsize=7, color=hue[d],
                    fontweight="bold", va="center")
    ax.set_xticks(hs)
    ax.set_xlim(2.5, 13.6)
    ax.set_xlabel("horizon (steps: h3 / h6 / h12)")
    ax.set_ylabel("$\\Delta$ MAE under shift (%)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    hdl = [Line2D([], [], marker="o", color=INK, ls="-", ms=4, label="Chronos-2"),
           Line2D([], [], marker="^", color=INK, ls="--", ms=4, label="TimesFM")]
    ax.legend(handles=hdl, loc="upper left", fontsize=6.5, handlelength=1.8,
              borderaxespad=0.2)
    _save(fig, "G2_degradation_horizon")


# ================================================================ G3
# Split-sensitivity: tau across peak-window variants (S2) and k values (S3).
# Data: gap2_split_sensitivity.json (S2_peak_variants, S3_k_variants).
def g3_split_sensitivity():
    j = load("gap2_split_sensitivity.json")
    pk = j["S2_peak_variants"]
    kv = j["S3_k_variants"]
    pk_names = list(pk.keys())
    pk_short = ["base", "wide", "narrow", "shifted"][:len(pk_names)]
    k_names = list(kv.keys())
    k_x = [int(s.split("=")[1]) for s in k_names]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.35),
                                 gridspec_kw={"width_ratios": [1.1, 1]})
    def _tau(cell):
        # cell is a float (older format), or {'tau':..}, or {'error':..}
        # for a failed cell, or None. Missing/failed -> None (skipped).
        if cell is None:
            return None
        if isinstance(cell, dict):
            return cell.get("tau")  # None if the cell errored
        return cell

    # left: S2 peak-window variants
    # D36: PEMS-BAY dashed and drawn last so it stays visible where its
    # tau coincides with another dataset (right panel, all k at 1.0).
    _ls = {"pems_bay": (0, (3, 1.5))}
    _zo = {"pems_bay": 4}
    for d in DATASETS:
        ys = [_tau(pk[v].get(d)) for v in pk_names]
        xs = [i for i, y in enumerate(ys) if y is not None]
        yy = [y for y in ys if y is not None]
        a1.plot(xs, yy, marker="o", linestyle=_ls.get(d, "-"),
                color=DSET_COLOR[d], lw=1.3, ms=4,
                markeredgecolor="white", markeredgewidth=0.5,
                zorder=_zo.get(d, 2), label=DLABEL[d])
    a1.axhline(0.905, color=AXISC, lw=0.9, ls="--", zorder=1)
    a1.text(len(pk_short) - 1, 0.905, " $\\tau=0.905$", fontsize=6.3,
            color=MUTED, va="bottom", ha="right")
    a1.set_xticks(range(len(pk_short)))
    a1.set_xticklabels(pk_short, fontsize=6.8)
    a1.set_ylabel("Kendall $\\tau$ (ID vs shifted)")
    a1.set_title("S2 peak-window definition", fontsize=7.5, pad=3)
    a1.grid(axis="y"); a1.set_axisbelow(True)
    # right: S3 k values
    for d in DATASETS:
        pts = [(k, _tau(kv[s].get(d))) for k, s in zip(k_x, k_names)]
        pts = [(k, y) for k, y in pts if y is not None]
        a2.plot([k for k, _ in pts], [y for _, y in pts], marker="o",
                linestyle=_ls.get(d, "-"),
                color=DSET_COLOR[d], lw=1.3, ms=4,
                markeredgecolor="white", markeredgewidth=0.5,
                zorder=_zo.get(d, 2), label=DLABEL[d])
    a2.axhline(0.905, color=AXISC, lw=0.9, ls="--", zorder=1)
    a2.set_xticks(k_x)
    a2.set_xlabel("holdout cluster count $k$")
    a2.set_title("S3 holdout cluster count", fontsize=7.5, pad=3)
    a2.grid(axis="y"); a2.set_axisbelow(True)
    for ax in (a1, a2):
        ax.set_ylim(0.3, 1.05)
        ax.tick_params(length=0)
    a2.legend(loc="lower right", fontsize=6.3, handletextpad=0.3,
              borderaxespad=0.2, labelspacing=0.25)
    _save(fig, "G3_split_sensitivity")


# ================================================================ Fig 1 components
# Exported PDF+SVG, transparent, no titles -- composed externally.
def _save_component(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / "fig1_components" / f"{name}.{ext}",
                    bbox_inches="tight", transparent=True, dpi=300)
    plt.close(fig)
    print(f"[fig] wrote fig1_components/{name}.pdf/.svg/.png", flush=True)


def fig1_components():
    import sys as _sys
    _sys.path.insert(0, str(REPO))
    (FIG / "fig1_components").mkdir(parents=True, exist_ok=True)
    from data_loaders.windows import load_raw
    from shift.severity import _clean, _target_rows, w1
    from shift.splits import is_offpeak, is_peak, s2_split

    # (a) LargeST-SD peak vs off-peak density pair with W1 gap annotated
    raw, ts, _ = load_raw("largest_sd")
    sp = s2_split("largest_sd")
    a = _clean(raw[_target_rows(sp.train_anchors)].ravel())   # peak
    b = _clean(raw[_target_rows(sp.test_anchors)].ravel())    # off-peak
    w1v = w1(a, b)
    fig, ax = plt.subplots(figsize=(2.5, 1.7))
    bins = np.histogram_bin_edges(np.concatenate([a[:300000], b[:300000]]), bins=70)
    ax.hist(b, bins=bins, density=True, color=BLUE, alpha=0.6, lw=0)   # off-peak (frozen-side reference)
    ax.hist(a, bins=bins, density=True, color=ORANGE, alpha=0.6, lw=0)  # peak
    ax.annotate(f"$W_1={w1v:.0f}$", xy=(0.62, 0.8), xycoords="axes fraction",
                fontsize=8, color=INK, ha="left")
    ax.set_yticks([]); ax.set_xlabel("flow (veh/15 min)", fontsize=7)
    ax.tick_params(length=0)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    _save_component(fig, "a_density_sd_s2")

    # (b) one-day forecast trace: frozen Chronos-2 tracing weekday habit into weekend.
    # banked y_pred on SD; ground truth from load_raw. Pick one sensor, one 24h window.
    yp = np.load(REPO / "results/baselines/chronos2/largest_sd/y_pred.npy")
    # yp shape asserted; align a horizon-1 forecast strip against truth.
    assert yp.ndim >= 2, f"unexpected y_pred shape {yp.shape}"
    fig, ax = plt.subplots(figsize=(2.8, 1.5))
    # Flatten to a per-window h1 trace for one representative sensor slice.
    # y_pred is (windows, horizon, sensors) or (windows, sensors, horizon);
    # detect the horizon axis (== 12) robustly.
    shp = yp.shape
    haxis = [i for i, s in enumerate(shp) if s == 12]
    sens_axis = None
    if len(shp) == 3 and haxis:
        ha = haxis[0]
        others = [i for i in range(3) if i != ha]
        wax, sax = others[0], others[1]
        # take h1 (index 0 on horizon), one sensor, first 96 windows (24h @15min)
        idx = [slice(None)] * 3
        idx[ha] = 0
        idx[sax] = shp[sax] // 2
        strip = yp[tuple(idx)]
        strip = np.asarray(strip).ravel()[:96]
        gt, _, _ = load_raw("largest_sd")
        # ground-truth strip: same sensor, matched length (approximate, viz only)
        truth = _clean(gt[:, shp[sax] // 2].ravel())[:96] if gt.ndim == 2 else strip
        x = np.arange(len(strip))
        ax.plot(x, truth[:len(strip)], "-", color=INK, lw=1.0, alpha=0.5, zorder=2)
        ax.plot(x, strip, "-", color=BLUE, lw=1.4, zorder=3)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_xlabel("one day (15-min steps)", fontsize=7)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        _save_component(fig, "b_forecast_trace_sd")
    else:
        print(f"[fig] SKIP fig1b: y_pred shape {shp} has no clear horizon axis")

    # (c) mini ranking-slope element: LargeST-SD S1, stripped.
    t3 = load("t3_rankings.json")["datasets"]["largest_sd"]["S1"]
    left = t3.get("models_ranked_id") or []
    right = t3.get("models_ranked_shift") or []
    fig, ax = plt.subplots(figsize=(1.7, 2.0))
    L = {m: len(left) - i for i, m in enumerate(left)}
    R = {m: len(right) - i for i, m in enumerate(right)}
    for m in left:
        moved = left.index(m) != right.index(m)
        ax.plot([0, 1], [L[m], R[m]], "-", color=ORANGE if moved else AXISC,
                lw=1.6 if moved else 1.0, zorder=3 if moved else 1,
                solid_capstyle="round")
    ax.set_xlim(-0.1, 1.1); ax.axis("off")
    _save_component(fig, "c_slope_mini")


STALE = ["F1_severity", "F3_mae_horizon_sd_s1", "F6_ttra_gate_diagnostic"]


def main():
    t1_severity_table()
    f2_degradation()
    f3_sd_s1_case()
    f4_ranking()
    f5_scatter()
    g1_indist()
    g2_deg_horizon()
    g3_split_sensitivity()
    fig1_components()
    for stem in STALE:  # superseded by T1 / F3_sd_s1_case; git history retains
        for ext in ("pdf", "png"):
            p = FIG / f"{stem}.{ext}"
            if p.exists():
                p.unlink()
                print(f"[fig] removed stale {p.name}", flush=True)
    print("[fig] done:", FIG)


if __name__ == "__main__":
    main()
