#!/usr/bin/env python3
"""Recompute every result file from the datasets and the saved forecasts.

Writes one JSON file per result to artifacts/results/ and prints the main
numbers. scripts/make_tables.py turns those files into the paper's tables.

Usage:  python scripts/make_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy.stats import kendalltau  # noqa: E402

from src import historical_average, router  # noqa: E402
from src.data import LABELS, NAMES, load_forecast, test_targets  # noqa: E402
from src.degradation import (degradation, naive_scale, relative_degradation,  # noqa: E402
                             skill_change)
from src.metrics import mae  # noqa: E402
from src.ranking import day_block_bootstrap, ranking_stability  # noqa: E402
from src.sensitivity import cluster_variants, peak_variants  # noqa: E402
from src.severity import MEASURES, pooled_w1, severity, standardized_severity  # noqa: E402
from src.splits import SHIFTS, heldout_sensors, test_pools, weekend_day_pools  # noqa: E402
from src.stats import correlate, normalize_within_dataset  # noqa: E402

OUT = ROOT / "artifacts/results"
FROZEN = ("chronos2", "timesfm")
ALL_MODELS = ("ha", "arima", "dcrnn", "gwnet", "staeformer", "chronos2", "timesfm")
# The ranking set was fixed before the shift runs. STAEformer was not yet
# trained on LargeST-SD then, so there it is in Table III but not in the
# ranking; the M = 7 variant is reported in the result files only.
RANKED = {"metr_la": list(ALL_MODELS), "pems_bay": list(ALL_MODELS),
          "largest_sd": ["ha", "arima", "dcrnn", "gwnet", "chronos2", "timesfm"]}


def save(name, obj):
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1))


def table2():
    """Severity of each setting and the degradation of the frozen models."""
    sev, deg, skill = {}, {}, {}
    for name in NAMES:
        sev[name] = severity(name)
        y_true = test_targets(name)
        y_ha = load_forecast("ha", name)
        deg[name], skill[name] = {}, {}
        for model in FROZEN:
            y_pred = load_forecast(model, name)
            assert y_pred.shape == y_true.shape, (model, name)
            deg[name][model] = {s: degradation(name, s, y_true, y_pred) for s in SHIFTS}
            skill[name][model] = {s: skill_change(name, s, y_true, y_pred, y_ha) for s in SHIFTS}

    print("\nTable II: severity (W1) and degradation (%), Chronos-2 / TimesFM")
    for name in NAMES:
        cells = [f"{sev[name][s]['w1']:6.1f} {100 * deg[name]['chronos2'][s]['degradation']['avg']:+6.1f}"
                 f" {100 * deg[name]['timesfm'][s]['degradation']['avg']:+6.1f}" for s in SHIFTS]
        print(f"{LABELS[name]:12}" + " |".join(cells))
    return sev, deg, skill


def correlations(sev, deg, skill):
    """Does severity rank the settings by degradation? (Section IV-C)"""
    avg_deg = {m: {d: {s: deg[d][m][s]["degradation"]["avg"] for s in SHIFTS} for d in NAMES}
               for m in FROZEN}
    out = {}
    for measure in MEASURES:
        norm = normalize_within_dataset({d: {s: sev[d][s][measure] for s in SHIFTS} for d in NAMES})
        out[measure] = correlate(norm, avg_deg)
    w1 = normalize_within_dataset({d: {s: sev[d][s]["w1"] for s in SHIFTS} for d in NAMES})
    by_skill = {m: {d: {s: skill[d][m][s] for s in SHIFTS} for d in NAMES} for m in FROZEN}
    out["w1_vs_skill_change"] = correlate(w1, by_skill)

    print("\nSeverity vs degradation: rho over 18 evaluations, rho over 9 settings (exact p)")
    for k, c in out.items():
        print(f"  {k:20} {c['rho_18']:+.2f}   {c['rho_9']:+.2f} (p = {c['p_9']:.2f})")
    return out


def all_models(name):
    """True targets and every model's forecasts, cut to the windows that
    every model covers (the LargeST-SD graph-model runs miss the last few)."""
    preds = {m: load_forecast(m, name) for m in ALL_MODELS}
    n = min(len(p) for m, p in preds.items() if m in RANKED[name])
    return test_targets(name)[:n], {m: p[:n] for m, p in preds.items()}


def table3_and_4():
    """Table III (MAE of every model on each pool), Table IV (Kendall tau),
    the split sensitivity of Section V, the LargeST-SD day-block
    intervals, the M = 7 tau and the router."""
    grid, rankings, sens = {}, {}, {}
    for name in NAMES:
        y_true, preds = all_models(name)
        n = len(y_true)
        _, weekend = test_pools(name, "S1", n)
        _, offpeak = test_pools(name, "S2", n)
        _, held_out = heldout_sensors(name)
        grid[name] = {m: {"std": mae(y_true, p),
                          "s1": mae(y_true[weekend], p[weekend]),
                          "s2": mae(y_true[offpeak], p[offpeak]),
                          "s3": mae(y_true[:, :, held_out], p[:, :, held_out])}
                      for m, p in preds.items()}
        rankings[name] = ranking_stability(name, y_true, {m: preds[m] for m in RANKED[name]})
        ranked = {m: preds[m] for m in RANKED[name]}
        sens[name] = {"peak_windows": peak_variants(name, y_true, ranked),
                      "cluster_counts": cluster_variants(name, y_true, ranked)}

        if name == "largest_sd":
            day_block = day_block_bootstrap(name, y_true, {m: preds[m] for m in RANKED[name]})
            weekday, _ = test_pools(name, "S1", n)
            m7 = ALL_MODELS
            tau7 = kendalltau([mae(y_true[weekday], preds[m][weekday]) for m in m7],
                              [mae(y_true[weekend], preds[m][weekend]) for m in m7]).statistic
            val = {m: load_forecast(m, name, "val") for m in ("chronos2", "timesfm")}
            val["ha"] = historical_average.forecast(name, "val")
            chosen, val_mae = router.choose(val)
            routed = router.route(chosen, y_true, preds)

    print("\nTable IV: Kendall tau (S1, S2, S3)")
    for name in NAMES:
        print(f"  {LABELS[name]:12}" + "  ".join(f"{rankings[name][s]['tau']:.3f}" for s in SHIFTS))
    print(f"  LargeST-SD S1 day-block tau interval {day_block['tau']}")
    print(f"  router picks {chosen}; MAE {routed}")
    save("mae_grid", grid)
    save("rankings", rankings)
    save("sensitivity", sens)
    save("day_block", day_block)
    save("ranking_with_staeformer", {"largest_sd_S1_tau_m7": float(tau7)})
    save("router", {"chosen": chosen, "validation_mae": val_mae, "test": routed})


def follow_ups(sev, deg):
    """Three checks discussed in Sections IV-C and IV-E.

    1. Severity on per-window standardized data, as the models see it.
    2. Degradation under sMAPE and MASE instead of MAE.
    3. LargeST-SD weekends split into Saturday and Sunday.
    """
    avg_deg = {m: {d: {s: deg[d][m][s]["degradation"]["avg"] for s in SHIFTS} for d in NAMES}
               for m in FROZEN}
    std_sev = {d: standardized_severity(d) for d in NAMES}
    std_corr = {measure: correlate(normalize_within_dataset(
        {d: {s: std_sev[d][s][measure] for s in SHIFTS} for d in NAMES}), avg_deg)
        for measure in MEASURES}

    rel = {}
    for name in NAMES:
        y_true, scale = test_targets(name), naive_scale(name)
        rel[name] = {m: {s: relative_degradation(name, s, y_true, load_forecast(m, name), scale)
                         for s in SHIFTS} for m in FROZEN}
    w1 = normalize_within_dataset({d: {s: sev[d][s]["w1"] for s in SHIFTS} for d in NAMES})
    rel_corr = {metric: correlate(w1, {m: {d: {s: rel[d][m][s][metric] for s in SHIFTS}
                                            for d in NAMES} for m in FROZEN})
                for metric in ("smape", "mase")}

    y_true = test_targets("largest_sd")
    pools = weekend_day_pools("largest_sd")
    days = {}
    for m in FROZEN:
        y_pred = load_forecast(m, "largest_sd")
        base = mae(y_true[pools["weekday"]], y_pred[pools["weekday"]])
        days[m] = {d: mae(y_true[pools[d]], y_pred[pools[d]]) / base - 1 for d in ("saturday", "sunday")}

    print("\nStandardized severity: rho over 9 settings, W1 %+.2f, KS %+.2f"
          % (std_corr["w1"]["rho_9"], std_corr["ks"]["rho_9"]))
    print("W1 vs degradation under MASE %+.2f, under sMAPE %+.2f"
          % (rel_corr["mase"]["rho_9"], rel_corr["smape"]["rho_9"]))
    print("LargeST-SD Saturday / Sunday degradation:", days)
    save("standardized_severity", {"severity": std_sev, "correlations": std_corr})
    save("relative_metrics", {"degradation": rel, "correlations": rel_corr})
    save("weekend_days", days)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sev, deg, skill = table2()
    save("severity", sev)
    save("degradation", deg)
    save("skill_change", skill)
    save("pooled_w1", {d: pooled_w1(d) for d in NAMES})
    save("correlations", correlations(sev, deg, skill))
    table3_and_4()
    follow_ups(sev, deg)


if __name__ == "__main__":
    main()
