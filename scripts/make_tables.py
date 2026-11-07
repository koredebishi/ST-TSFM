#!/usr/bin/env python3
"""Print the paper's tables and numbers from artifacts/results/.

Each line shows the value as the paper prints it next to the value in the
result files, rounded the same way. Needs no data, GPU or forecasts, and
runs in seconds. Exits with an error if any number differs.

Usage:  python scripts/make_tables.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "artifacts/results"
RUNS = RES / "runs"
DS = ("metr_la", "pems_bay", "largest_sd")
SHIFTS = ("S1", "S2", "S3")
FROZEN = ("chronos2", "timesfm")
FAILED = []


def load(path):
    return json.loads(Path(path).read_text())


def check(where, what, paper, files):
    ok = paper == files
    print(f"  {'OK  ' if ok else 'FAIL'} {where:<10} {what:<46} paper {paper:<26} files {files}")
    if not ok:
        FAILED.append(f"{where}: {what}")


def f1(x):
    return f"{x:.1f}"


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{round(100 * x):d}"


def pct1(x):
    return f"{100 * x:+.1f}" if x > 0 else f"{100 * x:.1f}"


def mean_sd(v):
    return f"{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f}"


def log_value(name, pattern=r"Average Test MAE: ([0-9.]+)"):
    return float(re.findall(pattern, (RUNS / "logs" / name).read_text())[-1])


def main():
    deg = load(RES / "degradation.json")
    avg = {(d, s, m): deg[d][m][s]["degradation"]["avg"] for d in DS for s in SHIFTS for m in FROZEN}
    sev = load(RES / "severity.json")
    corr = load(RES / "correlations.json")
    skill = load(RES / "skill_change.json")
    grid = load(RES / "mae_grid.json")
    rank = load(RES / "rankings.json")
    days = load(RES / "day_block.json")
    router = load(RES / "router.json")
    tm = lambda m, d: load(RUNS / "test_metrics" / m / f"{d}.json")
    gate = load(RUNS / "ttra/w6_gate.json")
    pin = load(RUNS / "ttra/w6_diag_pinball.json")
    ctx = load(RES / "context_length.json")

    print("\nAbstract and introduction")
    check("Abstract", "largest frozen-model degradation (%)", "29", pct(avg[("largest_sd", "S1", "chronos2")]))
    sd = grid["largest_sd"]
    check("Intro", "LargeST-SD weekend MAE, Chronos-2 / TimesFM", "26.14/23.82",
          f"{f2(sd['chronos2']['s1'])}/{f2(sd['timesfm']['s1'])}")
    check("Intro", "LargeST-SD weekend MAE, router", "21.57", f2(router["test"]["weekend"]))

    print("\nSection III (method)")
    s3 = {d: deg[d]["chronos2"]["S3"]["pool_sizes"] for d in DS}
    check("III-B", "S3 held-out / all sensors", "37/207,36/325,102/716",
          ",".join(f"{s3[d][1]}/{sum(s3[d])}" for d in DS))
    pooled = load(RES / "pooled_w1.json")["largest_sd"]
    check("III-C", "LargeST-SD pooled W1, S1/S3/S2", "49.7/68.3/140.8",
          f"{f1(pooled['S1'])}/{f1(pooled['S3'])}/{f1(pooled['S2'])}")
    prop1 = load(RUNS / "equivariance/prop1_equivariance.json")["rows"]
    b1000 = [r for r in prop1 if r["a"] == 1.0 and r["b"] == 1000.0]
    check("III-F", "Chronos-2 MAE ratio deviation at b=1000 (%)", "0.78",
          f"{100 * max(abs(r['mae_ratio_vs_a'] - 1) for r in b1000):.2f}")
    a2 = [r for r in prop1 if r["a"] == 2.0 and r["b"] == 0.0]
    check("III-F", "Chronos-2 at a=2 matches bit for bit", "yes",
          "yes" if all(r["max_elem_dev_in_window_scales"] == 0 for r in a2) else "no")
    e3 = load(RUNS / "equivariance/e3_timesfm_equivariance.json")
    check("III-F", "TimesFM worst relative deviation < 2e-7", "yes",
          "yes" if e3["worst_rel_dev"] < 2e-7 else f"{e3['worst_rel_dev']:.3g}")

    print("\nTable I (compute) and Section IV-A (setup)")
    check("Table I", "Chronos-2 inference, all datasets (h)", "0.7",
          f1(sum(tm("chronos2", d)["predict_wall_seconds"] for d in DS) / 3600))
    check("Table I", "TimesFM inference, all datasets (h)", "192.8",
          f1(sum(tm("timesfm", d)["predict_wall_seconds"] for d in DS) / 3600))
    check("Table I", "STAEformer training, LargeST-SD (h)", "4.0",
          f1(tm("staeformer", "largest_sd")["train_wall_seconds"] / 3600))
    check("IV-A", "sensors in all three datasets", "1,248", f"{sum(sum(s3[d]) for d in DS):,}")
    c2sd = tm("chronos2", "largest_sd")
    check("IV-A", "LargeST-SD target values (millions)", "60", f"{round(c2sd['n_windows'] * 12 * 716 / 1e6)}")
    check("IV-A", "sensors per forward pass", "256", str(c2sd["chunk"]))
    check("IV-A", "context length", "512", str(tm("timesfm", "largest_sd")["context_len"]))
    check("IV-A", "TimesFM checkpoint", "timesfm-2.5-200m-pytorch",
          tm("timesfm", "largest_sd")["model_id"].split("/")[-1])

    print("\nTable II (W1 severity, degradation % for Chronos-2 and TimesFM)")
    paper2 = {"metr_la": "4.3 -1.5 -4.5 | 6.8 -28.3 -35.5 | 5.5 +18.7 +20.6",
              "pems_bay": "4.2 -51.4 -48.1 | 7.4 -52.9 -54.7 | 3.5 -19.9 -20.1",
              "largest_sd": "55.1 +28.6 +8.4 | 142.2 -33.7 -35.2 | 92.6 -26.1 -25.7"}
    for d in DS:
        files = " | ".join(f"{f1(sev[d][s]['w1'])} {pct1(avg[(d, s, 'chronos2')])} {pct1(avg[(d, s, 'timesfm')])}"
                           for s in SHIFTS)
        check("Table II", d, paper2[d], files)

    print("\nTable III (MAE: Std S1 S2 S3 for METR-LA | PEMS-BAY | LargeST-SD)")
    paper3 = {
        "ha": "4.19 3.52 3.44 4.99 | 2.63 1.71 1.94 2.26 | 27.93 21.57 20.16 22.32",
        "arima": "5.23 5.45 4.77 5.87 | 2.23 1.28 1.76 1.84 | 55.43 42.16 45.47 42.62",
        "dcrnn": "3.16 2.73 2.71 3.73 | 1.61 1.02 1.23 1.34 | 21.20 17.73 16.31 16.10",
        "gwnet": "3.04 2.52 2.64 3.59 | 1.58 1.00 1.22 1.31 | 18.50 16.21 14.19 14.64",
        "staeformer": "2.94 2.47 2.58 3.45 | 1.54 0.98 1.18 1.24 | 18.44 16.17 13.65 14.34",
        "chronos2": "4.65 4.61 4.10 5.34 | 1.84 1.05 1.37 1.51 | 21.70 26.14 17.30 16.67",
        "timesfm": "3.91 3.78 3.38 4.56 | 1.69 1.02 1.24 1.38 | 22.20 23.82 17.30 17.13"}
    for m, paper in paper3.items():
        files = " | ".join(" ".join(f2(grid[d][m][k]) for k in ("std", "s1", "s2", "s3")) for d in DS)
        check("Table III", m, paper, files)

    print("\nFinding 1 (Section IV-C)")
    n_ok = sum(all(avg[(d, s, m)] <= 0 for m in FROZEN) for d in DS for s in SHIFTS)
    n_eval = sum(avg[k] <= 0 for k in avg)
    check("IV-C", "settings / evaluations without degradation", "7/14", f"{n_ok}/{n_eval}")
    best = min(avg[("pems_bay", s, m)] for s in SHIFTS for m in FROZEN)
    check("IV-C", "largest PEMS-BAY improvement (%)", "54.7", f"{-100 * best:.1f}")
    check("IV-C", "SD-S1 and LA-S3 degradation (%)", "29/8/19/21",
          "/".join(pct(avg[k]) for k in (("largest_sd", "S1", "chronos2"), ("largest_sd", "S1", "timesfm"),
                                         ("metr_la", "S3", "chronos2"), ("metr_la", "S3", "timesfm"))))
    check("IV-C", "rho over 18 evaluations, W1 / KS / mean diff.", "-0.51/-0.24/-0.44",
          "/".join(f2(corr[m]["rho_18"]) for m in ("w1", "ks", "dmean")))
    check("IV-C", "setting-level rho and p, W1", "-0.47/0.20", f"{f2(corr['w1']['rho_9'])}/{f2(corr['w1']['p_9'])}")
    order = lambda m: [sorted(SHIFTS, key=lambda s: sev[d][s][m]) for d in DS]
    check("IV-C", "mean difference orders settings like W1", "yes", "yes" if order("w1") == order("dmean") else "no")
    check("Fig. 2", "setting-level rho, W1 / KS / mean diff.", "-0.47,-0.25,-0.41",
          ",".join(f2(corr[m]["rho_9"]) for m in ("w1", "ks", "dmean")))
    check("Fig. 2", "setting-level p, W1 / KS / mean diff.", "0.20,0.51,0.28",
          ",".join(f2(corr[m]["p_9"]) for m in ("w1", "ks", "dmean")))
    sk = corr["w1_vs_skill_change"]
    check("IV-C", "skill-change rho and p", "+0.02/0.98", f"{sk['rho_9']:+.2f}/{f2(sk['p_9'])}")
    loss = {(d, s): [skill[d][m][s] for m in FROZEN] for d in DS for s in SHIFTS}
    check("IV-C", "SD-S1 skill loss, Chronos-2 / TimesFM", "0.56/0.40", "/".join(f2(x) for x in loss[("largest_sd", "S1")]))
    check("IV-C", "largest skill loss of the nine settings", "largest_sd/S1", "/".join(max(loss, key=lambda k: sum(loss[k]))))
    check("IV-C", "LA-S3: frozen models lose less skill than HA", "yes",
          "yes" if all(x < 0 for x in loss[("metr_la", "S3")]) else "no")
    std = load(RES / "standardized_severity.json")["correlations"]
    check("IV-C", "standardized severity rho, W1 / KS", "-0.56/-0.44", f"{f2(std['w1']['rho_9'])}/{f2(std['ks']['rho_9'])}")
    rel = load(RES / "relative_metrics.json")
    check("IV-C", "W1 rho under MASE / sMAPE", "-0.51/-0.12",
          f"{f2(rel['correlations']['mase']['rho_9'])}/{f2(rel['correlations']['smape']['rho_9'])}")
    check("IV-C", "SD off-peak Chronos-2 change under sMAPE (%)", "63",
          pct(rel["degradation"]["largest_sd"]["chronos2"]["S2"]["smape"]))

    print("\nFinding 2 (Section IV-D) and Table IV")
    paper4 = {"metr_la": "0.905 0.905 1.000", "pems_bay": "1.000 1.000 1.000", "largest_sd": "0.333 0.867 1.000"}
    for d in DS:
        check("Table IV", d, paper4[d], " ".join(f3(rank[d][s]["tau"]) for s in SHIFTS))
    check("IV-D", "SD-S1 day-block tau interval", "0.20/0.47", "/".join(f2(x) for x in days["tau"]))
    check("IV-D", "SD-S1 Chronos-2 degradation interval (%)", "13/45", "/".join(pct(x) for x in days["degradation"]["chronos2"]))
    check("IV-D", "SD-S1 TimesFM degradation interval (%)", "-2/20", "/".join(pct(x) for x in days["degradation"]["timesfm"]))
    r = rank["largest_sd"]["S1"]
    check("Table IV", "SD-S1 sensor bootstrap interval", "0.333/0.333", "/".join(f3(x) for x in r["interval"]))
    shifted, in_dist = r["ranked_shifted"], r["ranked_in_dist"]
    check("IV-D", "SD weekends: HA above both frozen models", "yes",
          "yes" if shifted.index("ha") < min(shifted.index(m) for m in FROZEN) else "no")
    check("Fig. 3", "Chronos-2 rank, weekdays -> weekends", "2->5",
          f"{in_dist.index('chronos2') + 1}->{shifted.index('chronos2') + 1}")
    passes = lambda r, a, b: (r["ranked_in_dist"].index(a) > r["ranked_in_dist"].index(b)
                              and r["ranked_shifted"].index(a) < r["ranked_shifted"].index(b))
    check("IV-D", "LA: HA passes TimesFM (S1) and Chronos-2 (S2)", "yes/yes",
          "/".join("yes" if passes(rank["metr_la"][s], "ha", m) else "no" for s, m in (("S1", "timesfm"), ("S2", "chronos2"))))
    check("Files", "LargeST-SD S1 tau with STAEformer (M = 7)", "0.524",
          f3(load(RES / "ranking_with_staeformer.json")["largest_sd_S1_tau_m7"]))

    print("\nFinding 3 (Section IV-E) and Table V")
    check("IV-E", "SD-S1 Chronos-2 degradation at h12 (%)", "51",
          pct(deg["largest_sd"]["chronos2"]["S1"]["degradation"]["h12"]))
    safe = [deg[d][m][s]["degradation"][h] for d in DS for s in SHIFTS for m in FROZEN for h in ("h3", "h6", "h12")
            if (d, s) not in (("largest_sd", "S1"), ("metr_la", "S3"))]
    check("IV-E", "other settings never exceed 9% at any horizon", "yes", "yes" if max(safe) <= 0.09 else "no")
    wd = load(RES / "weekend_days.json")
    check("IV-E", "Sat / Sun degradation, C2 then TFM (%)", "67/2/34/-10",
          "/".join(pct(wd[m][day]) for m in FROZEN for day in ("saturday", "sunday")))
    seeds = (2023, 2024, 2025)
    run = lambda name: [load(RUNS / "retraining" / f"{name}_s{s}.json")["average"]["mae"] for s in seeds]
    gwnet_weekday = [load(RUNS / "retraining/gwnet_weekday_s2023.json")["average"]["mae"],
                     log_value("gwnet_weekday_s2024_job47233067.log"),
                     log_value("gwnet_weekday_s2025_job47233068.log")]
    window_norm = [log_value(f) for f in ("gwnet_window_norm_s2023_job47232552.log",
                                          "gwnet_window_norm_s2024_job47233065.log",
                                          "gwnet_window_norm_s2025_job47233066.log")]
    for what, paper, files in (
            ("Chronos-2, none", "26.14", f2(gate["frozen_shift"]["avg"])),
            ("TimesFM, none", "23.82", f2(sd["timesfm"]["s1"])),
            ("Chronos-2, TTRA adaptation", "27.80", f2(gate["adapted"]["avg"])),
            ("Chronos-2, 48-hour context", "26.80", f2(ctx["chronos2"]["192"]["weekend"])),
            ("Chronos-2, 24-hour context", "29.76", f2(ctx["chronos2"]["96"]["weekend"])),
            ("TimesFM, 48-hour context", "23.11", f2(ctx["timesfm"]["192"]["weekend"])),
            ("TimesFM, 24-hour context", "33.15", f2(ctx["timesfm"]["96"]["weekend"])),
            ("Graph WaveNet, weekday-only retraining", "24.67 ± 1.79", mean_sd(gwnet_weekday)),
            ("Graph WaveNet, weekend-only training", "15.60 ± 0.13", mean_sd(run("gwnet_weekend"))),
            ("Graph WaveNet, standard subset", "15.35 ± 0.27", mean_sd(run("gwnet_std_match"))),
            ("Graph WaveNet, weekday subset", "29.27 ± 2.63", mean_sd(run("gwnet_weekday_match"))),
            ("Graph WaveNet, per-window normalization", "27.85 ± 0.61", mean_sd(window_norm)),
            ("DCRNN, weekday-only retraining", "28.96 ± 0.72", mean_sd(run("dcrnn_weekday"))),
            ("DCRNN, weekend-only training", "19.28 ± 1.55", mean_sd(run("dcrnn_weekend"))),
            ("Regime router", "21.57", f2(router["test"]["weekend"]))):
        check("Table V", what, paper, files)
    check("IV-E", "48 h: weekend MAE changes by about 3%, both", "3/3",
          "/".join(pct(abs(ctx[m]["192"]["weekend"] / ctx[m]["512"]["weekend"] - 1)) for m in FROZEN))
    check("IV-E", "48 h: weekend MAE stays above HA, both", "yes",
          "yes" if all(ctx[m]["192"]["weekend"] > sd["ha"]["s1"] for m in FROZEN) else "no")
    check("IV-E", "24 h: weekend MAE rises, both models", "yes",
          "yes" if all(ctx[m]["96"]["weekend"] > ctx[m]["512"]["weekend"] for m in FROZEN) else "no")
    check("IV-E", "24 h: weekday MAE more than doubles, both", "yes",
          "yes" if all(ctx[m]["96"]["weekday"] > 2 * ctx[m]["512"]["weekday"] for m in FROZEN) else "no")
    check("IV-E", "standard GWNet / DCRNN on the weekend pool", "16.21/17.73",
          f"{f2(sd['gwnet']['s1'])}/{f2(sd['dcrnn']['s1'])}")
    fv, av = pin["frozen"]["val_weekday"]["pinball"], pin["adapted"]["val_weekday"]["pinball"]
    ft, at = pin["frozen"]["test_weekend"]["pinball"], pin["adapted"]["test_weekend"]["pinball"]
    check("IV-E", "adapter pinball change, val weekdays (%)", "13.1", f"{100 * (fv - av) / fv:.1f}")
    check("IV-E", "adapter pinball change, test weekends (%)", "8.3", f"{100 * (at - ft) / ft:.1f}")
    check("IV-E", "adapter worse at every horizon", "yes",
          "yes" if all(gate["adapted"]["per_horizon"][h] > gate["frozen_shift"]["per_horizon"][h]
                       for h in ("3", "6", "12")) else "no")
    tv = log_value("gwnet_weekday_s2023_job46713086.log", r"Val loss decrease from [0-9.]+ to ([0-9.]+)")
    ev = log_value("gwnet_window_norm_s2023_job47232552.log", r"Val loss decrease from [0-9.]+ to ([0-9.]+)")
    check("IV-E", "GWNet val-to-test gap, plain / normalized (%)", "43/26",
          f"{pct(gwnet_weekday[0] / tv - 1)}/{pct(window_norm[0] / ev - 1)}")
    check("IV-E", "router MAE, weekends / full test", "21.57/20.46",
          f"{f2(router['test']['weekend'])}/{f2(router['test']['full_test'])}")
    check("IV-E", "router picks (weekday / weekend)", "chronos2/ha",
          f"{router['chosen']['weekday']}/{router['chosen']['weekend']}")
    check("IV-E", "standard DCRNN / GWNet, full test", "21.20/18.50", f"{f2(sd['dcrnn']['std'])}/{f2(sd['gwnet']['std'])}")

    print("\nTable VI (registered predictions)")
    check("Table VI", "P1: SD tau S1 < S2", "0.333/0.867",
          f"{f3(rank['largest_sd']['S1']['tau'])}/{f3(rank['largest_sd']['S2']['tau'])}")
    check("Table VI", "P3: STAEformer shifted rank, LA/BAY S1/S2", "1,1,1,1",
          ",".join(str(rank[d][s]["ranked_shifted"].index("staeformer") + 1)
                   for d in ("metr_la", "pems_bay") for s in ("S1", "S2")))

    print("\nSection V (split sensitivity)")
    sens = load(RES / "sensitivity.json")
    peak = [c for d in DS for c in sens[d]["peak_windows"].values()]
    usable = [c for c in peak if c["tau"] is not None]
    improve = sum(all(c["degradation"][m] < 0 for m in FROZEN) for c in usable)
    check("V", "peak-window pairs: usable / all", "11/12", f"{len(usable)}/{len(peak)}")
    check("V", "usable pairs where off-peak improves", "11", str(improve))
    check("V", "lowest tau across peak windows", "0.81", f2(min(c["tau"] for c in usable)))
    check("V", "lowest tau across cluster counts", "0.905",
          f3(min(c["tau"] for d in DS for c in sens[d]["cluster_counts"].values())))

    print("\nBatch independence of the frozen models")
    bi = load(RUNS / "equivariance/batch_independence.json")["models"]
    for m in FROZEN:
        check("Check", f"{m}: largest |batch - shuffled| difference", "0", f"{bi[m]['worst_max_abs_batch_vs_perm']:g}")

    print()
    if FAILED:
        print(f"{len(FAILED)} number(s) differ from the paper:")
        for f in FAILED:
            print("  ", f)
        sys.exit(1)
    print("Every number matches the paper.")


if __name__ == "__main__":
    main()
