#!/usr/bin/env python3
"""Regenerate every number the paper reports, from the shipped result files.

Reads only results/*.json and results/baselines/*/*/test_metrics.json, so it
runs on CPU in seconds and needs no model weights, no GPU, and no raw data.
Each check prints the paper's value beside the recomputed one and fails loudly
on any mismatch.

Usage:  python verify.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
FAIL = []


def load(name):
    return json.loads((RES / name).read_text())


def check(label, got, want, tol=5e-3):
    ok = abs(got - want) <= tol
    print(f"  {'OK ' if ok else 'FAIL'}  {label:<52} paper {want:>10.3f}   file {got:>10.3f}")
    if not ok:
        FAIL.append(label)


def main():
    print("\n== Table II: severity and frozen-model degradation ==")
    fdm = load("frozen_degradation_matrix.json")["matrix"]
    tfm = {d: load(f"timesfm_{k}_degradation.json")["cell"]
           for d, k in (("metr_la", "metr_la"), ("pems_bay", "bay"),
                        ("largest_sd", "largest_sd"))}
    S = "S1_weekday_to_weekend"
    check("SD S1 Chronos-2 degradation (%)",
          100 * fdm["largest_sd"][S]["rel_degradation"]["avg"], 28.6, 0.1)
    check("SD S1 TimesFM degradation (%)",
          100 * tfm["largest_sd"][S]["rel_degradation"]["avg"], 8.4, 0.1)

    print("\n== Table IV: ranking stability ==")
    tau = load("t3_rankings.json")["datasets"]
    check("LargeST-SD S1 Kendall tau", tau["largest_sd"]["S1"]["tau"], 0.333, 1e-3)
    check("LargeST-SD S2 Kendall tau", tau["largest_sd"]["S2"]["tau"], 0.867, 1e-3)
    check("METR-LA  S1 Kendall tau", tau["metr_la"]["S1"]["tau"], 0.905, 1e-3)

    print("\n== Section IV-C: severity-degradation correlations ==")
    e5b = load("e5b_ruler_correlations.json")["variants"]["per_sensor_s3"]
    for ruler, want in (("w1", -0.51), ("ks", -0.24), ("dmean", -0.44)):
        check(f"rho over 18 evaluations ({ruler})", e5b[ruler]["rho_18"], want, 0.01)

    print("\n== Table V: repair interventions on LargeST-SD weekends ==")
    tbig = load("tbig_grid.json")["datasets"]["largest_sd"]["models"]
    check("Chronos-2 frozen, weekend pool", tbig["chronos2"]["s1"], 26.14, 0.01)
    check("TimesFM frozen, weekend pool", tbig["timesfm"]["s1"], 23.82, 0.01)
    check("Graph WaveNet standard, weekend pool", tbig["gwnet"]["s1"], 16.21, 0.01)
    router = load("e4_router.json")
    check("Router, weekend pool", router["test"]["router"]["weekend_pool"], 21.57, 0.01)
    check("Router, full test period", router["test"]["router"]["full_test"], 20.46, 0.01)

    print("\n== Table III: STAEformer LargeST-SD cells ==")
    stae = load("staeformer_sd_cells.json")
    for k, want in (("std", 18.44), ("s1", 16.17), ("s2", 13.65), ("s3", 14.34)):
        check(f"STAEformer {k}", stae["grid"]["staeformer"][k], want, 0.01)
    check("Kendall tau, M=6 (paper)", stae["tau_m6_preregistered"], 0.333, 1e-3)
    check("Kendall tau, M=7 (artifacts)",
          stae["tau_m7_supplementary_artifacts_only"], 0.524, 1e-3)

    print("\n== Proposition 1 verification on released checkpoints ==")
    e3 = load("e3_timesfm_equivariance.json")
    print(f"  info  TimesFM worst relative deviation: {e3.get('worst_rel_dev', 'see file')}")

    print("\n== Batch independence of the two frozen backbones ==")
    bi = load("verification/batch_independence.json")["models"]
    for m in ("chronos2", "timesfm"):
        v = bi[m]["worst_max_abs_batch_vs_perm"]
        print(f"  {'OK ' if v == 0 else 'FAIL'}  {m}: worst |batch - permuted| = {v}")
        if v != 0:
            FAIL.append(f"{m} batch independence")

    print()
    if FAIL:
        print(f"FAILED {len(FAIL)} check(s): {FAIL}")
        sys.exit(1)
    print("All checks reproduce the values reported in the paper.")


if __name__ == "__main__":
    main()
