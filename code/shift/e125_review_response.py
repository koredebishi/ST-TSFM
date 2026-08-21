#!/usr/bin/env python3
"""D37 experiments E1 + E2 + E5 (review response), assembly-only.

E1  unified severity across the nine settings, two variants:
    (a) pooled-marginal W1 for every setting;
    (b) per-sensor W1 for S3 (each held-out sensor vs pooled retained
        marginal, sensor mean), matching S1/S2's functional.
E2  skill vs HA on both sides of every setting for the frozen models;
    Delta-skill; Spearman against each severity variant.
E5  grouped statistics: exact n=9 permutation, leave-one-dataset-out,
    day-block bootstrap for SD-S1 degradation and tau.

Reuse only (D37): shift.severity (w1, _clean, _target_rows),
shift.splits (s1/s2/s3_split, predicates, classify_targets),
shift.t3_rankings (MODELS, load_preds, truth), eval.metrics.mae,
data_loaders.windows (frozen_anchors, load_raw). Parity asserts tie
every recomputed quantity to a banked artifact before anything new is
derived from it. Outputs: results/shift/e{1,2,5}_*.json.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scipy.stats import kendalltau, rankdata, spearmanr  # noqa: E402

from data_loaders.windows import frozen_anchors, load_raw  # noqa: E402
from eval.metrics import mae as metrics_mae  # noqa: E402
from shift.severity import _clean, _target_rows, w1  # noqa: E402
from shift.splits import (classify_targets, is_offpeak, is_peak,  # noqa: E402
                          is_weekday, is_weekend, s1_split, s2_split,
                          s3_split)
from scripts.verification.frozen_degradation_matrix import truth  # noqa: E402
from shift.t3_rankings import MODELS, load_preds  # noqa: E402

SH = REPO / "results/shift"
DATASETS = ("metr_la", "pems_bay", "largest_sd")
SKEY = {"S1": "S1_weekday_to_weekend", "S2": "S2_peak_to_offpeak",
        "S3": "S3_spatial_holdout"}
TF_FILE = {"metr_la": "timesfm_metr_la_degradation.json",
           "pems_bay": "timesfm_bay_degradation.json",
           "largest_sd": "timesfm_largest_sd_degradation.json"}
FAILING = {("largest_sd", "S1"), ("metr_la", "S3")}
SEED = 2023
N_BOOT_DAYS = 2000


def pool_mae(yt, yp):
    """Masked pooled MAE, the house metric, float64."""
    return float(metrics_mae(yt.astype(np.float64), yp.astype(np.float64),
                             mask_zero=True))


# ------------------------------------------------------------------ E1
def e1():
    banked = json.loads((SH / "severity_summary.json").read_text())["severities"]
    out = {"spec": "decisions.md D37 E1", "variants": {}}
    cur, pooled_all, ps_s3 = {}, {}, {}
    for ds in DATASETS:
        raw, _, _ = load_raw(ds)
        s1, s2, s3 = s1_split(ds), s2_split(ds), s3_split(ds)
        rows_s3 = _target_rows(frozen_anchors(ds)["test"])

        cur[ds] = {"S1": banked[ds]["S1"]["severity_w1_mean"],
                   "S2": banked[ds]["S2"]["severity_w1_mean"],
                   "S3": banked[ds]["S3"]["severity_w1_pooled"]}

        pa = {}
        for sname, sp in (("S1", s1), ("S2", s2)):
            ra, rb = _target_rows(sp.train_anchors), _target_rows(sp.test_anchors)
            pa[sname] = w1(_clean(raw[ra, :].ravel()),
                           _clean(raw[rb, :].ravel()))
        pa["S3"] = w1(_clean(raw[np.ix_(rows_s3, s3.train_sensors)].ravel()),
                      _clean(raw[np.ix_(rows_s3, s3.test_sensors)].ravel()))
        # parity: pooled S3 must reproduce the banked pooled S3 value
        assert abs(pa["S3"] - cur[ds]["S3"]) / cur[ds]["S3"] < 5e-3, \
            (ds, pa["S3"], cur[ds]["S3"])
        pooled_all[ds] = pa

        ref = _clean(raw[np.ix_(rows_s3, s3.train_sensors)].ravel())
        per = []
        for j in s3.test_sensors:
            v = _clean(raw[rows_s3, j])
            if len(v) >= 10:
                per.append(w1(v, ref))
        ps_s3[ds] = {"S1": cur[ds]["S1"], "S2": cur[ds]["S2"],
                     "S3": float(np.mean(per))}
        print(f"[E1] {ds}: current={cur[ds]} pooled_all={pa} "
              f"per_sensor_S3={ps_s3[ds]['S3']:.3f}", flush=True)

    for vname, v in (("current_mixed", cur), ("pooled_all", pooled_all),
                     ("per_sensor_s3", ps_s3)):
        block = {}
        for ds in DATASETS:
            vals = v[ds]
            mx = max(vals.values())
            block[ds] = {
                "raw": {k: float(x) for k, x in vals.items()},
                "normalized": {k: float(x / mx) for k, x in vals.items()},
                "lowest_setting": min(vals, key=vals.get),
            }
        block["failing_is_lowest"] = {
            f"{ds}-{s}": bool(block[ds]["lowest_setting"] == s)
            for ds, s in [("largest_sd", "S1"), ("metr_la", "S3")]}
        out["variants"][vname] = block
    (SH / "e1_unified_severity.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ E2
def e2():
    fdm = json.loads((SH / "frozen_degradation_matrix.json").read_text())["matrix"]
    tf = {ds: json.loads((SH / TF_FILE[ds]).read_text())["cell"]
          for ds in DATASETS}
    out = {"spec": "decisions.md D37 E2", "datasets": {}}
    for ds in DATASETS:
        yt = truth(ds).astype(np.float32)
        n_test, _, N = yt.shape
        anc = frozen_anchors(ds)["test"]
        _, ts, _ = load_raw(ds)
        models = ["ha", "chronos2", "timesfm"]
        loaded = {m: load_preds(m, ds, n_test, N, yt) for m in models}
        n_use = min(min(yp.shape[0] for yp, _ in loaded.values()), n_test)
        yt, anc = yt[:n_use], anc[:n_use]
        preds = {m: loaded[m][0][:n_use] for m in models}

        sp3 = s3_split(ds)
        pools = {}
        for sname, bp, spred in (("S1", is_weekday, is_weekend),
                                 ("S2", is_peak, is_offpeak)):
            bm = classify_targets(anc, np.asarray(bp(ts)))
            sm = classify_targets(anc, np.asarray(spred(ts)))
            pools[sname] = {"id": (yt[bm], {m: preds[m][bm] for m in models}),
                            "sh": (yt[sm], {m: preds[m][sm] for m in models})}
        pools["S3"] = {
            "id": (yt[:, :, sp3.train_sensors],
                   {m: preds[m][:, :, sp3.train_sensors] for m in models}),
            "sh": (yt[:, :, sp3.test_sensors],
                   {m: preds[m][:, :, sp3.test_sensors] for m in models})}

        block = {}
        for sname, sides in pools.items():
            cell = {}
            for side in ("id", "sh"):
                yts, pd = sides[side]
                cell[side] = {m: pool_mae(yts, pd[m]) for m in models}
            # parity: frozen relative degradation vs banked artifacts
            for m, src in (("chronos2",
                            fdm[ds][SKEY[sname]]["rel_degradation"]["avg"]),
                           ("timesfm",
                            tf[ds][SKEY[sname]]["rel_degradation"]["avg"])):
                rel = (cell["sh"][m] - cell["id"][m]) / cell["id"][m]
                assert abs(rel - src) < 2e-3, (ds, sname, m, rel, src)
                cell[f"rel_deg_{m}"] = rel
            for m in ("chronos2", "timesfm"):
                sk_id = 1.0 - cell["id"][m] / cell["id"]["ha"]
                sk_sh = 1.0 - cell["sh"][m] / cell["sh"]["ha"]
                cell[f"skill_id_{m}"] = sk_id
                cell[f"skill_sh_{m}"] = sk_sh
                cell[f"delta_skill_{m}"] = sk_id - sk_sh
            block[sname] = cell
            print(f"[E2] {ds} {sname}: dskill C2={block[sname]['delta_skill_chronos2']:+.3f} "
                  f"TFM={block[sname]['delta_skill_timesfm']:+.3f}", flush=True)
        out["datasets"][ds] = block
    (SH / "e2_skill_grid.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ E5
def corr18(sev_norm, y_by_cell):
    xs, ys = [], []
    for ds in DATASETS:
        for s in ("S1", "S2", "S3"):
            for m in ("chronos2", "timesfm"):
                xs.append(sev_norm[ds][s])
                ys.append(y_by_cell[(ds, s, m)])
    r, p = spearmanr(xs, ys)
    return float(r), float(p)


def exact_perm9(x9, y9):
    rx, ry = rankdata(x9), rankdata(y9)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    obs = float(np.mean(rx * ry))
    count, total = 0, 0
    for perm in itertools.permutations(range(9)):
        r = float(np.mean(rx * ry[list(perm)]))
        count += (abs(r) >= abs(obs) - 1e-12)
        total += 1
    return obs, count / total


def e5(e1_out, e2_out):
    out = {"spec": "decisions.md D37 E5"}
    deg, dsk = {}, {}
    for ds in DATASETS:
        for s in ("S1", "S2", "S3"):
            c = e2_out["datasets"][ds][s]
            for m in ("chronos2", "timesfm"):
                deg[(ds, s, m)] = c[f"rel_deg_{m}"]
                dsk[(ds, s, m)] = c[f"delta_skill_{m}"]

    out["correlations"] = {}
    for vname, block in e1_out["variants"].items():
        sev = {ds: block[ds]["normalized"] for ds in DATASETS}
        ent = {}
        for yname, yv in (("rel_deg", deg), ("delta_skill", dsk)):
            r18, p18 = corr18(sev, yv)
            x9 = np.array([sev[ds][s] for ds in DATASETS
                           for s in ("S1", "S2", "S3")])
            y9 = np.array([np.mean([yv[(ds, s, "chronos2")],
                                    yv[(ds, s, "timesfm")]])
                           for ds in DATASETS for s in ("S1", "S2", "S3")])
            rho9, p9 = exact_perm9(x9, y9)
            lodo = {}
            for skip in DATASETS:
                keep = [i for i, ds in enumerate(
                    [d for d in DATASETS for _ in range(3)]) if ds != skip]
                r6, _ = spearmanr(x9[keep], y9[keep])
                lodo[skip] = float(r6)
            ent[yname] = {"rho_18": r18, "p_18_naive": p18,
                          "rho_9": rho9, "p_9_exact_twosided": p9,
                          "lodo_rho_6": lodo}
            print(f"[E5] {vname}/{yname}: rho18={r18:+.3f} rho9={rho9:+.3f} "
                  f"p9={p9:.4f} lodo={ {k: round(v,2) for k,v in lodo.items()} }",
                  flush=True)
        out["correlations"][vname] = ent

    # ---- day-block bootstrap, LargeST-SD S1 ----
    ds = "largest_sd"
    yt = truth(ds).astype(np.float32)
    n_test, _, N = yt.shape
    anc = frozen_anchors(ds)["test"]
    _, ts, _ = load_raw(ds)
    models = list(MODELS[ds])
    loaded = {m: load_preds(m, ds, n_test, N, yt) for m in models}
    n_use = min(min(yp.shape[0] for yp, _ in loaded.values()), n_test)
    yt, anc = yt[:n_use], anc[:n_use]
    preds = {m: loaded[m][0][:n_use] for m in models}
    bm = classify_targets(anc, np.asarray(is_weekday(ts)))
    sm = classify_targets(anc, np.asarray(is_weekend(ts)))
    day = np.asarray(ts)[np.asarray(anc) + 1].astype("datetime64[D]")

    def day_tables(mask):
        idx = np.flatnonzero(mask)
        days = day[idx]
        uniq = np.unique(days)
        sums = {m: np.zeros(len(uniq)) for m in models}
        cnts = np.zeros(len(uniq))
        for k, d in enumerate(uniq):
            rows = idx[days == d]
            ytd = yt[rows].astype(np.float64)
            # exact replica of eval.metrics._valid_mask (mask_zero=True)
            valid = (~np.isnan(ytd)) & (np.abs(ytd) > 1e-3)
            cnts[k] = valid.sum()
            for m in models:
                err = np.abs(preds[m][rows].astype(np.float64) - ytd)
                sums[m][k] = err[valid].sum()
        return uniq, sums, cnts

    rng = np.random.default_rng(SEED)
    u_id, s_id, c_id = day_tables(bm)
    u_sh, s_sh, c_sh = day_tables(sm)
    print(f"[E5] SD-S1 day blocks: {len(u_id)} weekday, {len(u_sh)} weekend",
          flush=True)
    # parity: full-pool MAE from day tables == direct pool MAE (chronos2)
    direct = pool_mae(yt[sm], preds["chronos2"][sm])
    tabled = s_sh["chronos2"].sum() / c_sh.sum()
    assert abs(direct - tabled) < 1e-6, (direct, tabled)

    stats = {m: [] for m in ("chronos2", "timesfm")}
    taus = []
    for _ in range(N_BOOT_DAYS):
        bi = rng.integers(0, len(u_id), len(u_id))
        bs = rng.integers(0, len(u_sh), len(u_sh))
        mae_id = {m: s_id[m][bi].sum() / c_id[bi].sum() for m in models}
        mae_sh = {m: s_sh[m][bs].sum() / c_sh[bs].sum() for m in models}
        for m in ("chronos2", "timesfm"):
            stats[m].append((mae_sh[m] - mae_id[m]) / mae_id[m])
        t, _ = kendalltau([mae_id[m] for m in models],
                          [mae_sh[m] for m in models])
        taus.append(t)
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    out["sd_s1_day_block"] = {
        "n_boot": N_BOOT_DAYS, "seed": SEED,
        "n_weekday_days": int(len(u_id)), "n_weekend_days": int(len(u_sh)),
        "rel_deg_ci_chronos2": q(stats["chronos2"]),
        "rel_deg_ci_timesfm": q(stats["timesfm"]),
        "tau_ci": q(taus),
        "tau_point_check": float(kendalltau(
            [s_id[m].sum() / c_id.sum() for m in models],
            [s_sh[m].sum() / c_sh.sum() for m in models])[0]),
    }
    print(f"[E5] SD-S1 day-block: degC2 CI={out['sd_s1_day_block']['rel_deg_ci_chronos2']} "
          f"tau CI={out['sd_s1_day_block']['tau_ci']} "
          f"tau point={out['sd_s1_day_block']['tau_point_check']:.3f}", flush=True)
    (SH / "e5_grouped_stats.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r1 = e1()
    r2 = e2()
    r5 = e5(r1, r2)
    print(f"[done] E1+E2+E5 job={os.environ.get('SLURM_JOB_ID')}", flush=True)
