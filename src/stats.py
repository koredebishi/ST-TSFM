"""Correlation of severity with degradation.

Severity is in different units on each dataset (mph or vehicles), so it is
divided by the dataset's largest severity before datasets are combined.

Two ways to correlate:
- over the 18 model-setting evaluations (9 settings x 2 models). The two
  models share each setting's severity, so the 18 points are not
  independent; this rho is descriptive only.
- over the 9 settings, with the two models' degradations averaged. The
  p-value is exact: the observed rho is compared with the rho of every
  one of the 9! = 362,880 orderings.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import rankdata, spearmanr


def normalize_within_dataset(sev):
    """sev[dataset][shift] -> the same divided by the dataset's maximum."""
    return {d: {s: v / max(by_shift.values()) for s, v in by_shift.items()}
            for d, by_shift in sev.items()}


def spearman(x, y):
    return float(spearmanr(x, y).statistic)


def exact_test(x, y):
    """Spearman rho over a small sample and its exact two-sided p-value."""
    rx, ry = rankdata(x), rankdata(y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    rho = float(np.mean(rx * ry))
    hits = total = 0
    for order in itertools.permutations(range(len(ry))):
        hits += abs(np.mean(rx * ry[list(order)])) >= abs(rho) - 1e-12
        total += 1
    return rho, hits / total


def correlate(sev_norm, by_model):
    """sev_norm[d][s]; by_model[model][d][s] -> rho over 18 and over 9."""
    models = list(by_model)
    cells = [(d, s) for d in sev_norm for s in sev_norm[d]]
    x18 = [sev_norm[d][s] for d, s in cells for _ in models]
    y18 = [by_model[m][d][s] for d, s in cells for m in models]
    x9 = [sev_norm[d][s] for d, s in cells]
    y9 = [np.mean([by_model[m][d][s] for m in models]) for d, s in cells]
    rho9, p9 = exact_test(x9, y9)
    return {"rho_18": spearman(x18, y18), "rho_9": rho9, "p_9": p9}
