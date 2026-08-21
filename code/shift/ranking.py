"""Ranking-stability statistic — Kendall's τ with bootstrap CI (§5.2, T3).

Given per-sensor MAE arrays for a set of models under two settings
(in-distribution and shifted), compute:
- the point Kendall's τ between the two model rankings (models ranked by
  MAE aggregated over sensors), and
- a bootstrap 95% CI over sensor subsamples: resample sensors with
  replacement, re-aggregate, re-rank, recompute τ.

Interpretation thresholds (plan §5.2): τ < 0.7 → re-ranking finding;
τ ≥ 0.9 everywhere → stability finding. Either is publishable.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

DEFAULT_BOOT = 1000
DEFAULT_SEED = 2023


def _aggregate_rank(per_sensor: np.ndarray) -> np.ndarray:
    """per_sensor: (M models, N sensors) MAE -> rank vector (M,).

    Rank 0 = best (lowest aggregate MAE). Ties broken by model order,
    which is deterministic because model order is the caller's dict order.
    """
    agg = np.nanmean(per_sensor, axis=1)
    return np.argsort(np.argsort(agg))


def kendall_tau(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    tau, _p = kendalltau(rank_a, rank_b)
    return float(tau)


def ranking_stability(id_mae: dict[str, np.ndarray],
                      shift_mae: dict[str, np.ndarray],
                      n_boot: int = DEFAULT_BOOT,
                      seed: int = DEFAULT_SEED) -> dict:
    """τ between ID and shifted model rankings + bootstrap 95% CI.

    id_mae / shift_mae: model name -> (N,) per-sensor MAE. Both dicts must
    share the same keys; sensor axis must align across models and settings.
    """
    models = list(id_mae)
    assert models == list(shift_mae), "model sets/order differ"
    A = np.stack([np.asarray(id_mae[m], dtype=float) for m in models])
    B = np.stack([np.asarray(shift_mae[m], dtype=float) for m in models])
    assert A.shape == B.shape, f"sensor axes differ: {A.shape} vs {B.shape}"
    M, N = A.shape
    assert M >= 3, "Kendall tau needs >= 3 models to be meaningful"

    tau_point = kendall_tau(_aggregate_rank(A), _aggregate_rank(B))

    rng = np.random.default_rng(seed)
    taus = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N, size=N)
        taus[i] = kendall_tau(_aggregate_rank(A[:, idx]),
                              _aggregate_rank(B[:, idx]))
    return {
        "tau": tau_point,
        "ci_low": float(np.percentile(taus, 2.5)),
        "ci_high": float(np.percentile(taus, 97.5)),
        "boot_tau_mean": float(taus.mean()),
        "n_models": M,
        "n_sensors": N,
        "n_boot": n_boot,
        "seed": seed,
        "models_ranked_id": [models[i] for i in
                             np.argsort(np.nanmean(A, axis=1))],
        "models_ranked_shift": [models[i] for i in
                                np.argsort(np.nanmean(B, axis=1))],
        "spec": "paper_plan §5.2",
    }
