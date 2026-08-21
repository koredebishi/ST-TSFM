# Severe but Harmless: Stress-Testing Shift Severity for Frozen Time-Series Forecasters

Code, result files, and verification for the paper. Everything the paper
reports can be re-derived here in seconds on CPU.

## Quick check

```bash
pip install -r requirements.txt
python verify.py
```

Reads only the JSON files in `results/` and prints each reported value beside
the one recomputed from the artifacts.

## Layout

```
verify.py                     reproduces every number in the paper
requirements.txt              dependencies for verify.py and the figures
code/shift/                   shift construction, severity, rankings, figures
code/eval/metrics.py          masked MAE and the other metrics
code/data_loaders/            window construction, frozen anchors
code/scripts/verification/    Proposition 1 checks, batch-independence check
code/scripts/experiments/     shift-pool builders, data converters
results/*.json                severity, degradation, rankings, router, D39 arms
results/baselines/*/*/        per-model test metrics
results/verification/         Proposition 1 and batch-independence outputs
```

## Where each paper number comes from

| Paper element | Result file | Producing script |
|---|---|---|
| Table II (severity, degradation) | `frozen_degradation_matrix.json`, `timesfm_*_degradation.json`, `e1_unified_severity.json` | `scripts/verification/frozen_degradation_matrix.py`, `shift/severity.py` |
| Table III (all-model MAE) | `tbig_grid.json`, `staeformer_sd_cells.json` | `shift/make_tbig.py`, `shift/staeformer_sd_cells.py` |
| Table IV (Kendall tau) | `t3_rankings.json` | `shift/ranking.py` |
| Table V (repair interventions) | `tbig_grid.json`, `e4_router.json` | `shift/e4_router.py` |
| Finding 1 correlations | `e5b_ruler_correlations.json`, `e5_grouped_stats.json` | `shift/e5b_ruler_correlations.py` |
| Skill-adjusted analysis | `e2_skill_grid.json` | `shift/e125_review_response.py` |
| Sensitivity analysis | `gap2_split_sensitivity.json`, `bay_s3_sweep.json` | `shift/gap2_split_sensitivity.py` |
| Proposition 1 verification | `prop1_equivariance.json`, `e3_timesfm_equivariance.json` | `scripts/verification/prop1_equivariance.py`, `scripts/verification/e3_timesfm_equivariance.py` |
| Batch independence | `verification/batch_independence.json` | `scripts/verification/batch_independence.py` |
| Figures 1--6 | as above | `shift/make_fig_shiftgrid.py`, `shift/make_fig_rulers.py`, `shift/make_figures.py` |

Each result file records the SLURM job identifier of the run that produced it.

## Re-running the experiments

Full reproduction needs the datasets (METR-LA and PEMS-BAY from the DCRNN
release, LargeST-SD from the LargeST release), GPU access, and the released
Chronos-2 and TimesFM checkpoints. The scripts in `code/` are the ones used;
they expect the dataset layout described in the paper. Sensor-coordinate CSVs
come from the DCRNN repository; set `DCRNN_DATA` to point at a local clone.

`verify.py` needs only the result files in this repository; the cached
prediction arrays used to produce them are not distributed.

## Pre-registration

Predictions were registered before the shift evaluation ran. The decision log
with the registered predictions and their outcomes accompanies this release.
