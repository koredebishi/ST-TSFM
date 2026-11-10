# Severe but Harmless: Stress-Testing Shift Severity for Frozen Time-Series Forecasters

Code and results for the IEEE BigData 2026 paper by Korede R. Bishi,
Dominic Okonkwo and Adetayo Okunoye.

**Question.** When the data shifts, a common check is to measure how far
the new values are from the old ones and to treat a larger distance as a
larger risk. Does that distance predict how much frozen time-series
foundation models (Chronos-2, TimesFM 2.5) degrade? We test this on three
traffic datasets (METR-LA, PEMS-BAY, LargeST-SD) under three shifts
(weekday to weekend, peak to off-peak, a held-out cluster of sensors).

## Quick start

```bash
pip install -r requirements/analysis.txt
python scripts/make_tables.py
```

`make_tables.py` checks 90 reported values against files in
`artifacts/results/`, showing each paper value beside the released value.
It runs in seconds on CPU and needs no data, GPU or model weights.
`paper_results_walkthrough.ipynb` explains the main tables and figures step by step.

## Layout

```
src/                  analysis code: data, splits, metrics, severity,
                      degradation, rankings, statistics, router
scripts/
  make_results.py     datasets + saved forecasts -> artifacts/results/
  make_tables.py      artifacts/results/ -> the paper's tables
artifacts/
  results/            the result files behind every table and figure
  results/runs/       records from the training and GPU runs
  figures/            the figures as printed in the paper
  manifest.json       the saved forecasts (not included), with hashes
forecasting/          the code that produced the forecasts and the adapter
paper_results_walkthrough.ipynb  annotated tables and figures from artifacts/results/
PREDICTIONS.md        predictions registered before the experiments
```

## Where each result comes from

| Paper | File in `artifacts/results/` | Code |
|---|---|---|
| Table I (compute) | `runs/test_metrics/` | `forecasting/` |
| Table II (severity, degradation) | `severity.json`, `degradation.json` | `src/severity.py`, `src/degradation.py` |
| Table III (MAE of every model) | `mae_grid.json` | `scripts/make_results.py` |
| Table IV (Kendall tau) | `rankings.json`, `day_block.json` | `src/ranking.py` |
| Table V (weekend interventions) | `runs/retraining/`, `runs/logs/`, `runs/ttra/`, `router.json`, `context_length.json` | `forecasting/`, `src/router.py` |
| Table VI (predictions) | see `PREDICTIONS.md` | |
| Severity vs degradation (IV-C) | `correlations.json`, `skill_change.json`, `standardized_severity.json`, `relative_metrics.json` | `src/stats.py`, `src/severity.py`, `src/degradation.py` |
| Saturday vs Sunday (IV-E) | `weekend_days.json` | `src/splits.py` |
| Proposition 1 (III-F) | `runs/equivariance/` | `forecasting/scripts/verification/` |
| Split sensitivity (Section V) | `sensitivity.json` | `src/sensitivity.py` |
| Figs. 1-6 | `artifacts/figures/` | |

`ranking_with_staeformer.json` holds the LargeST-SD ranking with STAEformer
added (M = 7), which the paper mentions but does not tabulate.

## Recomputing the results

`scripts/make_results.py` rebuilds every file in `artifacts/results/`
except `runs/` from the datasets and the saved forecasts. The forecasts
(3.8 GB) are not included; `artifacts/manifest.json` lists each file with
its shape and hash. To make them yourself, run the code in `forecasting/`
(a GPU is needed for Chronos-2 and TimesFM; TimesFM took about 193
GPU-hours on all three datasets).

**Data.**
- METR-LA and PEMS-BAY: download `metr-la.h5` and `pems-bay.h5` as described
  in the [DCRNN README](https://github.com/liyaguang/DCRNN), run its
  `generate_training_data` script, and place the files under
  `datasets/METR-LA/` and `datasets/PEMS-BAY/` (`processed/test.npz` for
  the windows). Sensor coordinates are in DCRNN's `data/sensor_graph/`;
  set `DCRNN_SENSOR_GRAPH` to that folder.
- LargeST-SD (2019): follow the [LargeST README](https://github.com/liuxu77/LargeST)
  to build the SD subset, and place `sd_his_2019.h5`, `sd_meta.csv` and the
  `2019/` folder (`his.npz`, `idx_*.npy`) under `datasets/LargeST-SD/`.

```bash
pip install -r requirements/analysis.txt
export ST_TSFM_FORECASTS=/path/to/forecasts   # default: ./forecasts
python scripts/make_results.py
python scripts/make_tables.py
```

`forecasting/` is the code as it ran. Its scripts expect `datasets/` and
`results/` inside that folder; see `forecasting/README.md`. DCRNN, Graph
WaveNet and STAEformer were trained with their authors' code (links in
`THIRD_PARTY_NOTICES.md`).

## License

MIT, see `LICENSE`. Third-party code and data: `THIRD_PARTY_NOTICES.md`.

## Citation

If you use this code or these results, please cite:

```bibtex
@inproceedings{bishi2026severe,
  title     = {Severe but Harmless: Stress-Testing Shift Severity for
               Frozen Time-Series Forecasters},
  author    = {Bishi, Korede R. and Okonkwo, Dominic and Okunoye, Adetayo},
  booktitle = {IEEE International Conference on Big Data (BigData)},
  year      = {2026}
}
```

The DOI will be added once the paper appears in IEEE Xplore.
