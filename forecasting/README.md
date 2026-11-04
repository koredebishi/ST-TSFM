# forecasting/

The code that produced the saved forecasts, the training sets for the
retrained models, the test-time adapter (TTRA) and the equivariance
checks. It is kept as it ran; only comments were tidied. Each script
expects `datasets/` and `results/` inside this folder (for example as
links to the repository's own copies).

| Script | What it does | Result used in the paper |
|---|---|---|
| `baselines/tsfm_zeroshot_eval.py` | Chronos-2 and TimesFM forecasts on the test windows (512-step context; `--context-len` to change it, `--window-start/--window-end` and `--merge` to split a run into shards) | Tables I-IV |
| `baselines/arima/run.py` | ARIMA(3,0,1) per sensor | Table III |
| `scripts/experiments/build_s1_pools.py` | weekday training sets for the retrained models | Table V |
| `scripts/experiments/build_d39_pools.py` | weekend and size-matched training sets | Table V |
| `scripts/experiments/build_staeformer_sd.py` | LargeST-SD in STAEformer's data format | Table III |
| `mechanism/train_ttra.py` | trains the adapter on weekdays, tests on weekends | Table V, Fig. 5 |
| `mechanism/diag_w6_pinball.py` | pinball loss of frozen and adapted model | Fig. 5 |
| `scripts/verification/prop1_equivariance.py` | Chronos-2: f(ax+b) against a f(x)+b | Section III-F |
| `scripts/verification/e3_timesfm_equivariance.py` | the same for TimesFM | Section III-F |
| `scripts/verification/batch_independence.py` | each sensor is forecast on its own | Section IV-A |

The historical average is in `src/historical_average.py`. DCRNN, Graph
WaveNet and STAEformer were trained with their authors' code.
