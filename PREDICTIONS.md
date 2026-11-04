# Registered predictions

We wrote these predictions on 2026-06-08, before any shift evaluation
ran. P4 was replaced by P4′ on 2026-07-04, before any adapter was
trained. The predictions are worded as recorded; only our internal
section labels are removed. Table VI of the paper reports the outcomes.

**P1.** Ranking stability τ will be lower for S1 (weekday→weekend) than
for S2 (peak→off-peak).

**P2.** Graph-based models (DCRNN, GWN, STGCN) will degrade more on S3
(spatial holdout) than per-sensor or sequence models (HA, ARIMA,
Chronos-2, TimesFM), because graph models exploit train-time adjacency
that does not extend to the held-out region.

**P3.** STAEformer will hold first or second rank under S2 but drop ≥ 2
positions under S1.

**P4.** TTRA on a frozen Chronos-2 will reduce shift-induced MAE
degradation by ≥ 10% on at least two of the three shifts (S1, S2, S3).

**P4′ (2026-07-04).** TTRA reduces frozen Chronos-2's SD-S1 degradation
by ≥10% on average, with the effect strongest at long horizons; reported
per-horizon.

Why P4 was replaced: the frozen-degradation results showed that frozen
Chronos-2 does not degrade on most settings, so "reduce degradation on
two of three shifts" could not be tested. P4′ targets the one setting
where both frozen models degrade.

**P5.** Inner-loop step count > 0 outperforms step count = 0 (i.e. the
*test-time* update, not just the registers themselves, is what matters).

## Outcomes

| ID | Outcome | Verdict | Where to check |
|---|---|---|---|
| P1 | Held on LargeST-SD (τ 0.333 < 0.867); ties on the other two datasets | Partially held | `artifacts/results/rankings.json` |
| P2 | Needs graph models retrained on the S3 split, which the protocol left out | Not testable | no run |
| P3 | STAEformer is rank 1 under both shifts on every dataset where it was ranked | Falsified | `artifacts/results/rankings.json` |
| P4′ | Adapted MAE 27.80 is above frozen 26.14 at every horizon | Falsified | `artifacts/results/runs/ttra/w6_gate.json` |
| P5 | Not run, because the adapter already failed P4′ | Not evaluated | no run |
