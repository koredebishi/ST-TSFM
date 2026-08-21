#!/usr/bin/env python3
"""Free pre-check for the TTRA W6 gate (ttra_training_spec assumption #1):
does FROZEN zero-shot Chronos-2 actually forecast WEEKENDS worse than
WEEKDAYS on METR-LA? If not, "adapted < frozen" on S1 is ill-posed.

Zero new model compute: re-slices the already-banked Chronos-2 LA forecasts
(results/baselines/chronos2/metr_la/y_pred.npy, from --save-preds) into
weekday-target vs weekend-target test windows and compares MAE via our
eval harness. Uses the SAME classifiers as shift/splits.py (D8).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]  # scripts/verification/ -> repo root
sys.path.insert(0, str(REPO))

from data_loaders.windows import frozen_anchors, load_raw  # noqa: E402
from eval.metrics import all_metrics  # noqa: E402
from shift.splits import classify_targets, is_weekday, is_weekend  # noqa: E402

OUT = REPO / "results/shift/s1_frozen_degradation_metr_la.json"


def main():
    y_pred = np.load(REPO / "results/baselines/chronos2/metr_la/y_pred.npy")
    y_true = np.load(REPO / "datasets/METR-LA/processed/test.npz")["y"][..., 0]
    assert y_pred.shape == y_true.shape, f"{y_pred.shape} vs {y_true.shape}"

    _, ts, _ = load_raw("metr_la")
    test_anchors = frozen_anchors("metr_la")["test"]
    assert len(test_anchors) == len(y_pred), \
        f"anchors {len(test_anchors)} vs preds {len(y_pred)}"

    wd = classify_targets(test_anchors, np.asarray(is_weekday(ts)))  # all-12 weekday
    we = classify_targets(test_anchors, np.asarray(is_weekend(ts)))  # all-12 weekend
    N = y_true.shape[2]

    def block(mask, label):
        yt, yp = y_true[mask], y_pred[mask].astype(np.float64)
        per_h = {h: round(all_metrics(yt[:, h - 1], yp[:, h - 1],
                                      mask_zero=True)["mae"], 4)
                 for h in (3, 6, 12)}
        avg = all_metrics(yt.reshape(-1, N), yp.reshape(-1, N),
                          mask_zero=True)["mae"]
        return {"label": label, "n_windows": int(mask.sum()),
                "mae_h3_6_12": per_h, "mae_avg": round(avg, 4)}

    wk, wkd = block(wd, "weekday-test"), block(we, "weekend-test")
    gap = wkd["mae_avg"] - wk["mae_avg"]
    rel = gap / wk["mae_avg"]
    out = {
        "question": "is frozen zero-shot Chronos-2 worse on weekends than weekdays (METR-LA)?",
        "weekday": wk, "weekend": wkd,
        "weekend_minus_weekday_avg_mae": round(gap, 4),
        "relative_degradation": round(rel, 4),
        "verdict": ("frozen IS shift-degraded -> S1 gate well-posed"
                    if rel > 0.05 else
                    "frozen NOT meaningfully degraded -> S1 gate ill-posed; rethink"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
