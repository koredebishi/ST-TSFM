#!/usr/bin/env python3
"""Dump S1/S2/S3 split summaries for all datasets to one JSON artifact.

Evidence file for the notes (governance rule 3: verified numbers only).
Output: results/shift/splits_summary.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shift.severity import severity_spatial, severity_temporal  # noqa: E402
from shift.splits import s1_split, s2_split, s3_split  # noqa: E402

OUT = REPO / "results/shift/splits_summary.json"
SEV_OUT = REPO / "results/shift/severity_summary.json"


def main():
    summaries, severities = {}, {}
    for ds in ("metr_la", "pems_bay", "largest_sd"):
        s1, s2, s3 = s1_split(ds), s2_split(ds), s3_split(ds)
        summaries[ds] = {"S1": s1.summary(), "S2": s2.summary(),
                         "S3": s3.summary()}
        print(f"[splits] {ds}: " + json.dumps(summaries[ds]), flush=True)
        # F1 matrix data: severity per (dataset, shift)
        severities[ds] = {
            "S1": severity_temporal(ds, s1),
            "S2": severity_temporal(ds, s2),
            "S3": severity_spatial(ds, s3),
        }
        print(f"[severity] {ds}: " + json.dumps(severities[ds]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"host": os.uname().nodename, "summaries": summaries}, indent=2))
    SEV_OUT.write_text(json.dumps(
        {"host": os.uname().nodename, "severities": severities}, indent=2))
    print(f"[splits] wrote {OUT}")
    print(f"[severity] wrote {SEV_OUT}")


if __name__ == "__main__":
    main()
