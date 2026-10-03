#!/usr/bin/env python
"""Marginal-shape control for scenario S2 (code D2): scenario M2 (Section 6.2, Supplementary Section B.15).

In S2 the change y = sqrt(1-a^2) e1 + a |x| e2 introduces uncorrelated dependence and, with it, a
change in the shape of the Y margin (variance 1, fourth moment 3 + 6a^4). Scenario M2 reproduces
that marginal change exactly while keeping X and Y independent: the multiplier |x'| is drawn
independently of x. Any detection under M2 is a response to the marginal change alone, so M2 is
to S2 what M1 is to a change in the X margin.

Protocol: that of Table 1 (run_e1_matched_null.py) unchanged -- 500 null and 500 alternative
replicates per level, the null being the pre-change regime held throughout, first half of the null
for the pointwise moments and second half for the studentized threshold at a false-alarm rate of
0.05, and the full E1 method set. The reported quantity is the detection rate (any detection).

Run:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_marginal_shape_control.py --li 1
      (--li 0, 1, 2 for a = 0.5, 0.7, 0.9), then --combine
Writes 04_DAOU/EXPERIMENT/marginal_shape_control/{results_M2_<a>.csv,
records_{null,alt}_M2_<a>.csv.gz, results.csv, run.log}
"""
from __future__ import annotations

import csv
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import run_e1_matched_null as M  # noqa: E402
from dots.synth import SCENARIOS  # noqa: E402

M.OUT = os.path.join(os.path.dirname(SRC), "04_DAOU", "EXPERIMENT", "marginal_shape_control")


def combine():
    rows = []
    for level in SCENARIOS["M2"]["levels"]:
        p = os.path.join(M.OUT, f"results_M2_{level}.csv")
        if os.path.exists(p):
            rows += list(csv.DictReader(open(p)))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(M.OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"combined {len(rows)} rows")


if __name__ == "__main__":
    if "--combine" in sys.argv:
        combine()
    else:
        sys.argv += ["--scen", "M2"]
        M.main()
