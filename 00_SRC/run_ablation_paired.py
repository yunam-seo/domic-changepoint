#!/usr/bin/env python
"""Paired comparison of the entropy and Frobenius functionals (Supplementary Section B.2).

Both functionals are computed on the same replicates, so they are compared by McNemar's exact test
on the discordant pairs of per-replicate detections. Recomputes the alternative replicates of the
ablation cells in which the Frobenius power is at least the entropy power, with the generator,
seeds and thresholds of run_ablation.py (so the powers equal the stored ones), and reports both
powers, the discordant counts and the two-sided exact p-value per cell. Three cells are compared;
Supplementary Section B.2 judges the p-values against a correction for three comparisons.

Run:  python 00_SRC/run_ablation_paired.py --procs 12
Writes 04_DAOU/ABLATION/paired_entropy_vs_frobenius.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from run_ablation import CFG, _alt_job  # noqa: E402
from dots.synth import SCENARIOS  # noqa: E402

DAOU = os.path.join(ROOT, "04_DAOU", "ABLATION")
ENT, FRO = "kQDg-Holevo|g", "kFrob|g"
# the three of the fifteen ablation cells in which the Frobenius power is not below the entropy power
CELLS = [("N1", 0), ("N2", 2), ("N2", 1)]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant counts (binomial at 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--reps", type=int, default=None)
    a = ap.parse_args()
    cfg = dict(CFG)
    if a.reps:
        cfg["reps"] = a.reps
    thresholds = json.load(open(os.path.join(DAOU, "thresholds.json")))

    out = {"config": {"reps": cfg["reps"], "base_seed": cfg["base_seed"],
                      "test": "McNemar exact, two-sided, on per-replicate localised detections",
                      "note": "cells where Frobenius power >= entropy power; in the other "
                              "cells the entropy power is higher"},
           "cells": []}
    print(f"{'cell':<14} {'entropy':>8} {'Frob':>8} {'b':>4} {'c':>4} {'McNemar p':>10}")
    for scen, li in CELLS:
        level = SCENARIOS[scen]["levels"][li]
        thr = thresholds[scen]
        with Pool(a.procs) as pool:
            recs = pool.map(_alt_job, [(scen, li, r, cfg, thr) for r in range(cfg["reps"])],
                            chunksize=2)
        e = [bool(r[ENT]["power_hit"]) for r in recs]
        f = [bool(r[FRO]["power_hit"]) for r in recs]
        b = sum(1 for x, y in zip(e, f) if x and not y)   # entropy only
        c = sum(1 for x, y in zip(e, f) if y and not x)   # Frobenius only
        p = mcnemar_exact(b, c)
        pe, pf = sum(e) / len(e), sum(f) / len(f)
        out["cells"].append({"scen": scen, "level": level, "level_idx": li,
                             "power_entropy": round(pe, 3), "power_frobenius": round(pf, 3),
                             "entropy_only": b, "frobenius_only": c, "mcnemar_p": round(p, 4)})
        print(f"{scen}@{level:<10g} {pe:8.3f} {pf:8.3f} {b:4d} {c:4d} {p:10.4f}")

    worst = min((x["mcnemar_p"] for x in out["cells"]), default=1.0)
    out["min_p"] = worst
    out["min_p_above_0_05"] = bool(worst > 0.05)
    print(f"\nsmallest p = {worst:.4f} (Bonferroni level for {len(CELLS)} comparisons: "
          f"{0.05 / len(CELLS):.4f})")
    p = os.path.join(DAOU, "paired_entropy_vs_frobenius.json")
    json.dump(out, open(p, "w"), indent=1)
    print("written:", p)


if __name__ == "__main__":
    main()
