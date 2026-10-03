#!/usr/bin/env python
"""Run ONE (scenario, level) cell of E3 (Supplementary Table B.8), writing its own results and records.

`run_experiment.run_e3` drives all six cells from a single Pool and writes `results.csv` only
at the end. This runner does one cell per process and writes a per-cell aggregate and per-replicate
records (e3/records_*.csv.gz).

Run:  python 00_SRC/run_e3_cell.py --scen D4 --li 2
Writes 04_DAOU/EXPERIMENT/e3/{results_<scen>_<level>.csv,records_*.csv.gz,run.log}.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from dots.synth import SCENARIOS  # noqa: E402
from dots.persist import save_records  # noqa: E402
import run_experiment as E  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", required=True)
    ap.add_argument("--li", type=int, required=True)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--procs", type=int, default=14)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = dict(E.CFG)
    cfg["K"] = 99                      # as run_e3 sets it
    scen, li = a.scen, a.li
    level = SCENARIOS[scen]["levels"][li]
    tag = f"{scen}_{level}"

    t0 = time.time()
    with Pool(a.procs) as pool:
        nulls = pool.map(E._e3_job, [(scen, li, r, True, cfg) for r in range(a.reps)], chunksize=1)
        alts = pool.map(E._e3_job, [(scen, li, r, False, cfg) for r in range(a.reps)], chunksize=1)
    save_records(OUT, f"records_null_{tag}.csv", nulls, {"scen": scen, "level": level})
    save_records(OUT, f"records_alt_{tag}.csv", alts, {"scen": scen, "level": level})

    rows = []
    for m in nulls[0]:
        fa = float(np.mean([d[m]["detect"] for d in nulls]))
        pw = float(np.mean([d[m]["detect"] and abs(d[m]["tau_hat"] - cfg["tau"]) <= cfg["tol"]
                            for d in alts]))
        rows.append(dict(scen=scen, level=level, method=m, K=cfg["K"], fpr_perm=fa, power_perm=pw))
    with open(os.path.join(OUT, f"results_{tag}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    line = (f"[{time.strftime('%H:%M:%S')}] {scen} {level}: {time.time()-t0:.0f}s | "
            + ", ".join(f"{r['method']}: P={r['power_perm']:.2f} FPR={r['fpr_perm']:.2f}"
                        for r in rows))
    print(line, flush=True)
    with open(os.path.join(OUT, "run.log"), "a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
