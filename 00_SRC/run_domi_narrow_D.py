#!/usr/bin/env python
"""Does DOMI's loss of power at a quarter-median bandwidth come from the D=8 feature budget?

Supplementary Table B.2 shows DOMI at sigma x 1/4 well below the full-Gram HSIC on S1 (r=0.35:
0.06 against 0.74), although it gains power at sigma x 1/2. A possible explanation is
representational rather than functional: a fixed draw of D random Fourier features approximates a
Gaussian kernel with an error that grows as the kernel narrows relative to the data scale, and the
full Gram matrix has no such budget. If that is the cause, raising D at the same bandwidth should
recover the lost power; if it is not, D will not help.

Runs the Section 5 protocol at sigma x 1/4 only, for D in {8, 12, 16}, on S1 (r=0.35) -- the cell
with the largest loss -- and S3 (tau=0.5) -- the cell the narrow kernel is wanted for.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_domi_narrow_D.py --scen D1
Writes 04_DAOU/EXPERIMENT/domi_narrow_D/{results_<scen>.csv,records_<scen>.csv.gz}.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from dots.extras import studentize  # noqa: E402
from dots.synth import SCENARIOS, generate  # noqa: E402
import run_domi_bandwidth as DB  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "domi_narrow_D")
CFG = DB.CFG
GMULT = 16.0                      # gamma x16  <->  sigma x 1/4
DIMS = [8, 12, 16]
LEVEL = {"D1": 1, "D3": 1}        # r = 0.35 ; tau = 0.5


def _job(a):
    s, li, r, null = a
    smp = generate(s, SCENARIOS[s]["levels"][li], r, null=null,
                   base_seed=CFG["base_seed"], level_idx=li)
    Z = smp["Z"]
    X, Y = Z[:, smp["blocks"][0]], Z[:, smp["blocks"][1]]
    out = {D: DB.curve(X, Y, CFG["w"][s], CFG["n"], D, GMULT)[0] for D in DIMS}
    grid = np.arange(CFG["w"][s], CFG["n"] - CFG["w"][s] + 1)
    return out, grid, int(smp["tau"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", required=True)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--procs", type=int, default=14)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    s, li = a.scen, LEVEL[a.scen]
    level = SCENARIOS[s]["levels"][li]

    t0 = time.time()
    with Pool(a.procs) as pool:
        nulls = pool.map(_job, [(s, li, r, True) for r in range(a.reps)], chunksize=2)
    print(f"[{s}] null {a.reps} reps {time.time()-t0:.0f}s", flush=True)
    half = a.reps // 2
    cal = {}
    for D in DIMS:
        N = np.array([d[0][D] for d in nulls])
        mu0, sd0 = N[:half].mean(0), N[:half].std(0)
        cal[D] = dict(mu0=mu0, sd0=sd0,
                      gs=float(np.quantile([studentize(c, mu0, sd0).max() for c in N[half:]],
                                           1 - CFG["fpr"])))

    t0 = time.time()
    with Pool(a.procs) as pool:
        alts = pool.map(_job, [(s, li, r, False) for r in range(a.reps)], chunksize=2)
    print(f"[{s}] alt  {a.reps} reps {time.time()-t0:.0f}s", flush=True)

    rows, recs = [], []
    for D in DIMS:
        c = cal[D]
        hit = det = 0
        for i, (out, grid, tau) in enumerate(alts):
            cur = studentize(out[D], c["mu0"], c["sd0"])
            mx = float(np.nanmax(cur)); tau_hat = int(grid[int(np.nanargmax(cur))])
            d_ = mx > c["gs"]; h_ = d_ and abs(tau_hat - tau) <= CFG["tol"]
            det += d_; hit += h_
            recs.append({"scen": s, "level": level, "rep": i, "D": D, "sigma_mult": 0.25,
                         "max_gs": mx, "tau_hat": tau_hat, "detected": bool(d_),
                         "power_hit": bool(h_)})
        rows.append(dict(scen=s, level=level, method="DOMI", sigma_mult=0.25, D=D, mode="gs",
                         power=round(hit / len(alts), 3), detect_rate=round(det / len(alts), 3)))
        print(f"   {s} level={level} D={D}: power={rows[-1]['power']:.3f}", flush=True)

    with open(os.path.join(OUT, f"results_{s}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    with gzip.open(os.path.join(OUT, f"records_{s}.csv.gz"), "wt", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(recs[0])); wr.writeheader(); wr.writerows(recs)
    print("written:", OUT)


if __name__ == "__main__":
    main()
