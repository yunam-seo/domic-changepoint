#!/usr/bin/env python
"""The Supplementary Section B.7 bandwidth sweeps for D3 and D4, calibrated against the matching null.

`run_domi_bandwidth.py` and `run_hsic_fullgram.py` both draw their 500 null replicates at
`SCENARIOS[s]["levels"][0]` and apply the resulting threshold to every alternative level -- the same
construction as `run_experiment.run_e1`, which is exact on D1, D2 and M1, whose null does not
depend on `level`. On D3 (Gaussian copula at Kendall's tau = level) and D4 (mixed-sign dependence at
r = level) the null does depend on it, so those columns are calibrated here against the
alternative's own pre-change regime held throughout (Section 5).

This runner repeats both sweeps on those two scenarios with the null drawn at the alternative's own
level. Both statistics keep their own five bandwidths and their own calibration; the only
difference from the level-0 sweeps is `li` in the null map.

ONE CELL PER INVOCATION, with per-cell record files -- see the header of
`run_e1_matched_null.py` for why.

Run:  python 00_SRC/run_bandwidth_matched_null.py --stat domi --scen D3 --li 1
Writes 04_DAOU/EXPERIMENT/bandwidth_matched_null/.
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

from dots.extras import studentize  # noqa: E402
from dots.synth import SCENARIOS  # noqa: E402
from dots.persist import save_records  # noqa: E402
import run_domi_bandwidth as QB  # noqa: E402
import run_hsic_fullgram as HF  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "bandwidth_matched_null")
CFG = QB.CFG

# (job, bandwidth keys, scored modes, label).  DOMI is scored studentized only, as in its runner;
# the full-Gram HSIC is scored in both forms and reported at its better one, as in Section 5.
STATS = {"domi":  (QB._job, list(QB.GAM.keys()), ("gs",), "DOMI"),
         "hsic": (HF._job, list(HF.MULTS),      ("g", "gs"), "HSIC-fullgram")}


def calibrate(nulls, keys, reps, fpr):
    half = reps // 2
    cal = {}
    for k in keys:
        N = np.array([d[0][k] for d in nulls])
        mu0, sd0 = N[:half].mean(0), N[:half].std(0)
        cal[k] = dict(mu0=mu0, sd0=sd0,
                      raw=float(np.quantile(N.max(1), 1 - fpr)),
                      gs=float(np.quantile([studentize(c, mu0, sd0).max() for c in N[half:]],
                                           1 - fpr)))
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", required=True, choices=sorted(STATS))
    ap.add_argument("--scen", required=True)
    ap.add_argument("--li", type=int, required=True)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--procs", type=int, default=14)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    job, keys, modes, mname = STATS[a.stat]
    s, li = a.scen, a.li
    level = SCENARIOS[s]["levels"][li]
    tag = f"{a.stat}_{s}_{level}"

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        with open(os.path.join(OUT, "run.log"), "a") as f:
            f.write(line + "\n")

    rows, recs = [], []
    with Pool(a.procs) as pool:
        t0 = time.time()
        nulls = pool.map(job, [(s, li, r, True) for r in range(a.reps)], chunksize=4)
        cal = calibrate(nulls, keys, a.reps, CFG["fpr"])
        say(f"{mname} {s} level={level}: matched null {a.reps} reps {time.time()-t0:.0f}s")

        t1 = time.time()
        alts = pool.map(job, [(s, li, r, False) for r in range(a.reps)], chunksize=4)
        say(f"{mname} {s} level={level}: alt {a.reps} reps {time.time()-t1:.0f}s")

    for k in keys:
        c = cal[k]
        for mode in modes:
            hit = det = 0
            for i, (out, grid, tau) in enumerate(alts):
                cur = out[k] if mode == "g" else studentize(out[k], c["mu0"], c["sd0"])
                th = c["raw"] if mode == "g" else c["gs"]
                mx = float(np.nanmax(cur)); tau_hat = int(grid[int(np.nanargmax(cur))])
                d_ = mx > th; h_ = d_ and abs(tau_hat - tau) <= CFG["tol"]
                det += d_; hit += h_
                recs.append({"scen": s, "level": level, "rep": i, "bandwidth_key": k,
                             "mode": mode, "max": mx, "tau_hat": tau_hat,
                             "detected": bool(d_), "power_hit": bool(h_)})
            rows.append(dict(scen=s, level=level, method=mname, bandwidth_key=k, mode=mode,
                             null_level=level, power=round(hit / len(alts), 3),
                             detect_rate=round(det / len(alts), 3)))
    save_records(OUT, f"records_alt_{tag}.csv", recs, {})
    with open(os.path.join(OUT, f"results_{tag}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    say(f"{mname} {s} level={level}: " +
        ", ".join(f"x{r['bandwidth_key']}|{r['mode']}={r['power']:.3f}" for r in rows))


if __name__ == "__main__":
    main()
