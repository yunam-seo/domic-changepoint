#!/usr/bin/env python
"""Stage two (re-test) of the US financial panel (Section 6.8; Supplementary Section B.22).

For each pair that survives the Benjamini-Hochberg step of the main run (us_panel/summary.json), the stage two
of Section 4.7 at the maximizer of the pair's primary statistic, with the functions of run_e6_retest.py:
daily grid (unit = 1 trading day); block length from the whole-record diagnostic (us_panel/diag.json); the
largest window centered on the break whose half-width is a whole number of blocks; minimum segment one block and
at least 6% of the window; dependence test (weighted DOMI difference, ranks and features within the window,
D = 8, seeds 2026/2027) and marginal-scale test of each variable, K = 999 block permutations each, symmetric
studentization; permutation seeds 710000 + 100 * i + {1, 2, 3} (i = 0, 1, 2 for SPX-GLD, SPX-TNX, SPX-DXY);
20 further feature draws of the dependence test (seeds 10000 + 2j, 10001 + 2j) on the same permutation stream.
Classification: dependence change at BH q <= 0.10 over the pairs entering stage two, marginal change at
min(p_X, p_Y) <= 0.05.

Run:    python 00_SRC/run_us_finance_stage2.py --procs 8
Writes 04_DAOU/EXPERIMENT/us_panel/{stage2_jobs.jsonl (per test, resumable), stage2.json, records_stage2_perm.csv.gz}
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from math import ceil  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
from dots.hourly import features  # noqa: E402
import run_e6_retest as C  # noqa: E402
import run_us_finance_panel as U  # noqa: E402
from run_e8_summary import step_up_q  # noqa: E402

OUT = U.OUT
K, N_DRAWS, UNIT = 999, 20, 1
ORDER = ["SPX-GLD", "SPX-TNX", "SPX-DXY"]


def _job(a):
    name, date, b, kind, j, seed = a
    dates, x, y = U.pair_data(name)
    c = dates.index(date)
    lo_i, hi_i = C.window(len(x), c, UNIT, b)
    xw, yw = x[lo_i:hi_i], y[lo_i:hi_i]
    n = len(xw)
    lo = max(int(ceil(max(b, 1) / UNIT)), int(ceil(0.06 * n)))
    rng = np.random.default_rng(seed)
    if kind == "dep":
        sx, sy = (2026, 2027) if j == 0 else (10000 + 2 * j, 10001 + 2 * j)
        FX, FY, J = features(xw, yw, sx=sx, sy=sy)
        O = C.outer_products(FX, FY, J)
        ts, obs = C.dep_curve(O, UNIT, lo)
        R = np.empty((K, len(ts)))
        for k in range(K):
            o = C.perm_index(n, b, rng)
            R[k] = C.dep_curve(tuple(M[o] for M in O), UNIT, lo)[1]
    else:
        v = xw if kind == "marg_x" else yw
        ts, obs = C.marg_curve(v, UNIT, lo)
        R = np.empty((K, len(ts)))
        for k in range(K):
            R[k] = C.marg_curve(v[C.perm_index(n, b, rng)], UNIT, lo)[1]
    p, T = C.perm_pvalue(obs, R)
    return dict(pair=name, date=date, block=b, kind=kind, draw=j, window=f"{dates[lo_i]}-{dates[hi_i - 1]}",
                n_window=n, n_blocks=n // b, min_seg=lo, p=float(p), T_obs=float(T[0]),
                T_rep=" ".join(f"{t:.6f}" for t in T[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    summ = json.load(open(os.path.join(OUT, "summary.json")))
    diag = json.load(open(os.path.join(OUT, "diag.json")))
    todo = [n for n in ORDER if summ[n]["survives_bh"]]
    jobs, dates_ = [], {}
    for n in todo:
        i = ORDER.index(n)
        stat = summ[n]["primary_statistic"]
        date = summ[n]["tau_date"][stat]
        dates_[n] = date
        b = diag[n]["block"]
        base = 710000 + 100 * i
        jobs += [(n, date, b, "dep", 0, base + 1), (n, date, b, "marg_x", 0, base + 2), (n, date, b, "marg_y", 0, base + 3)]
        jobs += [(n, date, b, "dep", j, base + 1) for j in range(1, N_DRAWS + 1)]
    jobs.sort(key=lambda t: t[4] > 0)                       # decision tests (draw 0) first, sensitivity draws after
    path = os.path.join(OUT, "stage2_jobs.jsonl")           # one line per finished test; a rerun resumes
    done = {(r["pair"], r["kind"], r["draw"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(_job, [j for j in jobs if (j[0], j[3], j[4]) not in done], chunksize=1):
            with open(path, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(r["pair"], r["kind"], r["draw"], round(r["p"], 4), flush=True)
    out = [json.loads(l) for l in open(path)]
    if len(out) < len(jobs):
        sys.exit(f"{len(out)} of {len(jobs)} tests finished")
    with gzip.open(os.path.join(OUT, "records_stage2_perm.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader(); w.writerows(out)
    get = lambda n, kind, j=0: next(r for r in out if r["pair"] == n and r["kind"] == kind and r["draw"] == j)  # noqa: E731
    pd = np.array([get(n, "dep")["p"] for n in todo])
    q = step_up_q(pd)
    res = {}
    for i, n in enumerate(todo):
        pm = min(get(n, "marg_x")["p"], get(n, "marg_y")["p"])
        dep, marg = q[i] <= 0.10, pm <= 0.05
        draws = [get(n, "dep", j)["p"] for j in range(1, N_DRAWS + 1)]
        g = get(n, "dep")
        res[n] = dict(date=dates_[n], window=g["window"], n_window=g["n_window"], block=g["block"],
                      n_blocks=g["n_blocks"], min_seg=g["min_seg"], p_dep=float(pd[i]), bh_q_dep=float(q[i]),
                      p_marg_x=get(n, "marg_x")["p"], p_marg_y=get(n, "marg_y")["p"], p_marg_min=pm,
                      cls=("coupling change" if dep and not marg else "change in both" if dep and marg
                           else "marginal-driven" if marg else "undetermined"),
                      feature_draws=dict(median=float(np.median(draws)), min=float(min(draws)), max=float(max(draws)),
                                         frac_le_05=float(np.mean([d <= 0.05 for d in draws]))))
        print(n, res[n], flush=True)
    json.dump(dict(config=dict(K=K, unit=UNIT, n_draws=N_DRAWS),
                   results=res), open(os.path.join(OUT, "stage2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
