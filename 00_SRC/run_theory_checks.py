#!/usr/bin/env python
"""Supplementary Section B.3, Figure B.2 (middle and right panels) and Section A.7.

T2  Pair-permutation p-value under the null (P1): 300 null reps, K=49 -> empirical P(p <= alpha)
    at alpha = 0.02, 0.05, 0.10, 0.20.
T3  Relative localization error |tau_hat - tau|/n of a permutation-studentized estimator (120 reps,
    25 pair-permutation replicas), n in {300,600,1200,2400}; Figure B.2, right panel. Illustrative
    only; it does not verify P3, which concerns the unstudentized criterion.

Writes 04_DAOU/EXPERIMENT/theory/{results.json, records_theory_reps.csv}
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from dots.synth import generate  # noqa: E402
from dots.domi import DOMIContext, domi_stats  # noqa: E402
from dots.perm import ge  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "theory")
D = 8


def _t2_job(rep):
    smp = generate("D2", 0.7, rep, n=600, tau=300, null=True, base_seed=20260828, level_idx=1)
    Z = smp["Z"]
    X, Y = Z[:, [0]], Z[:, [1]]
    K = 49

    def curve(Xa, Ya):
        c = DOMIContext(Xa, Ya, 60, D=D, seed=2026, n_perm=0)
        return domi_stats(c, "global")["DOMI-diff"]

    obs = curve(X, Y)
    rng = np.random.default_rng([20260828, rep, 3])
    R = []
    for k in range(K):
        idx = rng.permutation(len(X))
        R.append(curve(X[idx], Y[idx]))
    R = np.array(R)
    # symmetric studentization: moments from ALL K+1 curves (observed included) -> exact validity (P1)
    A = np.vstack([obs[None, :], R])
    mu, sd = A.mean(0), A.std(0) + 1e-12
    T = [np.max((A[i] - mu) / sd) for i in range(K + 1)]
    p = (1 + sum(ge(t, T[0]) for t in T[1:])) / (K + 1)
    return float(p)


def _t3_job(a):
    n, rep = a
    smp = generate("D2", 0.9, rep, n=n, tau=n // 2, null=False, base_seed=20260829, level_idx=2)
    Z = smp["Z"]
    ctx = DOMIContext(Z[:, [0]], Z[:, [1]], max(40, n // 10), D=D, seed=2026, n_perm=0)
    g = domi_stats(ctx, "global")["DOMI-diff"]
    # studentize against 25 pair-permutation replicas
    rng = np.random.default_rng([20260829, rep, 5])
    R = []
    for k in range(25):
        idx = rng.permutation(n)
        cp = DOMIContext(Z[idx][:, [0]], Z[idx][:, [1]], max(40, n // 10), D=D, seed=2026, n_perm=0)
        R.append(domi_stats(cp, "global")["DOMI-diff"])
    R = np.array(R)
    T = (g - R.mean(0)) / (R.std(0) + 1e-12)
    tau_hat = int(ctx.grid[int(np.nanargmax(T))])
    return abs(tau_hat - n // 2) / n


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    REC = []
    with Pool(24, maxtasksperchild=40) as pool:
        # T2
        t0 = time.time()
        ps = np.array(pool.map(_t2_job, range(300), chunksize=2))
        REC += [dict(part="T2", config="pvalue", rep=i, value=float(p)) for i, p in enumerate(ps)]
        # p-values are discrete (multiples of 1/(K+1)) -> check level validity P(p<=alpha)<=alpha on a grid
        levels = {a: float(np.mean(ps <= a)) for a in (0.02, 0.05, 0.10, 0.20)}
        res["T2"] = dict(n=300, K=49, level_check=levels)
        print(f"T2 ({time.time()-t0:.0f}s): P(p<=α) at α=.02/.05/.10/.20 -> " +
              "/".join(f"{levels[a]:.3f}" for a in (0.02, 0.05, 0.10, 0.20)), flush=True)
        # T3
        t3 = {}
        for n in (300, 600, 1200, 2400):
            errs = pool.map(_t3_job, [(n, r) for r in range(120)], chunksize=2)
            REC += [dict(part="T3", config=f"n{n}", rep=i, value=float(e)) for i, e in enumerate(errs)]
            t3[str(n)] = dict(median_rel_err=float(np.median(errs)), q90=float(np.quantile(errs, 0.9)))
            print(f"T3 n={n}: median |τ̂-τ|/n = {t3[str(n)]['median_rel_err']:.4f}", flush=True)
        res["T3"] = t3
    import csv as _csv
    with open(os.path.join(OUT, "records_theory_reps.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["part", "config", "rep", "value"])
        w.writeheader(); w.writerows(REC)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)

    print("theory checks done ->", OUT, flush=True)

if __name__ == "__main__":
    main()
