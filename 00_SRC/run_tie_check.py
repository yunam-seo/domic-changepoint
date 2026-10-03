#!/usr/bin/env python
"""Tied observations and the permutation level.

The hourly weather records carry many tied values (averaged over consecutive two-year windows, 19-31% of temperature,
43-65% of wind-speed and 63-77% of humidity anomalies share their value with another observation).
dots.domi.ranks01 breaks ties by time order, and the weather stage two permutes features computed
once on the observed record. Under ties that rank vector is not exchangeable, so P1/P5 do not cover
that implementation. This script measures the level of the pair-permutation test on no-change data
discretized to a given tie share, under three implementations:

  order      ties broken by time order, features computed once and permuted (weather stage two)
  random     ties broken at random (independent uniform jitter), features computed once and permuted
  recompute  ties broken by time order, ranks and features recomputed on every permuted record

'random' and 'recompute' are covered by P1 (the jitter is data-independent, so P1 applies to the
record augmented by it); 'order' is not.

Null: constant Gaussian-copula dependence rho = 0.5, n = 600, no change point. Pair permutation,
K = 99, symmetric studentization over all K+1 curves; 400 replicates per tie share.
Run:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_tie_check.py --procs 16
Writes 04_DAOU/EXPERIMENT/tie_check/{results.json, records.csv.gz}
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
from dots.domi import ranks01, unit_rff  # noqa: E402
from dots.persist import save_records  # noqa: E402
from dots.perm import ge  # noqa: E402
from run_perm_null_check import ent  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "tie_check")
CFG = dict(n=600, w=60, D=8, K=99, reps=400, rho=0.5, alpha=0.05, base_seed=20260925)
STEPS = {"none": 0.0, "low": 0.002, "mid": 0.005, "high": 0.008, "extreme": 0.02}  # rounding step; tie shares about 0, 28, 54, 70, 91%


def feats(x, y, D):
    FX = unit_rff(ranks01(x[:, None]), D, 2026)
    FY = unit_rff(ranks01(y[:, None]), D, 2027)
    return FX, FY, np.einsum("ti,tj->tij", FX, FY).reshape(len(x), D * D)


def curve(FX, FY, J, w):
    n, D = FX.shape
    cx = np.zeros((n + 1, D, D)); cy = np.zeros((n + 1, D, D)); cj = np.zeros((n + 1, D * D, D * D))
    np.cumsum(np.einsum("ti,tj->tij", FX, FX), axis=0, out=cx[1:])
    np.cumsum(np.einsum("ti,tj->tij", FY, FY), axis=0, out=cy[1:])
    np.cumsum(np.einsum("ti,tj->tij", J, J), axis=0, out=cj[1:])

    def I(a, b):
        m = b - a
        return ent((cx[b] - cx[a]) / m) + ent((cy[b] - cy[a]) / m) - ent((cj[b] - cj[a]) / m)
    return np.array([np.sqrt(t * (n - t) / n) * abs(I(0, t) - I(t, n)) for t in range(w, n - w + 1)])


def pval(obs, R):
    A = np.vstack([obs[None, :], R])
    mu, sd = A.mean(0), A.std(0) + 1e-12
    T = np.nanmax((A - mu) / sd, axis=1)
    return (1 + int(np.sum(ge(T[1:], T[0])))) / len(T)


def tie_share(v):
    _, c = np.unique(v, return_counts=True)
    return float(c[c > 1].sum() / len(v))


def _job(a):
    level, rep = a
    c = CFG
    rng = np.random.default_rng([c["base_seed"], rep, list(STEPS).index(level)])
    x = rng.standard_normal(c["n"])
    y = c["rho"] * x + np.sqrt(1 - c["rho"] ** 2) * rng.standard_normal(c["n"])
    h = STEPS[level]
    if h > 0:
        x, y = np.round(x / h) * h, np.round(y / h) * h
    jit = rng.uniform(size=(c["n"], 2)) * 1e-9 * (h if h > 0 else 1.0)
    perms = [rng.permutation(c["n"]) for _ in range(c["K"])]
    out = dict(level=level, rep=rep, tie_x=tie_share(x), tie_y=tie_share(y))
    for mode in ("order", "random", "recompute"):
        xx, yy = (x + jit[:, 0], y + jit[:, 1]) if mode == "random" else (x, y)
        FX, FY, J = feats(xx, yy, c["D"])
        obs = curve(FX, FY, J, c["w"])
        R = np.empty((c["K"], len(obs)))
        for k, p in enumerate(perms):
            if mode == "recompute":
                R[k] = curve(*feats(x[p], y[p], c["D"]), c["w"])
            else:
                R[k] = curve(FX[p], FY[p], J[p], c["w"])
        out[f"p_{mode}"] = pval(obs, R)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--reps", type=int, default=CFG["reps"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with Pool(a.procs, maxtasksperchild=20) as pool:
        recs = pool.map(_job, [(lv, r) for lv in STEPS for r in range(a.reps)], chunksize=2)
    p = os.path.join(OUT, "records.csv.gz")
    if os.path.exists(p):
        os.remove(p)
    save_records(OUT, "records.csv", recs, {})
    res = {}
    for lv in STEPS:
        g = [r for r in recs if r["level"] == lv]
        res[lv] = dict(step=STEPS[lv], reps=len(g),
                       tie_share_mean=float(np.mean([(r["tie_x"] + r["tie_y"]) / 2 for r in g])),
                       **{f"level_{m}": float(np.mean([r[f"p_{m}"] <= CFG["alpha"] for r in g]))
                          for m in ("order", "random", "recompute")})
        print(lv, res[lv], flush=True)
    json.dump(dict(config=CFG, steps=STEPS, results=res, wall_clock_s=round(time.time() - t0)),
              open(os.path.join(OUT, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
