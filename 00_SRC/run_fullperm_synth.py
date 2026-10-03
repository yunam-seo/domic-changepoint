#!/usr/bin/env python
"""Synthetic behavior of the exploratory whole-procedure resampling check (run_e8_fullperm.py;
Supplementary Section B.14).

Purpose
-------
run_e8_fullperm.py permutes the whole two-stage weather analysis, holding the stage-one penalty and
bias correction at their observed values; its finite-sample validity and its adjustment for
selection are not established (B.14). This script checks, on a scaled-down copy of the same machinery, (a) its size
under a null of constant dependence with serially dependent margins, (b) its size under a
MARGINAL-ONLY change (scenario M1: the scale of X changes, X and Y independent throughout), where
the permutation null (block exchangeability) is false although the dependence null is true, and (c)
its power under a dependence change of the S2 type (code D2: independent -> dependence without
correlation, y = sqrt(1-a^2) e1 + a |x| e2). Alongside the whole-procedure rate it records the
NOMINAL rate (smallest inner stage-two p <= 0.05) and how often stage one fires.

Scaled-down pipeline (same code path as the weather analysis, smaller grid)
--------------------------------------------------------------------------
n = 600, grid unit 15 observations (B = 40 intervals), D = 8 features on global ranks.
Stage one: optimal partitioning with the energy-weighted von Neumann cost, centered by 3 block-
permuted copies, penalty = smallest value on a geometric grid at which <= 5% of 20 block-permuted
copies (each centered by 3 copies of its own) return a break; super-blocks of 2 intervals (30 obs).
Stage two per candidate: window +/- 10 intervals, ranks re-computed in the window, weighted DOMI-
difference curve over split points 2..Bw-2, studentized by K_IN = 99 block-permutation replicas of
1-interval blocks, T = max of the studentized curve.
Outer test: B_OUT = 99 permutations of the 20 super-blocks, stratified by four pseudo-season labels
(k mod 4; the synthetic series have no season, so the strata only mirror the deployed design);
stage-one calibration held at the observed record's values, as in run_e8_fullperm.py. p = (1 + #{b: max_k T*_bk >= max_j T_j}) / (B_OUT + 1), 1 if stage one
returns nothing; reject at p <= 0.05.

Cells
-----
  null_ar     X, Y AR(1) phi = 0.6 margins, Gaussian dependence rho = 0.5 throughout (size)
  M1_s1.6     dots.synth M1 at s = 1.6, change at 300 (size under marginal drift; iid)
  D2_a0.7     dots.synth D2 at a = 0.7, change at 300 (power; iid; defined but not in the reported run)
  D2_a0.9     dots.synth D2 at a = 0.9 (power; iid)

Run
---
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python 00_SRC/run_fullperm_synth.py --reps 200 --procs 4
Outputs: 04_DAOU/EXPERIMENT/fullperm_synth/{records_reps.csv (one row per replicate, with every
         permuted max statistic), summary.json}
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
import run_e8_hourly as H  # noqa: E402
from dots.hourly import features, block_moments, prefix, domi_diff_curve, block_perm_order  # noqa: E402
from dots import pelt as P  # noqa: E402
from dots.synth import generate  # noqa: E402
from dots.perm import ge  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "fullperm_synth")
SEED = 20260926
N, UNIT = 600, 15
SUPER1, SUPER2 = 2, 1
HALF, MINW, LO = 10, 6, 2
K_IN, B_OUT = 99, 99
NPERM, ALPHA = 20, 0.05
BETAS = np.geomspace(0.05, 2000, 80)
NSTRATA = 4
CELLS = {"null_ar": ("null", None), "M1_s1.6": ("M1", 1.6), "D2_a0.7": ("D2", 0.7),
         "D2_a0.9": ("D2", 0.9)}


def ar1(rng, n, phi):
    e = rng.standard_normal(n + 200)
    x = np.zeros(n + 200)
    s = np.sqrt(1 - phi ** 2)
    for t in range(1, n + 200):
        x[t] = phi * x[t - 1] + s * e[t]
    return x[200:]


def data(cell, rep):
    kind, lev = CELLS[cell]
    if kind == "null":
        rng = np.random.default_rng([SEED, 0, rep])
        ex, ey = ar1(rng, N, 0.6), ar1(rng, N, 0.6)
        return ex, 0.5 * ex + np.sqrt(0.75) * ey
    li = {"M1": [1.3, 1.6, 2.0], "D2": [0.5, 0.7, 0.9]}[kind].index(lev)
    Z = generate(kind, lev, rep, n=N, tau=N // 2, level_idx=li)["Z"]
    return Z[:, 0], Z[:, 1]


def build(x, y):
    edges = np.arange(0, len(x) + 1, UNIT)
    if edges[-1] != len(x):
        edges = np.append(edges, len(x))
    FX, FY, J = features(x, y)
    return prefix(*block_moments(FX, FY, J, edges)), len(edges) - 1, FX, FY, J, edges


def stage_one(x, y, rng):
    """run_e8_hourly._job stage one, scaled: returns (cps, beta, Cp3, moments, B, edges, beta_edge)."""
    Pr, B, FX, FY, J, edges = build(x, y)
    C = H.cost_matrix(Pr, B)
    Cp = np.zeros_like(C)
    for _ in range(3):
        Cp += H.cost_matrix(H.permuted_prefix(FX, FY, J, edges, B, rng, super_=SUPER1), B)
    Cp3 = Cp / 3
    fa = {bb: 0 for bb in BETAS}
    for _ in range(NPERM):
        Cq = H.cost_matrix(H.permuted_prefix(FX, FY, J, edges, B, rng, super_=SUPER1), B)
        Cqp = np.zeros_like(Cq)
        for _ in range(3):
            Cqp += H.cost_matrix(H.permuted_prefix(FX, FY, J, edges, B, rng, super_=SUPER1), B)
        with np.errstate(invalid="ignore"):
            Cq = np.where(np.isfinite(Cq), Cq - Cqp / 3, np.inf)
        for bb in BETAS:
            fa[bb] += len(P.pelt_from_costs(Cq, bb)) > 0
    ok = [bb for bb in BETAS if fa[bb] / NPERM <= ALPHA]
    beta = ok[0] if ok else BETAS[-1]
    with np.errstate(invalid="ignore"):
        cps = P.pelt_from_costs(np.where(np.isfinite(C), C - Cp3, np.inf), beta)
    edge = bool(beta == BETAS[0] or not ok)
    return cps, float(beta), Cp3, block_moments(FX, FY, J, edges), B, edges, edge


def stage2_T(x, y, c, edges, B, rng):
    lo, hi = max(0, c - HALF), min(B, c + HALF)
    if hi - lo < MINW:
        return None
    sl = slice(edges[lo], edges[hi])
    Pw, Bw, FX, FY, J, ew = build(x[sl], y[sl])
    ts, obs = domi_diff_curve(Pw, Bw, lo=LO)
    mx, my, mj, cnt = block_moments(FX, FY, J, ew)
    R = np.empty((K_IN, len(ts)))
    for k in range(K_IN):
        o = block_perm_order(Bw, SUPER2, rng)
        R[k] = domi_diff_curve(prefix(mx[o], my[o], mj[o], cnt[o]), Bw, lo=LO)[1]
    A = np.vstack([obs[None, :], R])
    mu, sd = A.mean(0), A.std(0) + 1e-12
    T = [float(np.nanmax((A[i] - mu) / sd)) for i in range(K_IN + 1)]
    return T[0], (1 + sum(ge(t, T[0]) for t in T[1:])) / (K_IN + 1)


def outer_order(nb, B, rng):
    labels = np.arange(nb) % NSTRATA
    order = np.arange(nb)
    for s in range(NSTRATA):
        g = np.flatnonzero(labels == s)
        order[g] = g[rng.permutation(len(g))]
    idx = np.concatenate([np.arange(k * SUPER1, (k + 1) * SUPER1) for k in order])
    return np.concatenate([idx, np.arange(nb * SUPER1, B)])


def _rep(args):
    cell, rep = args
    t0 = time.time()
    x, y = data(cell, rep)
    ci = list(CELLS).index(cell)
    rng = np.random.default_rng([SEED, 1, ci, rep])
    # the data: global ranks (continuous synthetic data have no ties)
    from dots.domi import ranks01
    x, y = ranks01(x)[:, 0], ranks01(y)[:, 0]
    cps, beta, Cp3, (mx, my, mj, cnt), B, edges, edge = stage_one(x, y, rng)
    obs = []
    for c in cps:
        r = stage2_T(x, y, int(c), edges, B, np.random.default_rng([SEED, 2, ci, rep, 0, int(c)]))
        if r is not None:
            obs.append((int(c), r[0], r[1]))
    Tobs = max((t for _, t, _ in obs), default=-np.inf)
    nb = B // SUPER1
    Tperm = []
    for b in range(1, B_OUT + 1):
        o = outer_order(nb, B, np.random.default_rng([SEED, 3, ci, rep, b]))
        Cb = H.cost_matrix(prefix(mx[o], my[o], mj[o], cnt[o]), B)
        with np.errstate(invalid="ignore"):
            cb = P.pelt_from_costs(np.where(np.isfinite(Cb), Cb - Cp3, np.inf), beta)
        hidx = np.concatenate([np.arange(edges[k], edges[k + 1]) for k in o])
        px, py = x[hidx], y[hidx]
        ts = []
        for c in cb:
            r = stage2_T(px, py, int(c), edges, B, np.random.default_rng([SEED, 2, ci, rep, b, int(c)]))
            if r is not None:
                ts.append(r[0])
        Tperm.append(max(ts, default=-np.inf))
    Tperm = np.array(Tperm)
    p = 1.0 if not obs else float((1 + np.sum(ge(Tperm, Tobs))) / (B_OUT + 1))
    return dict(cell=cell, rep=rep, beta=round(beta, 4), beta_at_grid_edge=int(edge),
                n_cand=len(obs), cands=";".join(f"{c}:{t:.6f}:{q:.4f}" for c, t, q in obs),
                T_obs=Tobs, p_nominal_min=min((q for _, _, q in obs), default=1.0), p_fullperm=p,
                perm_fire=int(np.isfinite(Tperm).sum()),
                T_perm=" ".join("-inf" if not np.isfinite(t) else f"{t:.6f}" for t in Tperm),
                secs=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--procs", type=int, default=4)
    # default = the three cells reported in Supplementary Section B.14; D2_a0.7 stays in CELLS so that every
    # cell keeps its seed (seeds use the position in CELLS), but it is not part of the reported run
    ap.add_argument("--cells", type=str, default="null_ar,M1_s1.6,D2_a0.9")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cells = a.cells.split(",")
    path = os.path.join(OUT, "records_reps.csv")
    done = set()
    if os.path.exists(path):
        done = {(r["cell"], int(r["rep"])) for r in csv.DictReader(open(path))}
    jobs = [(c, r) for r in range(a.reps) for c in cells if (c, r) not in done]
    print(f"{len(jobs)} replicate jobs ({len(done)} on disk)", flush=True)
    fields = ["cell", "rep", "beta", "beta_at_grid_edge", "n_cand", "cands", "T_obs",
              "p_nominal_min", "p_fullperm", "perm_fire", "T_perm", "secs"]
    new = not os.path.exists(path)
    t0 = time.time()
    with open(path, "a", newline="") as f, Pool(a.procs, maxtasksperchild=10) as pool:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for i, r in enumerate(pool.imap_unordered(_rep, jobs)):
            w.writerow(r); f.flush()
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    rows = list(csv.DictReader(open(path)))
    summ = dict(config=dict(n=N, unit=UNIT, B=N // UNIT, super1=SUPER1, super2=SUPER2, half=HALF,
                            K_in=K_IN, B_out=B_OUT, strata=NSTRATA, alpha=ALPHA), cells={})
    print(f"\n  {'cell':<10}{'reps':>5}{'fullperm':>10}{'(se)':>7}{'nominal':>9}{'(se)':>7}"
          f"{'stage1 fires':>13}{'perm fire':>10}")
    for c in cells:
        rr = [r for r in rows if r["cell"] == c]
        m = len(rr)
        if not m:
            continue
        rf = np.mean([float(r["p_fullperm"]) <= ALPHA for r in rr])
        rn = np.mean([float(r["p_nominal_min"]) <= ALPHA for r in rr])
        fire = np.mean([int(r["n_cand"]) > 0 for r in rr])
        pf = np.mean([int(r["perm_fire"]) / B_OUT for r in rr])
        se = lambda q: float(np.sqrt(q * (1 - q) / m))
        # descriptive selection-conditional reading (run_e8_fullperm.p_cond_pooled_descriptive):
        # observed max T against the T of candidates selected on the permuted records of the
        # OTHER replicates of the cell; no candidate -> no rejection
        tp = [np.array([float(v) for v in r["T_perm"].split()]) for r in rr]
        fired = [t[np.isfinite(t)] for t in tp]
        allf = np.concatenate(fired)
        rc = []
        for r, fo in zip(rr, fired):
            if int(r["n_cand"]) == 0:
                rc.append(False); continue
            T = float(r["T_obs"])
            pool_n = len(allf) - len(fo)
            k = np.sum(allf >= T) - np.sum(fo >= T)
            rc.append((1 + k) / (1 + pool_n) <= ALPHA)
        rcd = float(np.mean(rc))
        summ["cells"][c] = dict(reps=m, reject_fullperm=float(rf), se_fullperm=se(rf),
                                reject_nominal=float(rn), se_nominal=se(rn),
                                stage1_fire_rate_obs=float(fire), stage1_fire_rate_perm=float(pf),
                                n_beta_at_grid_edge=int(sum(int(r["beta_at_grid_edge"]) for r in rr)),
                                reject_cond_pooled_descriptive=rcd, se_cond=se(rcd))
        print(f"  {c:<10}{m:>5}{rf:>10.3f}{se(rf):>7.3f}{rn:>9.3f}{se(rn):>7.3f}{fire:>13.3f}{pf:>10.3f}"
              f"   cond(pooled) {rcd:.3f} ({se(rcd):.3f})")
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
