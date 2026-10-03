#!/usr/bin/env python
"""E8 - the hourly weather analysis in two stages (Section 6.7).

Hourly anomalies at twelve ASOS stations, three variable pairs per station (36 analyses).
  1. SEGMENT with Holevo partitioning on permutation-bias-corrected costs over a two-week
     candidate grid, the penalty calibrated so that at most 5% of copies permuted in 12-week
     super-blocks produce any break. This step carries the multi-break power (Section 6.3) but
     responds to the joint state, marginal structure included.
  2. RE-TEST each candidate break with the DOMI-difference statistic on a +/- 1 year window under
     BLOCK permutation of six-week blocks: the unrestricted scheme of Supplementary Section B.8.
     This step is specific to dependence (Section 6.2), and the block form allows for serial
     dependence. Here stage two runs at K = 99 as a screen; the reported unrestricted stage two
     (K = 999) is run_e8_retest.py, and the season-restricted one of Section 6.7 is run_e8_season.py.

A break that passes stage 2 is evidence of a dependence change; one that does not may still be a
regime change, but of the joint state rather than of the coupling.

Writes 04_DAOU/EXPERIMENT/e8/{results.json, records_stage1_beta_perm.csv, records_stage2_perm.csv}
"""
from __future__ import annotations
import json, os, sys, time
from multiprocessing import Pool
import zlib

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.hourly import (features, block_moments, prefix, seg_cost,  # noqa: E402
                         domi_diff_curve, block_perm_order)
from dots import pelt as P  # noqa: E402
from dots.perm import ge  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8")
ANOM = os.path.join(ROOT, "02_MART", "WEATHER_HOURLY_ANOM.npz")
UNIT = 336            # candidate grid: two weeks of hourly observations
SUPER = 6             # block permutation unit: 6 grid intervals = 12 weeks
K = 99
ALPHA = 0.05
NAMES = {"105": "Gangneung", "108": "Seoul", "112": "Incheon", "119": "Suwon", "129": "Seosan",
         "133": "Daejeon", "143": "Daegu", "146": "Jeonju", "156": "Gwangju", "159": "Busan",
         "165": "Mokpo", "184": "Jeju"}
PAIRS = [("ta", "hm"), ("ta", "ws"), ("hm", "ws")]


def cost_matrix(P_, B):
    C = np.full((B + 1, B + 1), np.inf)
    for i in range(B):
        for j in range(i + 1, B + 1):
            C[i, j] = seg_cost(P_, i, j)
    return C


def build(x, y):
    edges = np.arange(0, len(x) + 1, UNIT)
    if edges[-1] != len(x):
        edges = np.append(edges, len(x))
    B = len(edges) - 1
    FX, FY, J = features(x, y)
    return prefix(*block_moments(FX, FY, J, edges)), B, FX, FY, J, edges


def permuted_prefix(FX, FY, J, edges, B, rng, super_=SUPER):
    """Block-permuted copy at the level of grid intervals (preserves within-block dependence)."""
    order = block_perm_order(B, super_, rng)
    mx, my, mj, cnt = block_moments(FX, FY, J, edges)
    return prefix(mx[order], my[order], mj[order], cnt[order])


def _job(args):
    stn, (u, v) = args
    z = np.load(ANOM)
    a, b = z[f"{stn}|{u}"], z[f"{stn}|{v}"]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    tm = z[f"{stn}|tm"][ok]
    Pr, B, FX, FY, J, edges = build(a, b)
    rng = np.random.default_rng(20260914 + zlib.crc32((stn + u + v).encode()) % 9999)

    # ---- stage 1: Holevo partitioning with permutation-bias-corrected costs
    C = cost_matrix(Pr, B)
    Cp = np.zeros_like(C)
    for _ in range(3):
        Cp += cost_matrix(permuted_prefix(FX, FY, J, edges, B, rng), B)
    fin = np.isfinite(C)
    Cbc = np.where(fin, C - Cp / 3, np.inf)
    betas = list(np.geomspace(5, 40000, 80))   # geometric and wide: costs scale with n,
    # and a truncated grid silently returns the ceiling instead of a calibrated penalty
    fa = {bb: 0 for bb in betas}
    beta_stars = []      # per permuted copy: smallest beta returning no break (the number of
                         # breaks is non-increasing in the penalty, so this scalar regenerates fa[bb] for every bb)
    NPERM = 20
    for _ in range(NPERM):
        Cq = cost_matrix(permuted_prefix(FX, FY, J, edges, B, rng), B)
        Cqp = np.zeros_like(Cq)
        for _ in range(3):
            Cqp += cost_matrix(permuted_prefix(FX, FY, J, edges, B, rng), B)
        Cq = np.where(np.isfinite(Cq), Cq - Cqp / 3, np.inf)
        ks = {bb: len(P.pelt_from_costs(Cq, bb)) for bb in betas}
        for bb in betas:
            fa[bb] += ks[bb] > 0
        beta_stars.append(next((bb for bb in betas if ks[bb] == 0), float("inf")))
    beta = next((bb for bb in betas if fa[bb] / NPERM <= ALPHA), betas[-1])
    beta_fa = fa[beta] / NPERM
    cps = P.pelt_from_costs(Cbc, beta)

    # ---- stage 2: re-test each break with the marginal-invariant statistic, block permutation
    passed = []
    for c in cps:
        lo, hi = max(0, c - 26), min(B, c + 26)          # +/- one year of two-week intervals
        if hi - lo < 12:
            continue
        sl = slice(edges[lo], edges[hi])
        aa, bb2 = a[sl], b[sl]
        Pw, Bw, FXw, FYw, Jw, ew = build(aa, bb2)
        ts, obs = domi_diff_curve(Pw, Bw, lo=3)
        R = np.empty((K, len(ts)))
        r2 = np.random.default_rng(777 + c)
        for k in range(K):
            Pp = permuted_prefix(FXw, FYw, Jw, ew, Bw, r2, super_=3)
            R[k] = domi_diff_curve(Pp, Bw, lo=3)[1]
        A = np.vstack([obs[None, :], R])
        mu, sd = A.mean(0), A.std(0) + 1e-12
        T = [float(np.nanmax((A[i] - mu) / sd)) for i in range(K + 1)]
        p = (1 + sum(ge(t, T[0]) for t in T[1:])) / (K + 1)
        passed.append(dict(interval=int(c), date=str(tm[min(edges[c], len(tm) - 1)])[:8],
                              p_block=round(p, 3), passed=bool(p <= ALPHA),
                              _T=" ".join(f"{v:.6f}" for v in T)))
    return dict(stn=stn, station=NAMES[stn], pair=f"{u}-{v}", n=int(len(a)), n_intervals=int(B),
                beta=round(float(beta), 1), beta_fa=round(float(beta_fa), 3), n_breaks=len(cps),
                breaks=passed, _beta_stars=[float(b) for b in beta_stars])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--stations", type=str, default="")
    a_ = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    z = np.load(ANOM)
    stns = sorted({k.split("|")[0] for k in z.files})
    if a_.stations:
        stns = [s for s in stns if s in a_.stations.split(",")]
    jobs = [(s, pr) for s in stns for pr in PAIRS]
    print(f"{len(jobs)} station-pair jobs", flush=True)
    t0 = time.time()
    with Pool(a_.procs, maxtasksperchild=2) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    for r in res:
        passed_breaks = [b for b in r["breaks"] if b["passed"]]
        print(f"  {r['station']:<10} {r['pair']:<6} n={r['n']:>6} beta={r['beta']:>6} "
              f"(fa={r['beta_fa']:.2f}) breaks={r['n_breaks']:>2} passed={len(passed_breaks):>2} "
              + (", ".join(f"{b['date']}(p={b['p_block']})" for b in passed_breaks) if passed_breaks else ""), flush=True)
    import csv as _csv
    with open(os.path.join(OUT, "records_stage1_beta_perm.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["stn", "pair", "copy", "beta_star"])
        w.writeheader()
        w.writerows([dict(stn=r["stn"], pair=r["pair"], copy=i, beta_star=b)
                     for r in res for i, b in enumerate(r.pop("_beta_stars", []))])
    with open(os.path.join(OUT, "records_stage2_perm.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["stn", "pair", "interval", "date", "T"])
        w.writeheader()
        w.writerows([dict(stn=r["stn"], pair=r["pair"], interval=b["interval"], date=b["date"],
                          T=b.pop("_T", "")) for r in res for b in r["breaks"]])
    json.dump(dict(config=dict(unit_hours=UNIT, super_blocks=SUPER, K=K, alpha=ALPHA), results=res),
              open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(f"E8 done in {time.time()-t0:.0f}s", flush=True)
