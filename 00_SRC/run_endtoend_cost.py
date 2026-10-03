#!/usr/bin/env python
"""End-to-end cost of the procedure: the complete single-break test and the stage-one segmentation.

Purpose
-------
Table 2 times one difference curve. The procedure of Section 4.6 needs more: the single-break test
(Algorithm 1) evaluates the curve on the observed series and on K pair-permuted replicas, and the
two-stage procedure (Algorithm 2) first builds and segments a cost matrix. This script measures both
on one core.

(i)  Algorithm 1 as specified: random-feature DOMI, D = 8, global weighted difference over the grid
     k = 60..n-60, K = 99 pair permutations (rows of (X, Y) permuted jointly, ranks and features
     recomputed on each replica), per-candidate studentization with moments over all K + 1 curves,
     p = (1 + #{T_k >= T_0}) / (K + 1). Series: scenario S2 at a = 0.7 (the Table 2 design,
     run_scaling.sample), n = 600, 4800, 19200. Each n runs in a fresh process; wall-clock and peak
     resident memory are recorded.
(ii) Stage one of Algorithm 2 exactly as run_e8_hourly.py runs it on the hourly weather series
     (functions imported from that script): per-interval moments on a two-week grid, the cost
     matrix C(s) = E_s S(rho_XY,s) over all interval pairs, three block-permuted copies for the bias
     correction, the penalty calibration on 20 permuted copies (each with its own three-copy
     correction) over 80 penalties, and the final partition. One station-pair of the application
     (station 108, temperature-humidity) if 02_MART/WEATHER_HOURLY_ANOM.npz is present, otherwise a
     synthetic series of the same length.
(iii) Split-gain check on the same series: for every triple of grid points i < t < j, the gain
     C(i,j) - C(i,t) - C(t,j) of the uncorrected cost (Holevo information times E, never negative by
     concavity of the von Neumann entropy) and of the permutation-bias-corrected cost (centered at zero,
     so negative gains occur). PELT pruning with pruning constant 0 is exact only when every gain
     is >= 0.

Run:   python 00_SRC/run_endtoend_cost.py            (all parts; single core throughout)
       python 00_SRC/run_endtoend_cost.py alg1 4800  (one part, used internally per process)
Outputs: 04_DAOU/EXPERIMENT/endtoend_cost/{alg1.csv, stage1.json, split_gains.json}
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.perm import ge  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "endtoend_cost")
W, D, K = 60, 8, 99


def peak_mib():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


# ------------------------------------------------------------------ (i) Algorithm 1
def domi_curve(X, Y):
    from dots.domi import DOMIContext
    ctx = DOMIContext(X, Y, W, D=D, seed=2026, n_perm=0)
    n = ctx.n
    g = ctx.grid
    out = np.array([np.sqrt(t * (n - t) / n) * abs(ctx.domi(0, t) - ctx.domi(t, n)) for t in g])
    return g, out


def alg1(n):
    import run_scaling as S
    X, Y = S.sample(n)
    t0 = time.perf_counter()
    grid, obs = domi_curve(X, Y)
    t_obs = time.perf_counter() - t0
    rng = np.random.default_rng([20260925, n])
    A = np.empty((K + 1, len(grid)))
    A[0] = obs
    for k in range(K):
        idx = rng.permutation(n)
        A[k + 1] = domi_curve(X[idx], Y[idx])[1]
    mu, sd = A.mean(0), A.std(0) + 1e-12
    T = ((A - mu) / sd).max(1)
    p = (1 + int(np.sum(ge(T[1:], T[0])))) / (K + 1)
    tau_hat = int(grid[int(np.argmax((obs - mu) / sd))])
    total = time.perf_counter() - t0
    print(json.dumps(dict(n=n, D=D, K=K, grid=len(grid), observed_curve_sec=round(t_obs, 2),
                          total_sec=round(total, 1), peak_rss_MiB=peak_mib(), p_value=p, tau_hat=tau_hat,
                          tau=n // 2)))


# ------------------------------------------------------------------ (ii) stage one
def load_series():
    anom = os.path.join(ROOT, "02_MART", "WEATHER_HOURLY_ANOM.npz")
    if os.path.exists(anom):
        z = np.load(anom)
        a, b = z["108|ta"], z["108|hm"]
        ok = np.isfinite(a) & np.isfinite(b)
        return a[ok], b[ok], "station 108, ta-hm (02_MART/WEATHER_HOURLY_ANOM.npz)"
    rng = np.random.default_rng(20260925)
    n = 70000
    return rng.standard_normal(n), rng.standard_normal(n), "synthetic, n = 70000"


def stage1():
    import run_e8_hourly as E
    from dots import pelt as P
    a, b, label = load_series()
    rng = np.random.default_rng(20260914 + zlib.crc32(("108" + "ta" + "hm").encode()) % 9999)
    tm = {}
    t0 = time.perf_counter()
    Pr, B, FX, FY, J, edges = E.build(a, b)
    tm["features_and_interval_moments"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    C = E.cost_matrix(Pr, B)
    tm["one_cost_matrix"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    Cp = np.zeros_like(C)
    perm_C = []
    for _ in range(3):
        Cq = E.cost_matrix(E.permuted_prefix(FX, FY, J, edges, B, rng), B)
        perm_C.append(Cq)
        Cp += Cq
    fin = np.isfinite(C)
    Cbc = np.where(fin, C - Cp / 3, np.inf)
    tm["bias_correction_3_copies"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    betas = list(np.geomspace(5, 40000, 80))
    fa = {bb: 0 for bb in betas}
    NPERM = 20
    t_dp = 0.0
    for _ in range(NPERM):
        Cq = E.cost_matrix(E.permuted_prefix(FX, FY, J, edges, B, rng), B)
        Cqp = np.zeros_like(Cq)
        for _ in range(3):
            Cqp += E.cost_matrix(E.permuted_prefix(FX, FY, J, edges, B, rng), B)
        Cq = np.where(np.isfinite(Cq), Cq - Cqp / 3, np.inf)
        td = time.perf_counter()
        ks = {bb: len(P.pelt_from_costs(Cq, bb)) for bb in betas}
        t_dp += time.perf_counter() - td
        for bb in betas:
            fa[bb] += ks[bb] > 0
    beta = next((bb for bb in betas if fa[bb] / NPERM <= 0.05), betas[-1])
    tm["penalty_calibration_20_copies"] = time.perf_counter() - t1
    tm["of_which_partitioning_1600_runs"] = t_dp
    t1 = time.perf_counter()
    cps = P.pelt_from_costs(Cbc, beta)
    tm["final_partition"] = time.perf_counter() - t1
    tm["stage_one_total"] = time.perf_counter() - t0
    res = dict(series=label, n=int(len(a)), grid_unit_hours=E.UNIT, n_intervals=int(B),
               cost_matrices_built=1 + 3 + NPERM * 4, beta=float(beta), n_breaks=len(cps),
               seconds={k: round(v, 2) for k, v in tm.items()}, peak_rss_MiB=peak_mib())
    gains = split_gains(C, Cbc, perm_C)
    return res, gains


def split_gains(C, Cbc, perm_C):
    """Gains C(i,j) - C(i,t) - C(t,j) over all grid triples i < t < j."""
    m = C.shape[0]
    out = {}
    for name, M in (("uncorrected", C), ("bias_corrected", Cbc), ("one_permuted_copy_uncorrected", perm_C[0])):
        allg = []
        for i in range(m):
            for j in range(i + 2, m):
                t = np.arange(i + 1, j)
                g = M[i, j] - M[i, t] - M[t, j]
                g = g[np.isfinite(g)]
                if g.size:
                    allg.append(g)
        g = np.concatenate(allg)
        scale = np.nanmax(np.abs(M[np.isfinite(M)]))
        out[name] = dict(triples=int(g.size), min_gain=float(g.min()), max_gain=float(g.max()),
                         fraction_negative=float(np.mean(g < -1e-9 * scale)),
                         min_gain_relative_to_max_cost=float(g.min() / scale))
    return out


# ------------------------------------------------------------------ driver
def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for n in (600, 4800, 19200):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "alg1", str(n)],
                           capture_output=True, text=True, env=dict(os.environ))
        r = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else \
            dict(n=n, error=p.stderr.strip().splitlines()[-1][:300])
        rows.append(r)
        print(r, flush=True)
        with open(os.path.join(OUT, "alg1.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for x in rows for k in x}), restval="")
            w.writeheader()
            w.writerows(rows)
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "stage1"],
                       capture_output=True, text=True, env=dict(os.environ))
    print(p.stdout[-3000:], p.stderr[-2000:], flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "alg1":
        alg1(int(sys.argv[2]))
    elif cmd == "stage1":
        os.makedirs(OUT, exist_ok=True)
        res, gains = stage1()
        json.dump(res, open(os.path.join(OUT, "stage1.json"), "w"), indent=1)
        json.dump(gains, open(os.path.join(OUT, "split_gains.json"), "w"), indent=1)
        print(json.dumps(res), json.dumps(gains))
    else:
        main()
