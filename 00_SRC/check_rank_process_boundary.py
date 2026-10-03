"""Global versus within-segment ranks at the independence boundary ([NV-8], Supplementary Section B.3;
the rank corollary of Section A.12).

The deployed scan (dots/domi.py, DOMIContext) ranks each variable ONCE over the whole window
of n observations and reuses those pseudo-observations for every left segment [0, t) and
right segment [t, n).  Proposition P6-R is stated with ranks computed inside the sample
whose DOMI is taken.  The rank Corollary of A.12 states that, at the independence
boundary, both choices -- and the oracle probability integral transform -- share one
process-level limit, because the rank correction enters each segment moment as
A (x) rho_Y + rho_X (x) B with traceless A, B, a direction on which the boundary quadratic
form Phi vanishes together with its cross terms.

This script simulates the null (X, Y independent, continuous) and computes, on the same
replicates, the segment DOMI plug-ins with
    (a) global ranks, exactly as deployed (DOMIContext, D = 8, seed 2026, w = 60),
    (b) ranks recomputed within each segment, same feature map (same W, b, bandwidth),
    (c) oracle uniforms F(X_t), F(Y_t), same feature map,
and compares the laws of
    n * I_L(lambda), n * I_R(lambda)  at lambda in {0.25, 0.5},
    sup over the candidate grid t = w..n-w of  n * sqrt(lambda (1 - lambda)) * |I_L - I_R|
    (= sqrt(n) times the deployed unstudentized 'DOMI-diff' curve),
by two-sample Kolmogorov-Smirnov distances, quantiles and paired differences.
Part 2 repeats the pointwise comparison at n in {300, 600, 1200, 2400} to show that the
paired differences (a)-(c) and (b)-(c) shrink with n, as the argument predicts.

Run:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/check_rank_process_boundary.py [--reps 2000] [--procs 4]
Writes 04_DAOU/EXPERIMENT/theory_rank_process/ :
    records_part1.csv.gz         one row per replicate and method: bandwidths, n*I_L and n*I_R at
                                 lambda = 0.25 and 0.5, the scan supremum and its location
    records_part2.csv.gz         one row per sample size, replicate, lambda and method: n*I_L
    boundary_ranks_summary.json  KS distances, quantiles, paired differences
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dots.domi import DOMIContext, ranks01  # noqa: E402
from dots.extras import rff  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "theory_rank_process")
D, SEED, W_MIN, N = 8, 2026, 60, 600
LAMS = (0.25, 0.5)
BASE = 20260925


def deployed_gamma(U: np.ndarray, seed: int) -> float:
    """The median-heuristic bandwidth exactly as dots.extras.rff computes it when gamma=None."""
    Z = U[:, None] if U.ndim == 1 else U
    n = Z.shape[0]
    idx = np.random.default_rng(seed + 1).choice(n, size=min(n, 400), replace=False)
    S = Z[idx]
    sq = (S * S).sum(1)
    D2 = np.clip(sq[:, None] + sq[None, :] - 2 * S @ S.T, 0, None)
    med = np.median(D2[np.triu_indices(len(idx), 1)])
    return 1.0 / (med if med > 0 else 1.0)


def feat(u: np.ndarray, seed: int, gamma: float) -> np.ndarray:
    F = rff(u[:, None], D=D, seed=seed, gamma=gamma)
    return F / np.linalg.norm(F, axis=1, keepdims=True)


def ent_batch(M: np.ndarray) -> np.ndarray:
    lam = np.clip(np.linalg.eigvalsh(M), 0.0, None)
    lam = lam / lam.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(lam > 0, -lam * np.log(lam), 0.0)
    return t.sum(-1)


def moment(FX: np.ndarray, FY: np.ndarray):
    m = FX.shape[0]
    J = (FX[:, :, None] * FY[:, None, :]).reshape(m, D * D)
    return FX.T @ FX / m, FY.T @ FY / m, J.T @ J / m


def dom_from_moments(MX, MY, MXY):
    return np.maximum(ent_batch(MX) + ent_batch(MY) - ent_batch(MXY), 0.0)


def prefix_curves(FX, FY, grid):
    """I_L(t), I_R(t) on [0,t), [t,n) from fixed per-row features, via prefix sums."""
    n = FX.shape[0]
    J = (FX[:, :, None] * FY[:, None, :]).reshape(n, D * D)
    SX = np.concatenate([np.zeros((1, D, D)), np.cumsum(np.einsum("ti,tj->tij", FX, FX), 0)])
    SY = np.concatenate([np.zeros((1, D, D)), np.cumsum(np.einsum("ti,tj->tij", FY, FY), 0)])
    SJ = np.concatenate([np.zeros((1, D * D, D * D)), np.cumsum(np.einsum("ti,tj->tij", J, J), 0)])
    t = grid
    L = dom_from_moments(SX[t] / t[:, None, None], SY[t] / t[:, None, None], SJ[t] / t[:, None, None])
    r = (n - t)[:, None, None]
    R = dom_from_moments((SX[n] - SX[t]) / r, (SY[n] - SY[t]) / r, (SJ[n] - SJ[t]) / r)
    return L, R


def within_curves(x, y, grid, gx, gy):
    n = len(x)
    Ls, Rs = [], []
    for t in grid:
        FXl, FYl = feat(ranks01(x[:t])[:, 0], SEED, gx), feat(ranks01(y[:t])[:, 0], SEED + 1, gy)
        FXr, FYr = feat(ranks01(x[t:])[:, 0], SEED, gx), feat(ranks01(y[t:])[:, 0], SEED + 1, gy)
        Ls.append(moment(FXl, FYl))
        Rs.append(moment(FXr, FYr))
    L = dom_from_moments(*[np.stack([s[k] for s in Ls]) for k in range(3)])
    R = dom_from_moments(*[np.stack([s[k] for s in Rs]) for k in range(3)])
    return L, R


def one_rep(r: int):
    rng = np.random.default_rng(BASE + r)
    U0, V0 = rng.uniform(size=N), rng.uniform(size=N)
    x, y = np.log(U0 / (1 - U0)), np.log(V0 / (1 - V0))    # continuous margins; oracle F known
    ctx = DOMIContext(x[:, None], y[:, None], W_MIN, D=D, seed=SEED, n_perm=0)
    grid = ctx.grid
    aL = np.array([ctx.domi(0, t) for t in grid])
    aR = np.array([ctx.domi(t, N) for t in grid])
    gx, gy = deployed_gamma(ctx.UX, SEED), deployed_gamma(ctx.UY, SEED + 1)
    bL, bR = within_curves(x, y, grid, gx, gy)
    cL, cR = prefix_curves(feat(U0, SEED, gx), feat(V0, SEED + 1, gy), grid)
    return np.stack([np.stack([aL, aR]), np.stack([bL, bR]), np.stack([cL, cR])]).astype(np.float64), gx, gy


def one_rep_part2(args):
    n, r = args
    rng = np.random.default_rng(BASE + 10_000_000 + 100_000 * n + r)
    U0, V0 = rng.uniform(size=n), rng.uniform(size=n)
    UX, UY = ranks01(U0)[:, 0], ranks01(V0)[:, 0]
    gx, gy = deployed_gamma(UX[:, None], SEED), deployed_gamma(UY[:, None], SEED + 1)
    out = np.empty((3, len(LAMS)))
    for j, lam in enumerate(LAMS):
        t = int(round(lam * n))
        a = dom_from_moments(*moment(feat(UX[:t], SEED, gx), feat(UY[:t], SEED + 1, gy)))
        b = dom_from_moments(*moment(feat(ranks01(U0[:t])[:, 0], SEED, gx), feat(ranks01(V0[:t])[:, 0], SEED + 1, gy)))
        c = dom_from_moments(*moment(feat(U0[:t], SEED, gx), feat(V0[:t], SEED + 1, gy)))
        out[:, j] = n * np.array([a, b, c]).ravel()
    return out


def ks2(a, b):
    a, b = np.sort(a), np.sort(b)
    z = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, z, "right") / len(a) - np.searchsorted(b, z, "right") / len(b))))


def summarise(v):
    q = [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), **{f"q{int(100*p)}": float(np.quantile(v, p)) for p in q}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--reps2", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with Pool(args.procs) as pool:
        res = pool.map(one_rep, range(args.reps), chunksize=10)
    curves = np.stack([r[0] for r in res])               # (reps, 3 methods, 2 sides, grid)
    gam = np.array([[r[1], r[2]] for r in res])
    grid = np.arange(W_MIN, N - W_MIN + 1)
    lam = grid / N
    scan = N * np.sqrt(lam * (1 - lam)) * np.abs(curves[:, :, 0] - curves[:, :, 1])   # (reps, 3, grid)
    sup = scan.max(-1)
    argmax = grid[scan.argmax(-1)]
    names = ["global_ranks_deployed", "within_segment_ranks", "oracle_uniforms"]
    summ = {"config": {"n": N, "D": D, "seed": SEED, "w": W_MIN, "grid": [int(grid[0]), int(grid[-1])],
                       "reps": args.reps, "base_seed": BASE,
                       "bandwidth": "median heuristic on the global pseudo-observations, as deployed; "
                                    "same W, b, gamma reused for (b) and (c)",
                       "gamma_x_range": [float(gam[:, 0].min()), float(gam[:, 0].max())]}}
    stats = {}
    for lamv in LAMS:
        k = int(np.where(grid == int(round(lamv * N)))[0][0])
        for side, s in (("L", 0), ("R", 1)):
            stats[f"nI_{side}_lam{lamv}"] = N * curves[:, :, s, k]
    stats["sup_scan"] = sup
    for key, v in stats.items():
        e = {nm: summarise(v[:, i]) for i, nm in enumerate(names)}
        e["KS_a_vs_b"] = ks2(v[:, 0], v[:, 1])
        e["KS_a_vs_c"] = ks2(v[:, 0], v[:, 2])
        e["KS_b_vs_c"] = ks2(v[:, 1], v[:, 2])
        e["median_abs_paired_diff_a_c"] = float(np.median(np.abs(v[:, 0] - v[:, 2])))
        e["median_abs_paired_diff_b_c"] = float(np.median(np.abs(v[:, 1] - v[:, 2])))
        e["corr_a_c"] = float(np.corrcoef(v[:, 0], v[:, 2])[0, 1])
        e["corr_b_c"] = float(np.corrcoef(v[:, 1], v[:, 2])[0, 1])
        summ[key] = e
    summ["argmax_agreement_a_c"] = float(np.mean(np.abs(argmax[:, 0] - argmax[:, 2]) <= 5))
    summ["argmax_agreement_a_b"] = float(np.mean(np.abs(argmax[:, 0] - argmax[:, 1]) <= 5))
    cols = [k for k in stats if k != "sup_scan"]
    with gzip.open(os.path.join(OUT, "records_part1.csv.gz"), "wt", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["rep", "method", "gamma_x", "gamma_y"] + cols + ["sup_scan", "argmax_t"])
        for r in range(args.reps):
            for i, nm in enumerate(names):
                w.writerow([r, nm, repr(float(gam[r, 0])), repr(float(gam[r, 1]))]
                           + [repr(float(stats[c][r, i])) for c in cols] + [repr(float(sup[r, i])), int(argmax[r, i])])

    # ---- part 2: paired differences shrink with n
    ns = [300, 600, 1200, 2400]
    jobs = [(n, r) for n in ns for r in range(args.reps2)]
    with Pool(args.procs) as pool:
        p2 = np.stack(pool.map(one_rep_part2, jobs, chunksize=20)).reshape(len(ns), args.reps2, 3, len(LAMS))
    with gzip.open(os.path.join(OUT, "records_part2.csv.gz"), "wt", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["n", "rep", "lambda", "method", "nI_L"])
        for i, n in enumerate(ns):
            for r in range(args.reps2):
                for j, lamv in enumerate(LAMS):
                    for k, nm in enumerate(names):
                        w.writerow([n, r, lamv, nm, repr(float(p2[i, r, k, j]))])
    part2 = {}
    for i, n in enumerate(ns):
        for j, lamv in enumerate(LAMS):
            v = p2[i, :, :, j]
            part2[f"n{n}_lam{lamv}"] = {
                "median_nI": [float(np.median(v[:, k])) for k in range(3)],
                "median_abs_diff_a_c": float(np.median(np.abs(v[:, 0] - v[:, 2]))),
                "median_abs_diff_b_c": float(np.median(np.abs(v[:, 1] - v[:, 2]))),
                "median_abs_diff_a_b": float(np.median(np.abs(v[:, 0] - v[:, 1]))),
                "KS_a_vs_c": ks2(v[:, 0], v[:, 2]), "KS_b_vs_c": ks2(v[:, 1], v[:, 2])}
    summ["part2_pointwise_vs_n"] = {"reps": args.reps2, "segment": "left segment [0, lambda n)", **part2}
    summ["elapsed_s"] = time.time() - t0
    with open(os.path.join(OUT, "boundary_ranks_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)

    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk.startswith("KS") or kk.startswith("median_abs")})
                      for k, v in summ.items() if k != "config"}, indent=1))


if __name__ == "__main__":
    main()
