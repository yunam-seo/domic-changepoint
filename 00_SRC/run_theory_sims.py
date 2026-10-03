#!/usr/bin/env python
"""Numerical support for the three statements the paper does not prove.

  NV-5  identifiability in D for the deployed feature map (A.8(iii)): population DOMI as a function of
         the feature dimension D, for dependent families and for independence.
  NV-6  studentized scan (open in Section 3.5 / Section 7): the null law of the deployed
         studentized maximum across n, and its stability.
  NV-7  Holevo-partitioning segment-number consistency (open in Section 7): P(K-hat = 3) as n grows.

Run:  python 00_SRC/run_theory_sims.py --part {nv5,nv6,nv7,all} [--procs 28] [--reps N]
Out:  04_DAOU/EXPERIMENT/theory_sims/{nv5,nv6,nv7}.json
Seed-deterministic; the same command reproduces every reported value.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import zlib

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from dots.encode import MomentCache  # noqa: E402
from dots import pelt as P  # noqa: E402
from dots.domi import DOMIContext, ranks01, unit_rff, domi_stats  # noqa: E402
from dots.synth import generate, SCENARIOS  # noqa: E402
from dots.perm import ge  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "theory_sims")
os.makedirs(OUT, exist_ok=True)
BASE_SEED = 20260829


def say(tag, msg):
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: {msg}", flush=True)


def _vn_entropy(M):
    lam = np.clip(np.linalg.eigvalsh(M), 0, None)
    s = lam.sum()
    if s <= 0:
        return 0.0
    lam = lam[lam > 0] / s
    return float(-(lam * np.log(lam)).sum())


def _joint_moment(fx, fy, chunk=2000):
    """(1/m) sum_t (phi_X phi_Y^T) vectorized outer, accumulated in chunks (never holds all of J)."""
    m, D = fx.shape
    Mj = np.zeros((D * D, D * D))
    for a in range(0, m, chunk):
        b = min(m, a + chunk)
        Jc = np.einsum("ti,tj->tij", fx[a:b], fy[a:b]).reshape(b - a, D * D)
        Mj += Jc.T @ Jc
    return Mj / m


def _domi_from_pair(x, y, D, seed=2026):
    """Plug-in DOMI of one sample, deployed construction (ranks -> unit RFF -> tensor state)."""
    fx = unit_rff(ranks01(x[:, None]), D, seed)
    fy = unit_rff(ranks01(y[:, None]), D, seed + 1)
    Mx = fx.T @ fx / len(fx)
    My = fy.T @ fy / len(fy)
    Mj = _joint_moment(fx, fy)
    return _vn_entropy(Mx) + _vn_entropy(My) - _vn_entropy(Mj)


# ================================================================ NV-5
# A.8(iii) is proved for the paired feature map; for the deployed map the main text needs that a
# dependent pair keeps I > 0 as D grows (no dimension at which the statistic goes blind), and
# that independence sits at 0. Both are measured here on a sample large enough that the O(1/m)
# plug-in bias is negligible relative to the signal.
# (scenario, level, which regime carries the dependence): D1/D2/D3 change INTO the dependent
# regime, D4 changes OUT of it, so the dependent block is the post-change one except for D4.
FAMILIES = {
    "independent": None,
    "gaussian_r0.5": ("D1", 0.5, "post"),
    "uncorrelated_a0.9": ("D2", 0.9, "post"),
    "clayton_tau0.5": ("D3", 0.5, "post"),
    "signmixed_r0.85": ("D4", 0.85, "pre"),
}


def _nv5_job(a):
    fam, D, rep, m = a
    rng = np.random.default_rng([BASE_SEED, 16, zlib.crc32(fam.encode()) % 10**6, D, rep])
    if fam == "independent":
        x, y = rng.standard_normal(m), rng.standard_normal(m)
    else:
        scen, level, which = FAMILIES[fam]
        li = SCENARIOS[scen]["levels"].index(level)
        # the dependent regime of the scenario, sampled as one homogeneous block
        smp = generate(scen, level, rep, n=2 * m, tau=m, null=False, base_seed=BASE_SEED, level_idx=li)
        Z = smp["Z"]
        sl = slice(m, None) if which == "post" else slice(0, m)
        x, y = Z[sl, 0], Z[sl, 1]
    # bias-corrected: subtract the plug-in value under the independence null on the same draw
    idx = rng.permutation(len(y))
    raw = _domi_from_pair(x, y, D)
    nul = _domi_from_pair(x, y[idx], D)
    return dict(fam=fam, D=D, rep=rep, raw=float(raw), null=float(nul), corrected=float(raw - nul))


def run_nv5(procs, reps, m=20000):
    Ds = [4, 8, 16, 32]
    jobs = [(f, D, r, m) for f in FAMILIES for D in Ds for r in range(reps)]
    t0 = time.time()
    with Pool(procs, maxtasksperchild=8) as pool:
        recs = pool.map(_nv5_job, jobs, chunksize=1)
    res = {}
    for f in FAMILIES:
        res[f] = {}
        for D in Ds:
            v = [r["corrected"] for r in recs if r["fam"] == f and r["D"] == D]
            vr = [r["raw"] for r in recs if r["fam"] == f and r["D"] == D]
            res[f][str(D)] = dict(mean=float(np.mean(v)), sd=float(np.std(v)),
                                  raw_mean=float(np.mean(vr)), n_rep=len(v))
    out = dict(config=dict(m=m, reps=reps, Ds=Ds, base_seed=BASE_SEED,
                           note="corrected = plug-in DOMI minus its value on the same sample with Y permuted"),
               results=res)
    _write_records("records_nv5_reps.csv", ["fam", "D", "rep", "corrected", "raw"],
                   [dict(fam=r["fam"], D=r["D"], rep=i, corrected=r["corrected"], raw=r["raw"])
                    for i, r in enumerate(recs)])
    json.dump(out, open(os.path.join(OUT, "nv5.json"), "w"), indent=1)
    for f in FAMILIES:
        say("NV-5", f + ": " + ", ".join(f"D={D} {res[f][str(D)]['mean']:.4f}" for D in Ds))
    say("NV-5", f"done in {time.time()-t0:.0f}s")
    return out


# ================================================================ NV-6
# The limit theory for the studentized scan is open. Here the null law of the deployed
# statistic -- the symmetric-studentized maximum, exactly the object P1 calibrates -- is
# measured across n, to see whether it stabilizes (as a limit theory would require) or drifts.
def _nv6_job(a):
    n, rep, K, D, w = a
    rng = np.random.default_rng([BASE_SEED, 17, n, rep])
    x = rng.standard_normal(n)
    y = 0.5 * x + np.sqrt(1 - 0.25) * rng.standard_normal(n)  # dependent, constant in time: H0

    def curve(xa, ya):
        ctx = DOMIContext(xa[:, None], ya[:, None], w, D=D, seed=2026, n_perm=0)
        return domi_stats(ctx, "global")["DOMI-diff"]

    obs = curve(x, y)
    R = np.empty((K, len(obs)))
    for k in range(K):
        idx = rng.permutation(n)
        R[k] = curve(x[idx], y[idx])
    A = np.vstack([obs[None, :], R])
    mu, sd = A.mean(0), A.std(0) + 1e-12
    T = np.max((A - mu) / sd, axis=1)          # all K+1 studentized maxima
    return dict(n=n, T_obs=float(T[0]), T_rep=[float(v) for v in T[1:]])


def run_nv6(procs, reps, K=49, D=8):
    ns = [300, 600, 1200, 2400]
    jobs = [(n, r, K, D, max(30, n // 10)) for n in ns for r in range(reps)]
    t0 = time.time()
    with Pool(procs, maxtasksperchild=8) as pool:
        recs = pool.map(_nv6_job, jobs, chunksize=1)
    qs = [0.5, 0.75, 0.9, 0.95, 0.99]
    res = {}
    pooled = {}
    for n in ns:
        T = np.array([v for r in recs if r["n"] == n for v in ([r["T_obs"]] + r["T_rep"])])
        pooled[n] = T
        res[str(n)] = dict(n_draws=int(T.size), mean=float(T.mean()), sd=float(T.std()),
                           quantiles={str(q): float(np.quantile(T, q)) for q in qs})
    # Kolmogorov distance between consecutive sample sizes: shrinking => stabilizing
    ks = {}
    for i in range(len(ns) - 1):
        a, b = pooled[ns[i]], pooled[ns[i + 1]]
        grid = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 2000)
        Fa = np.searchsorted(np.sort(a), grid, side="right") / a.size
        Fb = np.searchsorted(np.sort(b), grid, side="right") / b.size
        ks[f"{ns[i]}->{ns[i+1]}"] = float(np.max(np.abs(Fa - Fb)))
    # level of the p-value rule at each n (should be exact by P1 -- a control on this experiment)
    lev = {}
    for n in ns:
        ps = []
        for r in recs:
            if r["n"] != n:
                continue
            T0, Tr = r["T_obs"], r["T_rep"]
            ps.append((1.0 + sum(ge(t, T0) for t in Tr)) / (len(Tr) + 1.0))
        lev[str(n)] = {str(al): float(np.mean([p <= al for p in ps])) for al in (0.05, 0.10, 0.20)}
    out = dict(config=dict(ns=ns, reps=reps, K=K, D=D, base_seed=BASE_SEED,
                           null="constant Gaussian dependence r=0.5 throughout (no change point)"),
               law=res, ks_consecutive=ks, pvalue_level=lev)
    _write_records("records_nv6_maxima.csv",
                   ["n", "rep", "T_obs", "T_rep"],
                   [dict(n=r["n"], rep=i, T_obs=r["T_obs"],
                         T_rep=" ".join(f"{v:.6f}" for v in r["T_rep"])) for i, r in enumerate(recs)])
    json.dump(out, open(os.path.join(OUT, "nv6.json"), "w"), indent=1)
    for n in ns:
        say("NV-6", f"n={n}: mean {res[str(n)]['mean']:.3f} q95 {res[str(n)]['quantiles']['0.95']:.3f} "
                     f"level@0.05 {lev[str(n)]['0.05']:.3f}")
    say("NV-6", "KS between consecutive n: " + ", ".join(f"{k} {v:.3f}" for k, v in ks.items()))
    say("NV-6", f"done in {time.time()-t0:.0f}s")
    return out


# ================================================================ NV-7
# Consistency of Holevo partitioning in the number of segments is open. The MB design of Section 6.3
# is run at three sample sizes with the break fractions held fixed, so that only n varies.
def _nv7_sample(n, rep, a, null, rng_seed):
    """MB design (scenario P1) rescaled to length n: breaks at n/4, n/2, 3n/4."""
    rng = np.random.default_rng([BASE_SEED, 18, n, rep, int(null), rng_seed])
    E = rng.standard_normal((n, 3))
    X = E[:, :2].copy()
    t1, t2, t3 = n // 4, n // 2, 3 * n // 4
    if not null:
        X[t1:t2, 1] = np.sqrt(1 - a * a) * E[t1:t2, 1] + a * np.abs(E[t1:t2, 0]) * E[t1:t2, 2]
        r = 0.5
        X[t3:, 1] = r * E[t3:, 0] + np.sqrt(1 - r * r) * E[t3:, 1]
    return X, [t1, t2, t3]


def _nv7_job(a):
    n, rep, null, betas, D, level = a
    Z, taus = _nv7_sample(n, rep, level, null, 0)
    X, Y = Z[:, [0]], Z[:, [1]]
    fx = unit_rff(ranks01(X), D, 2026)
    fy = unit_rff(ranks01(Y), D, 2027)
    J = np.einsum("ti,tj->tij", fx, fy).reshape(n, D * D)
    step = max(1, n // 180)                      # fixed grid resolution in relative terms
    grid = np.arange(0, n + 1, step)
    C = P.cost_matrix_vn(MomentCache(J), grid)
    prng = np.random.default_rng([BASE_SEED, 18, n, rep, int(null), 13])
    Cp = np.zeros_like(C)
    for _ in range(3):                            # same 3-permutation bias correction as Section 4.4
        Cp += P.cost_matrix_vn(MomentCache(J[prng.permutation(n)]), grid)
    C = C - Cp / 3
    out = {}
    for b in betas:
        cps = [int(grid[i]) for i in P.pelt_from_costs(C, b)]
        out[b] = dict(k=len(cps), cps=cps,
                      ari=float(P.rand_index_adj(taus, cps, n)),
                      haus=float(P.hausdorff(taus, cps, n)))
    return dict(n=n, rep=rep, null=null, out=out)


REC18 = []


def _write_records(name, fields, rows):
    import csv as _csv
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_nv7(procs, reps, D=8, level=0.9):
    global REC18
    ns = [1800, 3600, 7200]
    betas = [float(b) for b in np.linspace(2, 60, 15)]
    res, t0 = {}, time.time()
    with Pool(procs, maxtasksperchild=4) as pool:
        for n in ns:
            t1 = time.time()
            nulls = pool.map(_nv7_job, [(n, r, True, betas, D, level) for r in range(reps)], chunksize=1)
            # penalty: smallest beta with at most 5% of null replicates returning any break
            beta = betas[-1]
            for b in betas:
                fa = float(np.mean([d["out"][b]["k"] > 0 for d in nulls]))
                if fa <= 0.05:
                    beta = b
                    break
            alts = pool.map(_nv7_job, [(n, r, False, betas, D, level) for r in range(reps)], chunksize=1)
            k3 = float(np.mean([d["out"][beta]["k"] == 3 for d in alts]))
            ari = float(np.mean([d["out"][beta]["ari"] for d in alts]))
            kk = [d["out"][beta]["k"] for d in alts]
            REC18 += [dict(n=n, side="null", rep=d["rep"], beta=beta, k=d["out"][beta]["k"],
                           ari=d["out"][beta]["ari"], haus=d["out"][beta]["haus"]) for d in nulls]
            REC18 += [dict(n=n, side="alt", rep=d["rep"], beta=beta, k=d["out"][beta]["k"],
                           ari=d["out"][beta]["ari"], haus=d["out"][beta]["haus"]) for d in alts]
            res[str(n)] = dict(beta=beta, null_fa=float(np.mean([d["out"][beta]["k"] > 0 for d in nulls])),
                               p_k3=k3, ari_mean=ari, k_mean=float(np.mean(kk)),
                               k_hist={str(v): int(kk.count(v)) for v in sorted(set(kk))},
                               rel_haus_mean=float(np.mean([d["out"][beta]["haus"] for d in alts]) / n))
            say("NV-7", f"n={n}: beta={beta:.1f} FA={res[str(n)]['null_fa']:.3f} "
                         f"P(K=3)={k3:.3f} ARI={ari:.3f} E[K]={res[str(n)]['k_mean']:.2f} ({time.time()-t1:.0f}s)")
    out = dict(config=dict(ns=ns, reps=reps, D=D, level=level, base_seed=BASE_SEED,
                           note="MB design with break fractions fixed at 1/4, 1/2, 3/4; grid ~180 points at every n"),
               results=res)
    _write_records("records_nv7_reps.csv", ["n", "side", "rep", "beta", "k", "ari", "haus"], REC18)
    json.dump(out, open(os.path.join(OUT, "nv7.json"), "w"), indent=1)
    say("NV-7", f"done in {time.time()-t0:.0f}s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["nv5", "nv6", "nv7", "all"])
    ap.add_argument("--procs", type=int, default=28)
    ap.add_argument("--reps", type=int, default=0, help="0 = per-part default")
    a = ap.parse_args()
    if a.part in ("nv5", "all"):
        run_nv5(a.procs, a.reps or 20)
    if a.part in ("nv6", "all"):
        run_nv6(a.procs, a.reps or 200)
    if a.part in ("nv7", "all"):
        run_nv7(a.procs, a.reps or 200)
    say("theory_sims", "all done")
