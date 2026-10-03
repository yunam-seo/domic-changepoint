#!/usr/bin/env python
"""EXPERIMENT runner.  Parts: e1 e2 e3 e4.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_experiment.py --part e1 --procs 24
Writes 04_DAOU/EXPERIMENT/<part>/results.csv (+ run.log per part).
DOMI statistics at D=8 (part e4 also varies D: 4, 16).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from dots.synth import SCENARIOS, DEFAULT_W, generate  # noqa: E402
from dots.detect import Context  # noqa: E402
from dots.baselines import BASELINES  # noqa: E402
from dots.extras import studentize  # noqa: E402
from dots.evaluate import summarize_alt, aggregate  # noqa: E402
from dots.persist import save_records  # noqa: E402
from dots.domi import DOMIContext, domi_stats, DEP_BASELINES, ranks01, unit_rff  # noqa: E402
from dots.encode import MomentCache  # noqa: E402
from dots import pelt as P  # noqa: E402
from dots.perm import ge  # noqa: E402
from run_ablation import mmd_global_multi  # noqa: E402

BASE = os.path.join(ROOT, "04_DAOU", "EXPERIMENT")
CFG = dict(n=600, tau=300, tol=30, base_seed=20260825, fpr=0.05, D=8, w=DEFAULT_W, mults=[0.5, 1.0, 2.0])


def say(part, m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(os.path.join(BASE, part, "run.log"), "a") as f:
        f.write(line + "\n")


def write_rows(part, rows, name="results.csv"):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(BASE, part, name), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


# ================================================================ E1
def e1_stats(sample, w, cfg):
    Z = sample["Z"]
    X, Y = Z[:, sample["blocks"][0]], Z[:, sample["blocks"][1]]
    st = {}
    ctx = DOMIContext(X, Y, w, D=cfg["D"], seed=2026)
    for k, v in domi_stats(ctx, "global").items():
        st[f"{k}|g"] = v
    for k, f in DEP_BASELINES.items():
        st[f"{k}|g"] = f(ctx, "global")
    grid = ctx.grid
    st.update(mmd_global_multi(Z, grid, cfg["mults"], Z.shape[0]))
    st["GaussLR|g"] = BASELINES["GaussLR"](Context(Z, w, c_aug=1.0), "global")
    return st, grid


def _e1_null(a):
    s, r, cfg = a
    smp = generate(s, SCENARIOS[s]["levels"][0], r, null=True, base_seed=cfg["base_seed"], level_idx=0)
    return e1_stats(smp, cfg["w"][s], cfg)[0]


def _e1_alt(a):
    s, li, r, cfg, thr, mu0, sd0 = a
    smp = generate(s, SCENARIOS[s]["levels"][li], r, null=False, base_seed=cfg["base_seed"], level_idx=li)
    st, grid = e1_stats(smp, cfg["w"][s], cfg)
    for k in list(st):
        if k.endswith("|g"):
            st[k[:-1] + "gs"] = studentize(st[k], mu0[k], sd0[k])
    return summarize_alt(st, grid, smp["tau"], cfg["w"][s], cfg["n"], thr, cfg["tol"])


def run_e1(cfg, procs, reps):
    rows = []
    with Pool(procs, maxtasksperchild=30) as pool:
        for s in ["D1", "D2", "D3", "D4", "M1"]:
            t0 = time.time()
            nulls = pool.map(_e1_null, [(s, r, cfg) for r in range(reps)], chunksize=4)
            save_records(os.path.join(BASE, "e1"), "records_null.csv", nulls, {"scen": s})
            keys = list(nulls[0].keys())
            half = reps // 2
            mu0 = {k: np.mean([d[k] for d in nulls[:half]], axis=0) for k in keys if k.endswith("|g")}
            sd0 = {k: np.std([d[k] for d in nulls[:half]], axis=0) for k in keys if k.endswith("|g")}
            thr = {k: float(np.quantile([np.nanmax(d[k]) for d in nulls], 1 - cfg["fpr"])) for k in keys}
            for k in keys:
                if k.endswith("|g"):
                    thr[k[:-1] + "gs"] = float(np.quantile([np.nanmax(studentize(d[k], mu0[k], sd0[k])) for d in nulls[half:]], 1 - cfg["fpr"]))
            say("e1", f"{s}: null {reps} reps {time.time()-t0:.0f}s")
            for li, level in enumerate(SCENARIOS[s]["levels"]):
                t1 = time.time()
                recs = pool.map(_e1_alt, [(s, li, r, cfg, thr, mu0, sd0) for r in range(reps)], chunksize=4)
                # replicates are kept, not only their average (see dots/persist.py)
                save_records(os.path.join(BASE, "e1"), "records_alt.csv", recs, {"scen": s, "level": level})
                agg = aggregate(recs, thr)
                for k, v in agg.items():
                    name, mode = k.split("|")
                    rows.append(dict(scen=s, level=level, method=name, mode=mode, key=k, **v))
                top = sorted(agg.items(), key=lambda kv: -kv[1]["power"])[:5]
                say("e1", f"{s} level={level}: {time.time()-t1:.0f}s | " + ", ".join(f"{k}={v['power']:.2f}" for k, v in top))
    write_rows("e1", rows)


# ================================================================ E2
def _e2_job(a):
    li, r, null, betas_vn, betas_g, betas_ms, cfg = a
    level = SCENARIOS["P1"]["levels"][li]
    smp = generate("P1", level, r, null=null, base_seed=cfg["base_seed"], level_idx=li)
    Z = smp["Z"]
    n = Z.shape[0]
    X, Y = Z[:, smp["blocks"][0]], Z[:, smp["blocks"][1]]
    U = np.hstack([ranks01(X), ranks01(Y)])
    FX = unit_rff(ranks01(X), cfg["D"], 2026)
    FY = unit_rff(ranks01(Y), cfg["D"], 2027)
    J = np.einsum("ti,tj->tij", FX, FY).reshape(n, -1)
    grid = np.arange(0, n + 1, 10)
    Cvn = P.cost_matrix_vn(MomentCache(J), grid)
    Cg = P.cost_matrix_gauss2(MomentCache(U), grid)
    # segment-wise bias correction: subtract mean cost under joint pair permutation (exchangeability null).
    # removes the plug-in entropy / logdet length bias so that split gains are centered at 0 under H0.
    prng = np.random.default_rng([cfg["base_seed"], r, int(null), li, 13])
    n_perm_bc = 3
    Cvn_p = np.zeros_like(Cvn)
    Cg_p = np.zeros_like(Cg)
    for _ in range(n_perm_bc):
        idx = prng.permutation(n)
        Cvn_p += P.cost_matrix_vn(MomentCache(J[idx]), grid)
        Cg_p += P.cost_matrix_gauss2(MomentCache(U[idx]), grid)
    Cvn = Cvn - Cvn_p / n_perm_bc
    Cg = Cg - Cg_p / n_perm_bc
    out = {}
    for tag, C, betas in [("Holevo-partition", Cvn, betas_vn), ("PELT-Gauss", Cg, betas_g)]:
        out[tag] = {b: [int(grid[i]) for i in P.pelt_from_costs(C, b)] for b in betas}
    # BinSeg-MMD: max split gain of MMD statistic per segment (reuse global MMD on subsegments)
    from dots.detect import sqdist, prefix2d, block_sum
    D2 = sqdist(Z)
    med = np.median(D2[np.triu_indices(n, 1)])
    K = prefix2d(np.exp(-D2 / (med if med > 0 else 1.0)))

    def stat_fn(a0, b0):
        w = cfg["w"]["P1"]
        best = (None, -np.inf)
        for t in range(a0 + w, b0 - w + 1, 10):
            n1, n2 = t - a0, b0 - t
            kLL = block_sum(K, a0, t, a0, t) / n1**2
            kRR = block_sum(K, t, b0, t, b0) / n2**2
            kLR = block_sum(K, a0, t, t, b0) / (n1 * n2)
            g = n1 * n2 / (b0 - a0) * max(kLL + kRR - 2 * kLR, 0)
            if g > best[1]:
                best = (t, g)
        return best

    cands = P.binseg_from_stat(stat_fn, n, cfg["w"]["P1"])
    out["BinSeg-MMD"] = {b: sorted(t for t, g in cands if g > b) for b in betas_ms}
    return out


def run_e2(cfg, procs, reps):
    beta_grids = dict(betas_vn=list(np.linspace(2, 60, 15)), betas_g=list(np.linspace(2, 60, 15)),
                      betas_ms=list(np.linspace(0.05, 3.0, 15)))
    with Pool(procs, maxtasksperchild=30) as pool:
        t0 = time.time()
        nulls = pool.map(_e2_job, [(0, r, True, *beta_grids.values(), cfg) for r in range(reps)], chunksize=2)
        save_records(os.path.join(BASE, "e2"), "records_null.csv", nulls, {})
        say("e2", f"null {reps} reps {time.time()-t0:.0f}s")
        # pick beta per method: smallest beta with <=5% of null reps having any CP
        chosen = {}
        for m in nulls[0]:
            # PELT-Gauss is selected on betas_vn, which equals its own grid betas_g (both linspace(2, 60, 15))
            for b in beta_grids["betas_ms" if m == "BinSeg-MMD" else "betas_vn"]:
                fa = np.mean([len(d[m][b]) > 0 for d in nulls])
                if fa <= cfg["fpr"]:
                    chosen[m] = (b, fa)
                    break
            else:
                bb = beta_grids["betas_ms" if m == "BinSeg-MMD" else "betas_vn"][-1]
                chosen[m] = (bb, np.mean([len(d[m][bb]) > 0 for d in nulls]))
        say("e2", f"chosen betas: {chosen}")
        rows = []
        for li, level in enumerate(SCENARIOS["P1"]["levels"]):
            t1 = time.time()
            alts = pool.map(_e2_job, [(li, r, False, *beta_grids.values(), cfg) for r in range(reps)], chunksize=2)
            save_records(os.path.join(BASE, "e2"), "records_alt.csv", alts, {"level": level})
            taus = [450, 900, 1350]
            for m in alts[0]:
                b, fa = chosen[m]
                cps = [d[m][b] for d in alts]
                rows.append(dict(scen="P1", level=level, method=m, beta=b, null_fa=fa,
                                 k_err_mean=float(np.mean([abs(len(c) - 3) for c in cps])),
                                 k_correct=float(np.mean([len(c) == 3 for c in cps])),
                                 hausdorff_median=float(np.median([P.hausdorff(taus, c, 1800) for c in cps])),
                                 ari_mean=float(np.mean([P.rand_index_adj(taus, c, 1800) for c in cps]))))
            say("e2", f"P1 level={level}: {time.time()-t1:.0f}s | " +
                ", ".join(f"{r['method']}: K3={r['k_correct']:.2f} H={r['hausdorff_median']:.0f} ARI={r['ari_mean']:.2f}" for r in rows[-3:]))
    write_rows("e2", rows)


# ================================================================ E3
def _e3_job(a):
    scen, li, r, null, cfg = a
    level = SCENARIOS[scen]["levels"][li]
    smp = generate(scen, level, r, null=null, base_seed=cfg["base_seed"], level_idx=li)
    Z = smp["Z"]
    X, Y = Z[:, smp["blocks"][0]], Z[:, smp["blocks"][1]]
    w = cfg["w"][scen]

    def curves(Xa, Ya):
        ctx = DOMIContext(Xa, Ya, w, D=cfg["D"], seed=2026, n_perm=0)
        g = domi_stats(ctx, "global")
        return {"DOMI-diff": g["DOMI-diff"], "HSIC-diff": DEP_BASELINES["HSIC-diff"](ctx, "global"),
                "CopulaCvM": DEP_BASELINES["CopulaCvM"](ctx, "global")}, ctx.grid

    obs, grid = curves(X, Y)
    rng = np.random.default_rng([cfg["base_seed"], r, int(null), li, 11])
    K = cfg["K"]
    reps = {m: [] for m in obs}
    for k in range(K):
        idx = rng.permutation(len(Y))
        c, _ = curves(X[idx], Y[idx])
        for m in c:
            reps[m].append(c[m])
    res = {}
    for m in obs:
        R = np.array(reps[m])
        # Symmetric studentization (per-t moments from all K+1 curves) plus the permutation
        # p-value: exactly the statistic and the rule Proposition P1 covers. Leave-one-out
        # moments are not covered verbatim, and the (1-alpha) replica quantile drops the "+1"
        # and is anticonservative at finite K; see Section 4.3 and Supplementary Section B.3.
        A = np.vstack([obs[m][None, :], R])
        mu, sd = A.mean(0), A.std(0) + 1e-12
        Tall = [float(np.max((A[i] - mu) / sd)) for i in range(K + 1)]
        T = (obs[m] - mu) / sd
        pval = (1.0 + sum(ge(t, Tall[0]) for t in Tall[1:])) / (K + 1.0)
        res[m] = dict(detect=bool(pval <= cfg["fpr"]), tau_hat=int(grid[int(np.argmax(T))]))
    return res


def run_e3(cfg, procs, reps):
    cfg = dict(cfg)
    cfg["K"] = 99
    rows = []
    with Pool(procs, maxtasksperchild=6) as pool:
        for scen, li in [("D1", 1), ("D2", 1), ("D2", 2), ("D4", 1), ("D4", 2), ("M1", 1)]:
            level = SCENARIOS[scen]["levels"][li]
            t0 = time.time()
            nulls = pool.map(_e3_job, [(scen, li, r, True, cfg) for r in range(reps)], chunksize=1)
            alts = pool.map(_e3_job, [(scen, li, r, False, cfg) for r in range(reps)], chunksize=1)
            save_records(os.path.join(BASE, "e3"), "records_null.csv", nulls, {"scen": scen, "level": level})
            save_records(os.path.join(BASE, "e3"), "records_alt.csv", alts, {"scen": scen, "level": level})
            for m in nulls[0]:
                fa = float(np.mean([d[m]["detect"] for d in nulls]))
                pw = float(np.mean([d[m]["detect"] and abs(d[m]["tau_hat"] - cfg["tau"]) <= cfg["tol"] for d in alts]))
                rows.append(dict(scen=scen, level=level, method=m, K=cfg["K"], fpr_perm=fa, power_perm=pw))
            say("e3", f"{scen} {level}: {time.time()-t0:.0f}s | " +
                ", ".join(f"{r['method']}: P={r['power_perm']:.2f} FPR={r['fpr_perm']:.2f}" for r in rows[-3:]))
    write_rows("e3", rows)


# ================================================================ E4
def _e4_job(a):
    n, w, D, r, null, cfg = a
    smp = generate("D2", 0.7, r, n=n, tau=n // 2, null=null, base_seed=cfg["base_seed"] + 5, level_idx=1)
    Z = smp["Z"]
    X, Y = Z[:, smp["blocks"][0]], Z[:, smp["blocks"][1]]
    ctx = DOMIContext(X, Y, w, D=D, seed=2026, n_perm=0)
    g = domi_stats(ctx, "global")
    return g["DOMI-diff"], ctx.grid


def run_e4(cfg, procs, reps):
    base = dict(n=600, w=60, D=8)
    configs = [dict(base)] + [dict(base, n=v) for v in (300, 1200)] + [dict(base, w=v) for v in (40, 100)] + [dict(base, D=v) for v in (4, 16)]
    rows = []
    with Pool(procs, maxtasksperchild=30) as pool:
        for c in configs:
            t0 = time.time()
            nulls = pool.map(_e4_job, [(c["n"], c["w"], c["D"], r, True, cfg) for r in range(reps)], chunksize=4)
            alts = pool.map(_e4_job, [(c["n"], c["w"], c["D"], r, False, cfg) for r in range(reps)], chunksize=4)
            save_records(os.path.join(BASE, "e4"), "records_null.csv", nulls, dict(c))
            save_records(os.path.join(BASE, "e4"), "records_alt.csv", alts, dict(c))
            grid = nulls[0][1]
            half = reps // 2
            # studentize with null mean/sd curves (first half), threshold from second half (as in E1 |gs)
            mu0 = np.mean([d[0] for d in nulls[:half]], axis=0)
            sd0 = np.std([d[0] for d in nulls[:half]], axis=0) + 1e-12
            h = float(np.quantile([np.nanmax((d[0] - mu0) / sd0) for d in nulls[half:]], 1 - cfg["fpr"]))
            tol = max(30 * c["n"] // 600, 15)
            Ts = [((d[0] - mu0) / sd0) for d in alts]
            pw = float(np.mean([np.nanmax(T) > h and abs(int(grid[int(np.nanargmax(T))]) - c["n"] // 2) <= tol for T in Ts]))
            rows.append(dict(**c, method="DOMI-diff|gs", power=pw))
            say("e4", f"{c}: {time.time()-t0:.0f}s | power={pw:.2f}")
    write_rows("e4", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["e1", "e2", "e3", "e4"])
    ap.add_argument("--reps", type=int, default=0)
    ap.add_argument("--procs", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(os.path.join(BASE, a.part), exist_ok=True)
    cfg = dict(CFG)
    default_reps = dict(e1=500, e2=200, e3=100, e4=200)
    reps = a.reps or default_reps[a.part]
    json.dump(dict(cfg=cfg, reps=reps, part=a.part), open(os.path.join(BASE, a.part, "config.json"), "w"), indent=1)
    dict(e1=run_e1, e2=run_e2, e3=run_e3, e4=run_e4)[a.part](cfg, a.procs, reps)
    print(f"{a.part.upper()} done", flush=True)
