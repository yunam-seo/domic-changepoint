#!/usr/bin/env python
"""Section 4.3 claim, at the deployed configuration: under a null in which the series is dependent
throughout, calibrating by permuting only the Y rows (an independence null) is anticonservative,
whereas joint pair permutation is valid.

The rates quoted in Section 4.3 come from this script, at the configuration used in the paper:
D=8, K=99, 200 replicates, the DOMI difference (key DOMI-diff), global studentized statistic.

Nulls: no change point anywhere, both halves from the SAME dependent regime.
  - "dependent null"   : Gaussian copula at rho=0.6 throughout   (relevant to testing for the
                         disappearance of dependence, paper scenario S4)
  - "nonlinear null"   : y = sqrt(1-a^2) e1 + a|x| e2 at a=0.7 throughout (paper scenario S2 regime)
  - "independent null" : X, Y independent throughout (control; both schemes should be valid here)

Writes 04_DAOU/EXPERIMENT/perm_null/{results.csv, results.json, records_reps.csv}
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import zlib

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.domi import ranks01, unit_rff  # noqa: E402
from dots.perm import ge  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "perm_null")
CFG = dict(n=600, w=60, D=8, K=99, reps=200, alpha=0.05, base_seed=20260903)


def ent(M):
    lam = np.clip(np.linalg.eigvalsh(M), 1e-300, None)
    lam = lam[lam > 1e-14]
    return float(-(lam * np.log(lam)).sum())


def domi_curve(X, Y, w, D):
    """Weighted global DOMI-difference curve Q(t) over the candidate grid, from prefix sums."""
    n = len(X)
    FX = unit_rff(ranks01(X[:, None]), D, 2026)
    FY = unit_rff(ranks01(Y[:, None]), D, 2027)
    J = np.einsum("ti,tj->tij", FX, FY).reshape(n, D * D)
    cx = np.zeros((n + 1, D, D)); cy = np.zeros((n + 1, D, D)); cj = np.zeros((n + 1, D * D, D * D))
    np.cumsum(np.einsum("ti,tj->tij", FX, FX), axis=0, out=cx[1:])
    np.cumsum(np.einsum("ti,tj->tij", FY, FY), axis=0, out=cy[1:])
    np.cumsum(np.einsum("ti,tj->tij", J, J), axis=0, out=cj[1:])

    def I(a, b):
        m = b - a
        return (ent((cx[b] - cx[a]) / m) + ent((cy[b] - cy[a]) / m) - ent((cj[b] - cj[a]) / m))

    grid = np.arange(w, n - w + 1)
    return grid, np.array([np.sqrt(t * (n - t) / n) * abs(I(0, t) - I(t, n)) for t in grid])


def gen_null(kind, rep, cfg):
    """No change point: both halves from the same regime."""
    rng = np.random.default_rng(cfg["base_seed"] + 1000 * rep + zlib.crc32(kind.encode()) % 997)
    n = cfg["n"]
    if kind == "dependent":
        rho = 0.6
        x = rng.standard_normal(n)
        y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    elif kind == "nonlinear":
        a = 0.7
        x = rng.standard_normal(n)
        y = np.sqrt(1 - a**2) * rng.standard_normal(n) + a * np.abs(x) * rng.standard_normal(n)
    else:
        x, y = rng.standard_normal(n), rng.standard_normal(n)
    return x, y


def _job(a):
    kind, rep, cfg = a
    x, y = gen_null(kind, rep, cfg)
    K, D, w = cfg["K"], cfg["D"], cfg["w"]
    rng = np.random.default_rng(cfg["base_seed"] + 77 * rep)
    grid, obs = domi_curve(x, y, w, D)

    out = {}
    for scheme in ("pair", "yonly"):
        reps_curves = np.empty((K, len(grid)))
        for k in range(K):
            idx = rng.permutation(len(x))
            if scheme == "pair":          # jointly permute the pairs -> preserves dependence
                xp, yp = x[idx], y[idx]
            else:                          # permute Y only -> destroys dependence (independence null)
                xp, yp = x, y[idx]
            reps_curves[k] = domi_curve(xp, yp, w, D)[1]
        # symmetric studentization: per-t moments over ALL K+1 curves, the variant P1 covers
        # exactly and the one Algorithm 1 deploys (leave-one-out moments are not covered
        # and inflate the pair level)
        A = np.vstack([obs[None, :], reps_curves])
        mu, sd = A.mean(0), A.std(0) + 1e-12
        Tobs = np.nanmax((obs - mu) / sd)
        Trep = np.array([np.nanmax((reps_curves[k] - mu) / sd) for k in range(K)])
        thr = float(np.quantile(Trep, 1 - cfg["alpha"]))
        pval = (1 + int(np.sum(ge(Trep, Tobs)))) / (K + 1)
        out[scheme] = dict(reject_thr=bool(Tobs > thr), reject_p=bool(pval <= cfg["alpha"]))
    return kind, out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=CFG["reps"])
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    cfg = dict(CFG); cfg["reps"] = a.reps
    os.makedirs(OUT, exist_ok=True)

    kinds = ["dependent", "nonlinear", "independent"]
    rows, res, rec_rows = [], {}, []
    with Pool(a.procs, maxtasksperchild=20) as pool:
        for kind in kinds:
            t0 = time.time()
            got = pool.map(_job, [(kind, r, cfg) for r in range(cfg["reps"])], chunksize=2)
            for ri, g in enumerate(got):
                for scheme in ("pair", "yonly"):
                    rec_rows.append(dict(null=kind, scheme=scheme, rep=ri,
                                         reject_thr=int(g[1][scheme]["reject_thr"]),
                                         reject_p=int(g[1][scheme]["reject_p"])))
            for scheme in ("pair", "yonly"):
                fa_thr = float(np.mean([g[1][scheme]["reject_thr"] for g in got]))
                fa_p = float(np.mean([g[1][scheme]["reject_p"] for g in got]))
                rows.append(dict(null=kind, scheme=scheme, K=cfg["K"], D=cfg["D"], reps=cfg["reps"],
                                 fpr_threshold=fa_thr, fpr_pvalue=fa_p))
                res[f"{kind}|{scheme}"] = dict(fpr_threshold=fa_thr, fpr_pvalue=fa_p)
            print(f"[{time.strftime('%H:%M:%S')}] {kind} ({time.time()-t0:.0f}s): "
                  f"pair FPR={res[f'{kind}|pair']['fpr_pvalue']:.3f}  "
                  f"yonly FPR={res[f'{kind}|yonly']['fpr_pvalue']:.3f}", flush=True)

    with open(os.path.join(OUT, "records_reps.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rec_rows[0].keys())); wr.writeheader(); wr.writerows(rec_rows)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    json.dump(dict(config=cfg, results=res), open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print("perm_null done", flush=True)
