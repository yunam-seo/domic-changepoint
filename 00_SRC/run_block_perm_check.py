#!/usr/bin/env python
"""Serial dependence and the permutation null (Section 4.3; Supplementary Section B.17).

Proposition P1 is exact under exchangeability of the pairs, which fails when the series has serial
dependence (volatility clustering, autoregressive levels). This script quantifies the resulting
error and shows that permuting whole BLOCKS restores the level.

Block permutation: partition 1..n into floor(n/b) consecutive blocks and apply a random permutation
to the block ORDER, jointly to X and Y. Every observation is used exactly once, so this is a genuine
permutation, not a bootstrap resample; if the blocks are exchangeable under the no-change
hypothesis -- approximately true for a stationary series whose dependence range is short relative
to b -- then the P1 argument applies verbatim over the block-permutation group.

Nulls (no change point anywhere; contemporaneous dependence CONSTANT in time):
  iid    : serially independent, Gaussian copula rho=0.5           (control: pair permutation valid)
  ar1    : AR(1) levels, phi=0.6, correlated innovations           (serial dependence in the mean)
  garch  : GARCH(1,1) volatility clustering, correlated innovations (serial dependence in scale)

Studentization is symmetric: the pointwise moments are computed from all K+1 curves, observed
included, the construction P1 and P5 cover.

Writes 04_DAOU/EXPERIMENT/block_perm/{results.csv, results.json, records_<null>.csv.gz}
"""
from __future__ import annotations

import csv, json, os, sys, time
from multiprocessing import Pool

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from run_perm_null_check import domi_curve  # noqa: E402
from dots.persist import save_records  # noqa: E402
from dots.perm import ge  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "block_perm")
CFG = dict(n=600, w=60, D=8, K=99, reps=200, alpha=0.05, base_seed=20260905, rho=0.5)
BLOCKS = [20, 50]


def gen_null(kind, rep, cfg):
    """Stationary null with constant contemporaneous dependence and no change point."""
    rng = np.random.default_rng(cfg["base_seed"] + 1000 * rep + {"iid": 0, "ar1": 1, "garch": 2}[kind])
    n, rho = cfg["n"], cfg["rho"]
    e1 = rng.standard_normal(n)
    e2 = rho * e1 + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    if kind == "iid":
        return e1, e2
    if kind == "ar1":
        phi = 0.6
        x = np.empty(n); y = np.empty(n)
        x[0], y[0] = e1[0], e2[0]
        for t in range(1, n):
            x[t] = phi * x[t - 1] + e1[t]
            y[t] = phi * y[t - 1] + e2[t]
        return x, y
    # garch: common volatility dynamics per series, correlated standardized innovations
    om, al, be = 0.05, 0.10, 0.85
    hx = np.empty(n); hy = np.empty(n); x = np.empty(n); y = np.empty(n)
    hx[0] = hy[0] = om / (1 - al - be)
    x[0], y[0] = np.sqrt(hx[0]) * e1[0], np.sqrt(hy[0]) * e2[0]
    for t in range(1, n):
        hx[t] = om + al * x[t - 1] ** 2 + be * hx[t - 1]
        hy[t] = om + al * y[t - 1] ** 2 + be * hy[t - 1]
        x[t] = np.sqrt(hx[t]) * e1[t]
        y[t] = np.sqrt(hy[t]) * e2[t]
    return x, y


def block_perm_index(n, b, rng):
    """Permute the ORDER of consecutive length-b blocks; a genuine permutation of 1..n."""
    nb = n // b
    order = rng.permutation(nb)
    idx = np.concatenate([np.arange(k * b, (k + 1) * b) for k in order])
    if nb * b < n:                       # trailing partial block stays in place
        idx = np.concatenate([idx, np.arange(nb * b, n)])
    return idx


def _job(a):
    kind, rep, cfg = a
    x, y = gen_null(kind, rep, cfg)
    K, D, w = cfg["K"], cfg["D"], cfg["w"]
    rng = np.random.default_rng(cfg["base_seed"] + 77 * rep)
    grid, obs = domi_curve(x, y, w, D)

    schemes = [("pair", None)] + [(f"block{b}", b) for b in BLOCKS]
    out = {}
    for name, b in schemes:
        curves = np.empty((K, len(grid)))
        for k in range(K):
            idx = rng.permutation(len(x)) if b is None else block_perm_index(len(x), b, rng)
            curves[k] = domi_curve(x[idx], y[idx], w, D)[1]
        A = np.vstack([obs[None, :], curves])            # symmetric studentization over all K+1 curves (P1 / P5)
        mu, sd = A.mean(0), A.std(0) + 1e-12
        Tobs = np.nanmax((obs - mu) / sd)
        Trep = np.array([np.nanmax((curves[k] - mu) / sd) for k in range(K)])
        pval = (1 + int(np.sum(ge(Trep, Tobs)))) / (K + 1)
        # store the p-value so the rejection rate at any level can be recomputed
        out[name] = {"pval": float(pval), "reject": bool(pval <= cfg["alpha"])}
    return kind, out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=CFG["reps"])
    ap.add_argument("--procs", type=int, default=14)
    a = ap.parse_args()
    cfg = dict(CFG); cfg["reps"] = a.reps
    os.makedirs(OUT, exist_ok=True)

    rows, res = [], {}
    with Pool(a.procs, maxtasksperchild=20) as pool:
        for kind in ["iid", "ar1", "garch"]:
            t0 = time.time()
            got = pool.map(_job, [(kind, r, cfg) for r in range(cfg["reps"])], chunksize=2)
            save_records(OUT, f"records_{kind}.csv",
                         [{f"{nm}_{k}": v for nm, d in g[1].items() for k, v in d.items()}
                          for g in got], {"null": kind, "K": cfg["K"], "D": cfg["D"]})
            line = []
            for name in ["pair"] + [f"block{b}" for b in BLOCKS]:
                fpr = float(np.mean([g[1][name]["reject"] for g in got]))
                rows.append(dict(null=kind, scheme=name, K=cfg["K"], D=cfg["D"], reps=cfg["reps"], fpr=fpr))
                res[f"{kind}|{name}"] = fpr
                line.append(f"{name} {fpr:.3f}")
            print(f"[{time.strftime('%H:%M:%S')}] {kind} ({time.time()-t0:.0f}s): " + "  ".join(line), flush=True)

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    json.dump(dict(config=cfg, blocks=BLOCKS, studentisation="symmetric, K+1 curves", results=res), open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print("block_perm done", flush=True)
