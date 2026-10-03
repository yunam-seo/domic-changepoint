#!/usr/bin/env python
"""Power of the block-permutation calibration on a SERIALLY DEPENDENT series.

Section 4.3 shows block permutation restores the level under serial dependence. This script
measures the power of the same calibration against a genuine dependence change when the margins
are autocorrelated (Supplementary Section B.17), and the cost of the longer blocks, which is that
the number of exchangeable units falls from n to n/b.

Design (n = 600, break at 300, K = 99, alpha = 0.05):
  margins  : AR(1) with phi = 0.6 throughout, so the pairs are never exchangeable
  H0       : innovations independent throughout (no dependence change)
  H1       : innovations independent before the break, y-innovation = sqrt(1-a^2) e + a|e1| e'
             after it -- an appearance of nonlinear, correlation-free dependence
  schemes  : pair permutation, block b = 20 (30 blocks), block b = 50 (12 blocks)

Writes 04_DAOU/EXPERIMENT/block_power/{results.csv, results.json}
"""
from __future__ import annotations
import csv, json, os, sys, time
from multiprocessing import Pool
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.perm import ge  # noqa: E402
from run_perm_null_check import domi_curve  # noqa: E402
from run_block_perm_check import block_perm_index  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "block_power")
CFG = dict(n=600, tau=300, w=60, D=8, K=99, reps=200, alpha=0.05, phi=0.6, tol=60, base_seed=20260907)
BLOCKS = [20, 50]


def gen(a, rep, cfg, null):
    """AR(1) margins throughout; contemporaneous dependence appears at tau unless null."""
    rng = np.random.default_rng(cfg["base_seed"] + 1000 * rep + int(100 * a) + (7 if null else 0))
    n, tau, phi = cfg["n"], cfg["tau"], cfg["phi"]
    e1 = rng.standard_normal(n)
    e2 = rng.standard_normal(n)
    if not null:                       # after the break, e2 depends on e1 without correlation
        ep = rng.standard_normal(n)
        e2[tau:] = np.sqrt(1 - a * a) * e2[tau:] + a * np.abs(e1[tau:]) * ep[tau:]
    x = np.empty(n); y = np.empty(n)
    x[0], y[0] = e1[0], e2[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e1[t]
        y[t] = phi * y[t - 1] + e2[t]
    return x, y


def _job(args):
    a, rep, null, cfg = args
    x, y = gen(a, rep, cfg, null)
    rng = np.random.default_rng(cfg["base_seed"] + 77 * rep)
    grid, obs = domi_curve(x, y, cfg["w"], cfg["D"])
    out = {}
    for name, b in [("pair", None)] + [(f"block{b}", b) for b in BLOCKS]:
        R = np.empty((cfg["K"], len(grid)))
        for k in range(cfg["K"]):
            idx = rng.permutation(len(x)) if b is None else block_perm_index(len(x), b, rng)
            R[k] = domi_curve(x[idx], y[idx], cfg["w"], cfg["D"])[1]
        A = np.vstack([obs[None, :], R])                 # symmetric studentization (P1 / P5)
        mu, sd = A.mean(0), A.std(0) + 1e-12
        T = [float(np.nanmax((A[i] - mu) / sd)) for i in range(cfg["K"] + 1)]
        p = (1 + sum(ge(t, T[0]) for t in T[1:])) / (cfg["K"] + 1)
        tau_hat = int(grid[int(np.nanargmax((obs - mu) / sd))])
        out[name] = dict(rej=bool(p <= cfg["alpha"]),
                         hit=bool(p <= cfg["alpha"] and abs(tau_hat - cfg["tau"]) <= cfg["tol"]))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=CFG["reps"])
    ap.add_argument("--procs", type=int, default=14)
    a_ = ap.parse_args()
    cfg = dict(CFG); cfg["reps"] = a_.reps
    os.makedirs(OUT, exist_ok=True)
    rows, res, rec_rows = [], {}, []
    with Pool(a_.procs, maxtasksperchild=20) as pool:
        for lab, a, null in [("H0", 0.9, True), ("H1_a0.7", 0.7, False), ("H1_a0.9", 0.9, False)]:
            t0 = time.time()
            got = pool.map(_job, [(a, r, null, cfg) for r in range(cfg["reps"])], chunksize=2)
            for ri, g in enumerate(got):
                for name in ["pair"] + [f"block{b}" for b in BLOCKS]:
                    rec_rows.append(dict(regime=lab, scheme=name, rep=ri,
                                         rej=int(g[name]["rej"]), hit=int(g[name]["hit"])))
            line = []
            for name in ["pair"] + [f"block{b}" for b in BLOCKS]:
                rej = float(np.mean([g[name]["rej"] for g in got]))
                hit = float(np.mean([g[name]["hit"] for g in got]))
                rows.append(dict(regime=lab, scheme=name, reject_rate=rej, power_localised=hit,
                                 K=cfg["K"], reps=cfg["reps"], n_blocks=(cfg["n"] // int(name[5:])) if name != "pair" else cfg["n"]))
                res[f"{lab}|{name}"] = dict(reject_rate=rej, power_localised=hit)
                line.append(f"{name} rej={rej:.3f} loc={hit:.3f}")
            print(f"[{time.strftime('%H:%M:%S')}] {lab} ({time.time()-t0:.0f}s): " + "  ".join(line), flush=True)
    with open(os.path.join(OUT, "records_reps.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rec_rows[0].keys())); wr.writeheader(); wr.writerows(rec_rows)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    json.dump(dict(config=cfg, blocks=BLOCKS, results=res), open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print("block_power done", flush=True)
