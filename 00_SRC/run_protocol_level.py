#!/usr/bin/env python
"""Null level of the ADAPTIVE calibration protocol of Section 4.3.

What
----
Propositions P1 and P5 give the level of a FIXED permutation scheme. The protocol of Section 4.3
chooses the scheme (pair vs block) and the block length b from the same data, so its level is not
covered by either proposition and is measured here. On each null replicate:

  1. run the exchangeability diagnostic (five rank portmanteaux, pair-permutation calibrated at
     K=199, Bonferroni at 0.05; `run_exch_diag.exch_diag`, unchanged);
  2. no rejection -> pair permutation (b=1); any rejection -> block permutation with
     b = ceil(5 * tau_int_hat);
  3. run the single-break DOMI test of Algorithm 1 (D=8, window w=60, K=199 replicas of the
     chosen scheme, studentization with moments over all K+1 curves, p-value (9), alpha=0.05).

For comparison the same replicates are also tested with fixed schemes: pair always, block b=20
always, block b=50 always. The replicas for scheme b are drawn from a generator seeded by
(base_seed, design, rep, b), so when the protocol picks b in {1, 20, 50} its decision is
identical to the corresponding fixed-scheme decision on that replicate (paired comparison).

Null designs (n=600, no change point, constant contemporaneous dependence):
  iid        serially independent, Gaussian copula rho=0.5         (run_block_perm_check.gen_null)
  ar1        AR(1) margins phi=0.6, innovations correlated rho=0.5, i.e. a constant Gaussian-copula
             dependence with AR(1) margins                          (run_block_perm_check.gen_null)
  garch      GARCH(1,1) margins, correlated innovations             (run_block_perm_check.gen_null)
  ar1_indep  AR(1) margins phi=0.6, independent innovations         (run_block_power.gen, H0)
Diagnostic seeds follow run_exch_diag (seed=rep).

The DOMI curve is `run_perm_null_check.domi_curve` evaluated with the entropies batched over the
candidate grid (`curve`).

Run
---
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  python 00_SRC/run_protocol_level.py --design ar1 --reps 0:1000 --procs 12   # shardable, resumable
  python 00_SRC/run_protocol_level.py --aggregate [--cap ar1=200 --cap garch=200 --cap ar1_indep=200]

Reported run: iid replicates 0-719; ar1, garch and ar1_indep 0-199 (--cap ...=200; shards may
hold a few further replicates, which --cap excludes). Every replicate is seed-determined.

Outputs (04_DAOU/EXPERIMENT/protocol_level/)
-------
  shards/<design>_<a>-<b>.jsonl  one line per replicate as computed (resume state)
  records_<design>.csv           per-replicate records: diagnostic p-values and decision,
                                 tau_int_hat, b_hat, p-values of pair / block20 / block50 /
                                 adaptive, and the block length the protocol used
  results.csv, results.json      rejection rates at alpha=0.05 with Monte Carlo SEs, routing
                                 shares, distribution of b_hat
"""
from __future__ import annotations

import argparse
import csv
import glob
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
from dots.perm import ge  # noqa: E402
from run_block_perm_check import gen_null, block_perm_index, CFG as BP_CFG  # noqa: E402
from run_block_power import gen as gen_ar1_indep, CFG as PW_CFG  # noqa: E402
from run_exch_diag import exch_diag  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "protocol_level")
CFG = dict(n=600, w=60, D=8, K=199, alpha=0.05, L=20, diag_K=199, diag_alpha=0.05,
           base_seed=20260925, fixed_blocks=[20, 50])
DESIGNS = ["iid", "ar1", "garch", "ar1_indep"]
DESIGN_ID = {d: i for i, d in enumerate(DESIGNS)}


def gen(design, rep):
    if design == "ar1_indep":
        return gen_ar1_indep(0.9, rep, PW_CFG, True)   # a is unused under the null
    return gen_null(design, rep, BP_CFG)


# ---------------------------------------------------------------- DOMI curve (batched domi_curve)
def _ent_batch(M):
    lam = np.clip(np.linalg.eigvalsh(M), 1e-300, None)
    return -(np.where(lam > 1e-14, lam * np.log(lam), 0.0)).sum(-1)


def curve(X, Y, w, D):
    """Same quantity as run_perm_null_check.domi_curve, entropies batched over the grid."""
    n = len(X)
    FX = unit_rff(ranks01(X[:, None]), D, 2026)
    FY = unit_rff(ranks01(Y[:, None]), D, 2027)
    J = np.einsum("ti,tj->tij", FX, FY).reshape(n, D * D)
    cx = np.zeros((n + 1, D, D)); cy = np.zeros((n + 1, D, D)); cj = np.zeros((n + 1, D * D, D * D))
    np.cumsum(np.einsum("ti,tj->tij", FX, FX), axis=0, out=cx[1:])
    np.cumsum(np.einsum("ti,tj->tij", FY, FY), axis=0, out=cy[1:])
    np.cumsum(np.einsum("ti,tj->tij", J, J), axis=0, out=cj[1:])
    t = np.arange(w, n - w + 1)
    mL = t[:, None, None].astype(float)
    mR = (n - t)[:, None, None].astype(float)
    IL = _ent_batch(cx[t] / mL) + _ent_batch(cy[t] / mL) - _ent_batch(cj[t] / mL)
    IR = (_ent_batch((cx[n] - cx[t]) / mR) + _ent_batch((cy[n] - cy[t]) / mR)
          - _ent_batch((cj[n] - cj[t]) / mR))
    return t, np.sqrt(t * (n - t) / n) * np.abs(IL - IR)


# ---------------------------------------------------------------- Algorithm 1
def domi_test(x, y, b, rng, cfg, obs):
    """Single-break test, K replicas permuting pairs jointly in time blocks of length b (b=1: pair)."""
    n, K = len(x), cfg["K"]
    R = np.empty((K, len(obs)))
    for k in range(K):
        idx = rng.permutation(n) if b == 1 else block_perm_index(n, b, rng)
        R[k] = curve(x[idx], y[idx], cfg["w"], cfg["D"])[1]
    A = np.vstack([obs[None, :], R])                  # moments over all K+1 curves
    mu, sd = A.mean(0), A.std(0) + 1e-12
    T = np.nanmax((A - mu) / sd, axis=1)
    return float((1 + int(np.sum(ge(T[1:], T[0])))) / (K + 1))


def _job(args):
    design, rep, cfg = args
    t0 = time.time()
    x, y = gen(design, rep)
    d = exch_diag(x, y, L=cfg["L"], alpha=cfg["diag_alpha"], K=cfg["diag_K"], seed=rep)
    b_ad = d["b_hat"] if d["decision"] == "block" else 1
    obs = curve(x, y, cfg["w"], cfg["D"])[1]
    pv = {}
    for b in sorted({1, *cfg["fixed_blocks"], b_ad}):
        rng = np.random.default_rng([cfg["base_seed"], DESIGN_ID[design], rep, b])
        pv[b] = domi_test(x, y, b, rng, cfg, obs)
    rec = dict(design=design, rep=rep, decision=d["decision"],
               tau_int=d["tau_hat"], b_hat=d["b_hat"], b_used=b_ad,
               n_blocks_used=(len(x) // b_ad), **{f"diagp_{k}": v for k, v in d["pvalues"].items()},
               p_pair=pv[1], p_block20=pv[20], p_block50=pv[50], p_adaptive=pv[b_ad],
               secs=round(time.time() - t0, 1))
    return rec


# ---------------------------------------------------------------- run / aggregate
def _done(design):
    reps = set()
    for f in glob.glob(os.path.join(OUT, "shards", "*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r["design"] == design:        # match on the record's field: shard names of ar1 and ar1_indep share a prefix
                    reps.add(r["rep"])
    return reps


def run(design, a, b, procs):
    os.makedirs(os.path.join(OUT, "shards"), exist_ok=True)
    todo = [r for r in range(a, b) if r not in _done(design)]
    path = os.path.join(OUT, "shards", f"{design}_{a}-{b}.jsonl")
    print(f"[{time.strftime('%H:%M:%S')}] {design} reps {a}:{b}, {len(todo)} to do, {procs} procs", flush=True)
    with Pool(procs, maxtasksperchild=20) as pool, open(path, "a") as f:
        for i, rec in enumerate(pool.imap_unordered(_job, [(design, r, CFG) for r in todo])):
            f.write(json.dumps(rec) + "\n"); f.flush()
            if (i + 1) % 25 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {design} {i+1}/{len(todo)}", flush=True)


def _rate(rej):
    p = float(np.mean(rej)); return p, float(np.sqrt(p * (1 - p) / len(rej)))


def aggregate(caps=None):
    """caps: {design: n} uses replicates 0..n-1 only (a design stopped early is reported on a prefix)."""
    caps = caps or {}
    rows, res = [], {"config": CFG, "designs": {}}
    for design in DESIGNS:
        recs = {}
        for f in glob.glob(os.path.join(OUT, "shards", "*.jsonl")):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    if r["design"] == design:
                        recs[r["rep"]] = r
        if not recs:
            continue
        recs = [recs[k] for k in sorted(recs) if k < caps.get(design, 10**9)]
        with open(os.path.join(OUT, f"records_{design}.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(recs[0].keys())); wr.writeheader(); wr.writerows(recs)
        al = CFG["alpha"]
        blk = [r for r in recs if r["decision"] == "block"]
        bh = np.array([r["b_hat"] for r in blk]) if blk else np.array([])
        row = dict(design=design, reps=len(recs))
        for s in ["adaptive", "pair", "block20", "block50"]:
            row[f"rej_{s}"], row[f"se_{s}"] = _rate([r[f"p_{s}"] <= al for r in recs])
        # adaptive rate within each branch
        for br, sub in [("pairbranch", [r for r in recs if r["decision"] == "pair"]), ("blockbranch", blk)]:
            row[f"n_{br}"] = len(sub)
            row[f"rej_adaptive_{br}"] = float(np.mean([r["p_adaptive"] <= al for r in sub])) if sub else ""
        row["share_block"], row["se_share_block"] = _rate([r["decision"] == "block" for r in recs])
        for q, nm in [(0, "min"), (10, "p10"), (25, "q1"), (50, "median"), (75, "q3"), (90, "p90"), (100, "max")]:
            row[f"bhat_{nm}"] = float(np.percentile(bh, q)) if len(bh) else ""
        for lo, hi in [(1, 19), (20, 29), (30, 49), (50, 99), (100, 10**9)]:
            row[f"bhat_{lo}_{hi if hi < 10**9 else 'up'}"] = int(np.sum((bh >= lo) & (bh <= hi))) if len(bh) else 0
        row["mean_secs_per_rep"] = float(np.mean([r["secs"] for r in recs]))
        rows.append(row)
        res["designs"][design] = dict(row, bhat_counts={int(k): int(v) for k, v in
                                                         zip(*np.unique(bh, return_counts=True))} if len(bh) else {})
        print(f"{design:9s} n={len(recs)}  adaptive {row['rej_adaptive']:.3f}({row['se_adaptive']:.3f})  "
              f"pair {row['rej_pair']:.3f}  b20 {row['rej_block20']:.3f}  b50 {row['rej_block50']:.3f}  "
              f"block share {row['share_block']:.3f}  median b {row['bhat_median']}")
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, restval=""); wr.writeheader(); wr.writerows(rows)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=DESIGNS)
    ap.add_argument("--reps", default="0:1000")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--cap", action="append", default=[], metavar="DESIGN=N",
                    help="aggregate only replicates 0..N-1 of DESIGN (repeatable)")
    a = ap.parse_args()
    if a.design:
        lo, hi = map(int, a.reps.split(":"))
        run(a.design, lo, hi, a.procs)
    if a.aggregate:
        aggregate({c.split("=")[0]: int(c.split("=")[1]) for c in a.cap})
