#!/usr/bin/env python
"""Scenario MB with kernel change-point detection (KCP) and e.divisive added as baselines,
and the two-stage "segment, then re-test by DOMI" variant of each.

Purpose
-------
Section 6.3 / Supplementary Table B.7 compare Holevo partitioning with rank-Gaussian PELT and kernel binary
segmentation on scenario MB. This script adds the two standard nonparametric multiple change-point
methods (seg_baselines.py): KCP (Arlot, Celisse and Harchaoui, 2019) and e.divisive (Matteson and
James, 2014), each on the raw data and on column-wise ranks. It then asks whether the two-stage
design of Section 4.7 needs Holevo partitioning at stage one, by re-testing the candidates of
KCP-on-ranks, e.divisive-on-ranks and Holevo partitioning with the DOMI-difference permutation test.

Design (identical to run_experiment.py part e2)
--------------------------------------------------
  data        dots.synth.generate("P1", a, rep, base_seed=20260825, level_idx) with n = 1800,
              dependence breaks at 450 / 900 / 1350, a in {0.8, 0.9}, 200 replicates per level;
              the no-change null is generate(..., null=True, level_idx=0), 200 replicates;
  grid        candidate change points on multiples of 10;
  KCP         penalty beta per change point, chosen as the smallest value on the grid
              BETAS_KCP at which at most 5% of the 200 null replicates produce any change point
              (the e2 rule), separately for raw and ranks;
  e.divisive  alpha = 1, significance 0.05 by R = 99 within-segment permutations, minimum segment
              size 60 (the MB window w), split points on the same grid;
  metrics     P(K_hat = 3), median Hausdorff distance, mean adjusted Rand index (dots.pelt helpers),
              and the null rate of any detection.
Two-stage re-test: each stage-one candidate is re-tested with the DOMI-difference statistic
(run_domi_binseg.split_test: ranks and D = 8 features recomputed within the window, split points on a
step-10 grid with 60 points at each end, symmetric studentization, permutation p-value) on the window
centered on the candidate whose half-width is the distance to the nearer neighboring candidate or
series end. The series is i.i.d., so the calibration is pair permutation, with K = 99 replicas.
Candidates pass at Benjamini-Hochberg q <= 0.10 within the replicate (the weather rule).
Holevo-partitioning candidates are read from the stored e2 records at the e2 penalty, so that stage
one is exactly the e2 segmentation.

Marginal-only multi-break design (design key "M1", not the single-break scenario M1; last column
of Table B.7): X and Y independent throughout, n = 1800, the scale of X switching 1 -> s -> 1 -> s
at 450 / 900 / 1350 with s = 1.6; 200 replicates, seeds [20260925, rep]. No dependence change
exists, so every detection is a marginal detection and every stage-two pass a false pass. Holevo
partitioning is computed on this design by the e2 recipe (grid 10, three pair-permutation bias
corrections, the e2 beta).

Run
---
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_seg_baselines.py --procs 12

Outputs (04_DAOU/EXPERIMENT/seg_baselines/)
  results.json          calibrated penalties, null rates, the metric table per design and method
  results.csv           the same table, one row per (design, level, method)
  records_null_path.csv.gz  per null replicate: the smallest KCP penalty on the grid with no change
                        point (raw and ranks) -- the calibration regenerates from it
  records_reps.csv.gz   per (design, level, replicate, method): change points, K_hat, Hausdorff,
                        ARI; for two-stage rows the candidates, their DOMI p-values and BH q-values
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.synth import generate  # noqa: E402
from dots.domi import ranks01, unit_rff  # noqa: E402
from dots.encode import MomentCache  # noqa: E402
from dots import pelt as P  # noqa: E402
import seg_baselines as SB  # noqa: E402
import run_domi_binseg as QB  # noqa: E402
from run_e8_summary import step_up_q  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "seg_baselines")
E2 = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e2")
N, TAUS, BASE_SEED, REPS = 1800, [450, 900, 1350], 20260825, 200
GRID = np.arange(0, N + 1, 10)
BETAS_KCP = [round(float(b), 3) for b in np.arange(0.25, 20.0001, 0.25)]
ED = dict(sig=0.05, R=99, min_size=60, step=10)
K_RETEST, Q_RETEST = 99, 0.10
PELT_BETA = "10.285714285714286"          # Holevo-partitioning penalty of e2 (e2 results.csv)
M1_S = 1.6
QB.K = K_RETEST                              # replicas of the DOMI split test


def data(design, li, rep):
    if design == "null":
        Z = generate("P1", 0.8, rep, null=True, base_seed=BASE_SEED, level_idx=0)["Z"]
    elif design == "MB":
        Z = generate("P1", [0.8, 0.9][li], rep, null=False, base_seed=BASE_SEED, level_idx=li)["Z"]
    else:                                   # M1-like multi-break marginal design
        rng = np.random.default_rng([20260925, rep])
        Z = rng.standard_normal((N, 2))
        Z[450:900, 0] *= M1_S
        Z[1350:, 0] *= M1_S
    return Z


def ranks(Z):
    return np.hstack([ranks01(Z[:, [0]]), ranks01(Z[:, [1]])])


def kcp_min_beta(Z):
    """Smallest beta on BETAS_KCP giving no change point (inf if none)."""
    C = SB.kcp_cost_matrix(Z, GRID)
    return next((b for b in BETAS_KCP if not P.pelt_from_costs(C, b)), float("inf"))


def holevo_partition(Z, rep):
    """Holevo partitioning exactly as run_experiment._e2_job (used for the M1-like design only)."""
    n = len(Z)
    FX = unit_rff(ranks01(Z[:, [0]]), 8, 2026)
    FY = unit_rff(ranks01(Z[:, [1]]), 8, 2027)
    J = np.einsum("ti,tj->tij", FX, FY).reshape(n, -1)
    C = P.cost_matrix_vn(MomentCache(J), GRID)
    prng = np.random.default_rng([BASE_SEED, rep, 0, 99, 13])
    Cp = np.zeros_like(C)
    for _ in range(3):
        Cp += P.cost_matrix_vn(MomentCache(J[prng.permutation(n)]), GRID)
    return [int(GRID[i]) for i in P.pelt_from_costs(C - Cp / 3, float(PELT_BETA))]


def retest(Z, cands, rng):
    """DOMI re-test of each candidate on its centered neighbor window; BH within replicate."""
    cands = sorted(cands)
    if not cands:
        return [], [], []
    edges = [0] + cands + [len(Z)]
    ps = []
    for i, c in enumerate(cands):
        h = min(c - edges[i], edges[i + 2] - c)
        p, _ = QB.split_test(Z[c - h:c + h, 0], Z[c - h:c + h, 1], rng)
        ps.append(float(p))
    q = step_up_q(ps)
    return ps, [float(v) for v in q], [c for c, qq in zip(cands, q) if qq <= Q_RETEST]


def score(cps):
    return dict(k=len(cps), exact3=int(len(cps) == 3), hausdorff=P.hausdorff(TAUS, cps, N),
                ari=P.rand_index_adj(TAUS, cps, N))


def _null_path(rep):
    Z = data("null", 0, rep)
    return dict(rep=rep, kcp_raw=kcp_min_beta(Z), kcp_rank=kcp_min_beta(ranks(Z)))


def _job(args):
    design, li, rep, beta_raw, beta_rank, pelt_cps = args
    Z = data(design, li, rep)
    U = ranks(Z)
    rows = []
    stage1 = {}
    for tag, X, beta in (("raw", Z, beta_raw), ("rank", U, beta_rank)):
        cps = SB.kcp_segment(SB.kcp_cost_matrix(X, GRID), GRID, beta)
        stage1[f"KCP-{tag}"] = cps
        cps_e, trail = SB.e_divisive(X, rng=np.random.default_rng([BASE_SEED, 7, li, rep, int(tag == "rank")]),
                                     **ED)
        stage1[f"edivisive-{tag}"] = cps_e
        rows.append(dict(method=f"edivisive-{tag}", cps=cps_e,
                         extra=json.dumps([[int(t), round(q, 4), p] for t, q, p in trail]), **score(cps_e)))
        rows.append(dict(method=f"KCP-{tag}", cps=cps, extra=f"beta={beta}", **score(cps)))
    if pelt_cps is None:
        pelt_cps = holevo_partition(Z, rep)
    stage1["Holevo-partition"] = pelt_cps
    rows.append(dict(method="Holevo-partition", cps=pelt_cps, extra="", **score(pelt_cps)))
    crng = np.random.default_rng([BASE_SEED, 11, li, rep, {"null": 0, "MB": 1, "M1": 2}[design]])
    for m in ("KCP-rank", "edivisive-rank", "Holevo-partition"):
        ps, qs, passed = retest(Z, stage1[m], crng)
        rows.append(dict(method=f"{m}+DOMI", cps=passed,
                         extra=json.dumps(dict(cands=stage1[m], p=ps, q=qs)), **score(passed)))
    for r in rows:
        r.update(design=design, level=("" if design == "null" else [0.8, 0.9][li] if design == "MB"
                                       else M1_S), rep=rep)
        r["cps"] = json.dumps(r["cps"])
    return rows


def pelt_records():
    """Holevo-partitioning change points of e2 per replicate from the e2 records.

    In e2/records_alt.csv.gz the penalty column appears twice (pandas reads the two as PELT_BETA
    and PELT_BETA + ".1"); each row has its value in exactly one of them, and the non-empty one is
    taken."""
    import pandas as pd
    out = {}
    for fn, key in (("records_null.csv.gz", "null"), ("records_alt.csv.gz", "alt")):
        d = pd.read_csv(os.path.join(E2, fn))
        d = d[d["stat"] == "Holevo-partition"]
        for _, r in d.iterrows():
            vals = [r[c] for c in (PELT_BETA, PELT_BETA + ".1") if c in r.index]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            assert len(vals) <= 1, "ambiguous record"
            # an empty list is written as "[]"; a NaN in both columns never occurs for a written row
            cps = json.loads(vals[0]) if vals else []
            lvl = None if key == "null" else float(r["level"])
            out[(key, lvl, int(r["rep"]))] = [int(c) for c in cps]
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with Pool(a.procs, maxtasksperchild=20) as pool:
        path = pool.map(_null_path, range(a.reps), chunksize=4)
        with gzip.open(os.path.join(OUT, "records_null_path.csv.gz"), "wt", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["rep", "kcp_raw", "kcp_rank"])
            w.writeheader(); w.writerows(path)
        beta = {}
        for tag in ("raw", "rank"):
            mb = np.array([r[f"kcp_{tag}"] for r in path])
            b = next((b for b in BETAS_KCP if np.mean(mb > b) <= 0.05), BETAS_KCP[-1])
            beta[tag] = dict(beta=b, null_any=float(np.mean(mb > b)))
        print(f"KCP penalties: {beta}  ({time.time()-t0:.0f}s)", flush=True)
        pr = pelt_records()
        jobs = [("null", 0, r, beta["raw"]["beta"], beta["rank"]["beta"], pr[("null", None, r)])
                for r in range(a.reps)]
        jobs += [("MB", li, r, beta["raw"]["beta"], beta["rank"]["beta"], pr[("alt", lv, r)])
                 for li, lv in enumerate((0.8, 0.9)) for r in range(a.reps)]
        jobs += [("M1", 0, r, beta["raw"]["beta"], beta["rank"]["beta"], None) for r in range(a.reps)]
        rows = [x for part in pool.imap(_job, jobs, chunksize=1) for x in part]
    wall = time.time() - t0
    keys = ["design", "level", "rep", "method", "k", "exact3", "hausdorff", "ari", "cps", "extra"]
    with gzip.open(os.path.join(OUT, "records_reps.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    table = []
    methods = sorted({r["method"] for r in rows})
    for design, level in [("null", ""), ("MB", 0.8), ("MB", 0.9), ("M1", M1_S)]:
        for m in methods:
            rr = [r for r in rows if r["design"] == design and r["level"] == level and r["method"] == m]
            if not rr:
                continue
            ent = dict(design=design, level=level, method=m, reps=len(rr),
                       any_detection=float(np.mean([r["k"] > 0 for r in rr])),
                       mean_k=float(np.mean([r["k"] for r in rr])))
            if design == "MB":
                ent.update(k_correct=float(np.mean([r["exact3"] for r in rr])),
                           hausdorff_median=float(np.median([r["hausdorff"] for r in rr])),
                           ari_mean=float(np.mean([r["ari"] for r in rr])))
            table.append(ent)
    res = dict(config=dict(n=N, taus=TAUS, reps=a.reps, grid_step=10, base_seed=BASE_SEED,
                           kcp=dict(kernel="Gaussian exp(-d^2/median(d^2))", betas=[BETAS_KCP[0], BETAS_KCP[-1], 0.25],
                                    calibration="smallest beta with <=5% null replicates showing any change"),
                           edivisive=dict(alpha=1, **ED),
                           retest=dict(statistic="DOMI difference (run_domi_binseg.split_test)",
                                              calibration="pair permutation", K=K_RETEST, bh_q=Q_RETEST,
                                              window="centred, half-width = distance to nearer neighbour"),
                           holevo_partition=f"stored e2 records at beta={PELT_BETA} (MB, null); recomputed for M1",
                           m1_like=dict(scale=M1_S, breaks=TAUS)),
               kcp_penalty=beta, table=table, wall_clock_s=round(wall, 1))
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        ks = ["design", "level", "method", "reps", "any_detection", "mean_k", "k_correct",
              "hausdorff_median", "ari_mean"]
        w = csv.DictWriter(f, fieldnames=ks)
        w.writeheader(); w.writerows(table)
    for e in table:
        print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in e.items()))
    print(f"wall clock {wall:.0f}s")


if __name__ == "__main__":
    main()
