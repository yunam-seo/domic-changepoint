#!/usr/bin/env python
"""A marginal-invariant segmentation: recursive binary segmentation driven by the DOMI DIFFERENCE
statistic, in place of Holevo partitioning on the joint-entropy cost (Section 6.3).

Motivation. The segmentation stage of the applications is Holevo partitioning, whose cost
C(s) = E_s S(rho_XY,s) is a functional of the JOINT operator. `run_pelt_specificity.py` measures that this
cost fires with probability 1.00 on a pure marginal scale change with no dependence change at all,
the behavior Supplementary Table B.6 records for the joint-state statistic. The DOMI *difference*
statistic does not have that property (Table 1, row M1), because it is a contrast of
marginal-invariant quantities. This script asks whether a segmentation built on it keeps the
specificity while retaining usable multi-break recovery.

Procedure. On a segment, compute the studentized DOMI-difference curve from K joint pair
permutations, take the permutation p-value (the rule P1 covers exactly), and split at the argmax if
p <= alpha; recurse on both parts. Stop at p > alpha or when a part is shorter than min_len.

Two comparisons, each against the Holevo-partitioning number obtained under the same protocol:
  SPECIFICITY  X and Y independent throughout, only the marginal scale of X changing at t=300.
               Holevo partitioning: 0.45 / 1.00 / 1.00 at s = 1.3 / 1.6 / 2.0 (run_pelt_specificity.py).
  RECOVERY     scenario MB (n=1800, dependence breaks at 450/900/1350).
               Holevo partitioning: exactly K=3 in 47.5% of replicates (e2).

Writes 04_DAOU/EXPERIMENT/domi_binseg/results.json
"""
from __future__ import annotations
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.synth import generate            # noqa: E402
from dots.domi import ranks01, unit_rff     # noqa: E402
from dots.perm import ge  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "domi_binseg")
D, W, STEP, K, ALPHA, MINLEN = 8, 60, 10, 49, 0.05, 200


def ent(M):
    l = np.clip(np.linalg.eigvalsh(M), 1e-300, None); l = l[l > 1e-14]
    return float(-(l * np.log(l)).sum())


def domi_curve(x, y, sx=2026, sy=2027):
    """Weighted DOMI-difference curve on a coarse candidate grid. Ranks are recomputed within the
    segment, which is what makes the statistic invariant to the segment's own margins."""
    n = len(x)
    if n < 2 * W + STEP:
        return None, None
    FX = unit_rff(ranks01(np.asarray(x, float)[:, None]), D, sx)
    FY = unit_rff(ranks01(np.asarray(y, float)[:, None]), D, sy)
    J = np.einsum("ti,tj->tij", FX, FY).reshape(n, D * D)
    cx = np.zeros((n + 1, D, D)); cy = np.zeros((n + 1, D, D)); cj = np.zeros((n + 1, D * D, D * D))
    np.cumsum(np.einsum("ti,tj->tij", FX, FX), axis=0, out=cx[1:])
    np.cumsum(np.einsum("ti,tj->tij", FY, FY), axis=0, out=cy[1:])
    np.cumsum(np.einsum("ti,tj->tij", J, J), axis=0, out=cj[1:])

    def I(a, b):
        m = b - a
        return ent((cx[b] - cx[a]) / m) + ent((cy[b] - cy[a]) / m) - ent((cj[b] - cj[a]) / m)

    grid = np.arange(W, n - W + 1, STEP)
    return grid, np.array([np.sqrt(t * (n - t) / n) * abs(I(0, t) - I(t, n)) for t in grid])


def split_test(x, y, rng):
    """Permutation p-value and split point for one segment; symmetric studentization (P1)."""
    grid, obs = domi_curve(x, y)
    if grid is None:
        return 1.0, None
    R = np.empty((K, len(grid)))
    for k in range(K):
        idx = rng.permutation(len(x))
        R[k] = domi_curve(x[idx], y[idx])[1]
    A = np.vstack([obs[None, :], R])
    mu, sd = A.mean(0), A.std(0) + 1e-12
    T = [float(np.nanmax((A[i] - mu) / sd)) for i in range(K + 1)]
    p = (1 + sum(ge(t, T[0]) for t in T[1:])) / (K + 1)
    return p, int(grid[int(np.nanargmax((obs - mu) / sd))])


def binseg(x, y, rng, offset=0, depth=0):
    if len(x) < MINLEN or depth > 4:
        return []
    p, tau = split_test(x, y, rng)
    if p > ALPHA or tau is None:
        return []
    return (binseg(x[:tau], y[:tau], rng, offset, depth + 1)
            + [offset + tau]
            + binseg(x[tau:], y[tau:], rng, offset + tau, depth + 1))


# ---------------------------------------------------------------- specificity
def _spec(args):
    rep, s = args
    rng = np.random.default_rng(30000 + rep)
    x = rng.standard_normal(600); y = rng.standard_normal(600); x[300:] *= s   # X,Y independent
    cps = binseg(x, y, np.random.default_rng(80000 + rep))
    return dict(any=len(cps) > 0, near=any(abs(c - 300) <= 30 for c in cps), k=len(cps))


# ---------------------------------------------------------------- multi-break recovery
def _mb(args):
    rep, null = args
    smp = generate("P1", 0.9, rep, null=null, base_seed=20260913, level_idx=1)
    Z = smp["Z"]
    cps = binseg(Z[:, 0], Z[:, 1], np.random.default_rng(90000 + rep))
    taus = [450, 900, 1350]
    return dict(k=len(cps), exact3=len(cps) == 3,
                hits=sum(any(abs(c - t) <= 60 for c in cps) for t in taus), cps=cps)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = {}
    rec_rows = []
    with Pool(a.procs, maxtasksperchild=8) as pool:
        print("SPECIFICITY (X,Y independent throughout; only the X margin changes at t=300):", flush=True)
        spec = {}
        for s in (1.0, 1.3, 1.6, 2.0):
            t0 = time.time()
            r = pool.map(_spec, [(i, s) for i in range(a.reps)], chunksize=1)
            rec_rows += [dict(part="spec", level=s, rep=i, any=int(d["any"]), near=int(d["near"]),
                              k=d["k"]) for i, d in enumerate(r)]
            spec[str(s)] = dict(any=round(float(np.mean([d["any"] for d in r])), 3),
                                near=round(float(np.mean([d["near"] for d in r])), 3),
                                mean_k=round(float(np.mean([d["k"] for d in r])), 2))
            print(f"  s={s}: any-break={spec[str(s)]['any']:.3f}  at-the-change={spec[str(s)]['near']:.3f} "
                  f" mean k={spec[str(s)]['mean_k']:.2f}   ({time.time()-t0:.0f}s)", flush=True)
        res["specificity"] = spec
        print("MULTI-BREAK RECOVERY (scenario MB, breaks at 450/900/1350):", flush=True)
        t0 = time.time()
        nulls = pool.map(_mb, [(i, True) for i in range(a.reps)], chunksize=1)
        alts = pool.map(_mb, [(i, False) for i in range(a.reps)], chunksize=1)
        rec_rows += [dict(part="mb_null", level="", rep=i, any=int(d["k"] > 0), near="", k=d["k"])
                     for i, d in enumerate(nulls)]
        rec_rows += [dict(part="mb_alt", level="", rep=i, any=int(d["exact3"]), near=d["hits"],
                          k=d["k"]) for i, d in enumerate(alts)]
        res["multibreak"] = dict(
            null_any_break=round(float(np.mean([d["k"] > 0 for d in nulls])), 3),
            exactly_three=round(float(np.mean([d["exact3"] for d in alts])), 3),
            mean_k=round(float(np.mean([d["k"] for d in alts])), 2),
            mean_breaks_found=round(float(np.mean([d["hits"] for d in alts])), 2))
        print(f"  null any-break={res['multibreak']['null_any_break']:.3f} | exactly K=3: "
              f"{res['multibreak']['exactly_three']:.3f} | mean k={res['multibreak']['mean_k']:.2f} | "
              f"true breaks found (of 3)={res['multibreak']['mean_breaks_found']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    res["config"] = dict(D=D, w=W, grid_step=STEP, K=K, alpha=ALPHA, min_len=MINLEN, reps=a.reps)
    res["holevo_partition_reference"] = dict(specificity="0.45/1.00/1.00 at s=1.3/1.6/2.0 (pelt_specificity)",
                                        exactly_three="0.475 (e2)")
    import csv as _csv
    with open(os.path.join(OUT, "records_reps.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["part", "level", "rep", "any", "near", "k"])
        w.writeheader(); w.writerows(rec_rows)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print("domi_binseg done", flush=True)
