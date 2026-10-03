#!/usr/bin/env python
"""Vector-valued blocks (scenario MV).

The construction of Section 4.1 takes two *blocks* of a multivariate series, not two scalars:
ranks are computed per component and one random Fourier map is drawn per block, so the feature
dimension D is unchanged by the block dimension d. The other experiments use d = 1; this runner
evaluates d = 1, 2, 3 and 5 (Supplementary Section B.6).

Scenario MV-S2(d): X in R^d standard normal; before the break Y = eps, after it

    Y_j = sqrt(1 - a^2) eps_j + a |X_j| eps'_j,     j = 1..d,

with eps, eps' independent standard normal. This is scenario S2 applied coordinatewise, so
corr(X_j, Y_j) = 0 in both regimes and the change is correlation-free. The per-coordinate
coupling a is held fixed as d grows, so each single coordinate pair carries the same signal and
only the aggregation over coordinates differs.

Comparators, all on the same rank inputs:
  * DOMI block            -- one test on the pair of blocks
  * DOMI pairwise/Bonf    -- d separate scalar tests on (X_j, Y_j), each at level 0.05/d
  * HSIC / dCor / Spearman / CopulaCvM block -- the dependence-specific baselines of Section 5
Specificity control MV-M1(d): blocks independent throughout, marginal scale of X changes by s.

Calibration follows Section 5 exactly: every curve is studentized pointwise by the mean and
standard deviation of the first half of the null replicates, the threshold is the 0.95 quantile
of the studentized null maxima of the second half, and a detection counts only if
|tau_hat - tau| <= 30. The pointwise studentization is not optional: the weighted global curve
has a strongly t-dependent null variance, so without it the argmax is pulled to the boundary
and every method scores zero power under the localization requirement.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_mv_blocks.py --procs 28
Writes 04_DAOU/EXPERIMENT/mv_blocks/{results.csv, results.txt, records_results.csv}.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from dots.domi import DOMIContext, domi_stats, DEP_BASELINES  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "mv_blocks")
N, TAU, TOL, W, D = 600, 300, 30, 60, 8
FPR = 0.05
DIMS = (1, 2, 3, 5)
A = 0.7            # per-coordinate coupling, fixed across d
SCALE = 2.0        # marginal scale change of the MV-M1 control


# ---------------------------------------------------------------- data
def sample_mv(d: int, a: float, seed: int, null: bool):
    """MV-S2: correlation-free coupling appears at TAU in every coordinate pair."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d))
    E, E2 = rng.standard_normal((N, d)), rng.standard_normal((N, d))
    Y = E.copy()
    if not null:
        post = slice(TAU, N)
        Y[post] = np.sqrt(1 - a * a) * E[post] + a * np.abs(X[post]) * E2[post]
    return X, Y


def sample_m1(d: int, s: float, seed: int, null: bool):
    """MV-M1: the blocks stay independent; only the marginal scale of X changes."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d))
    if not null:
        X[TAU:] *= s
    return X, rng.standard_normal((N, d))


# ---------------------------------------------------------------- statistics
def curves(X, Y, seed=2026):
    """Every statistic's global difference curve on one pair of blocks."""
    ctx = DOMIContext(X, Y, W, D=D, seed=seed, n_perm=0)
    out = {"DOMI": domi_stats(ctx, "global")["DOMI-diff"]}
    for name, fn in DEP_BASELINES.items():
        out[name] = fn(ctx, "global")
    return out, ctx.grid


def one_replicate(arg):
    """Raw difference curves; the studentization needs the whole null sample, so it is not done here."""
    kind, d, r, null = arg
    seed = 20260830 + 1000 * d + r + (500000 if null else 0)
    X, Y = (sample_mv(d, A, seed, null) if kind == "S2" else sample_m1(d, SCALE, seed, null))
    st, grid = curves(X, Y)
    for j in range(d):                    # the d scalar tests of the Bonferroni comparator
        st[f"pair{j}"] = curves(X[:, [j]], Y[:, [j]])[0]["DOMI"]
    return st, grid


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--procs", type=int, default=28)
    ap.add_argument("--dims", default=",".join(str(d) for d in DIMS))
    ap.add_argument("--kinds", default="S2,M1")
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows, lines = [], []
    rec_rows = []

    def say(m=""):
        print(m, flush=True)
        lines.append(m)

    say(f"MV blocks: n={N} tau={TAU} tol={TOL} D={D} a={A} scale={SCALE} reps={a.reps}")
    say("calibration: pointwise studentisation from null half 1, threshold from null half 2")
    say()

    dims = tuple(int(x) for x in a.dims.split(","))
    # One worker pool per configuration.
    for kind in a.kinds.split(","):
        for d in dims:
            with Pool(a.procs) as pool:
                t0 = time.time()
                nulls = pool.map(one_replicate, [(kind, d, r, True) for r in range(a.reps)], chunksize=4)
                alts = pool.map(one_replicate, [(kind, d, r, False) for r in range(a.reps)], chunksize=4)
                grid = nulls[0][1]
                keys = [k for k in nulls[0][0] if not k.startswith("pair")]
                pairs = [f"pair{j}" for j in range(d)]
                half = a.reps // 2

                mu, sd = {}, {}
                for k in keys + pairs:
                    stack = np.array([r_[0][k] for r_ in nulls[:half]])
                    mu[k], sd[k] = stack.mean(0), stack.std(0)

                def smax(rec, k):
                    v = (rec[0][k] - mu[k]) / np.where(sd[k] > 0, sd[k], 1.0)
                    i = int(np.nanargmax(v))
                    return float(v[i]), int(grid[i])

                thr = {k: float(np.quantile([smax(r_, k)[0] for r_ in nulls[half:]], 1 - FPR))
                       for k in keys}
                # per-coordinate tests share one threshold at the Bonferroni level 0.05/d
                thr_pair = float(np.quantile(
                    np.concatenate([[smax(r_, p)[0] for r_ in nulls[half:]] for p in pairs]),
                    1 - FPR / d))

                def score(recs, k, side=""):
                    hit = far = 0
                    for ri, r_ in enumerate(recs):
                        mx, tau_hat = smax(r_, k)
                        if mx > thr[k]:
                            far += 1
                            hit += abs(tau_hat - TAU) <= TOL
                        if side:
                            rec_rows.append(dict(scen=f"MV-{kind}", d=d, method=k, side=side,
                                                 rep=ri, stat=float(mx), tau_hat=int(tau_hat),
                                                 detect=int(mx > thr[k])))
                    return hit / len(recs), far / len(recs)

                def score_pair(recs, side=""):
                    hit = far = 0
                    for ri, r_ in enumerate(recs):
                        best = max((smax(r_, p) for p in pairs), key=lambda q: q[0])
                        if best[0] > thr_pair:
                            far += 1
                            hit += abs(best[1] - TAU) <= TOL
                        if side:
                            rec_rows.append(dict(scen=f"MV-{kind}", d=d, method="DOMI-pairwise/Bonf",
                                                 side=side, rep=ri, stat=float(best[0]),
                                                 tau_hat=int(best[1]), detect=int(best[0] > thr_pair)))
                    return hit / len(recs), far / len(recs)

                for k in keys:
                    p, _ = score(alts, k, side="alt")
                    _, f = score(nulls[half:], k, side="null")
                    rows.append(dict(scen=f"MV-{kind}", d=d, method=k,
                                     power=round(p, 3), far=round(f, 3)))
                p, _ = score_pair(alts, side="alt")
                _, f = score_pair(nulls[half:], side="null")
                rows.append(dict(scen=f"MV-{kind}", d=d, method="DOMI-pairwise/Bonf",
                                 power=round(p, 3), far=round(f, 3)))
                sel = [r for r in rows if r["scen"] == f"MV-{kind}" and r["d"] == d]
                say(f"MV-{kind} d={d} ({time.time()-t0:.0f}s): "
                    + ", ".join(f"{r['method']}={r['power']:.2f}/{r['far']:.2f}" for r in sel))

    with open(os.path.join(OUT, "records_" + a.out), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["scen", "d", "method", "side", "rep", "stat",
                                           "tau_hat", "detect"])
        wr.writeheader()
        wr.writerows(rec_rows)
    with open(os.path.join(OUT, a.out), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["scen", "d", "method", "power", "far"])
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(OUT, a.out.replace(".csv", ".txt")), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwritten: {OUT}/{a.out}")


if __name__ == "__main__":
    main()
