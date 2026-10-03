#!/usr/bin/env python
"""Population DOMI contrast for vector-valued blocks (Supplementary Section B.6).

Supplementary Section B.6 reports that the block form of the construction loses power as the block
dimension d grows, and attributes the loss to the *population* contrast rather than to estimation variance.
This script computes the numbers behind that attribution: the plug-in DOMI of a single
long segment (n = 20,000, so the estimate stands in for the population value) before and after the
break of scenario MV-S2(d), and reports the contrast |I_post - I_pre|:

  * `aligned`  -- MV-S2 exactly as run by run_mv_blocks.py: Y_j depends on X_j only, so one
                  coordinate pair carries the whole coupling. Reported at D = 8 and D = 16.
  * `rotated`  -- the same coupling routed through a fixed orthogonal mixing of the coordinates,
                  Y_j depending on (XQ)_j, so that no single pair carries all of it. Reported
                  against the strongest single coordinate pair, at D = 8.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_mv_contrast.py
Writes 04_DAOU/EXPERIMENT/mv_blocks/population_contrast.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from dots.domi import ranks01, unit_rff  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "mv_blocks")
N, A = 20000, 0.7          # segment length standing in for the population; per-coordinate coupling
DIMS = (1, 2, 3, 5)
FEAT_SEED = 2026           # matches DOMIContext's default in run_mv_blocks.py (Y gets seed + 1)


def _ent(M):
    """von Neumann entropy of a unit-trace moment matrix."""
    lam = np.linalg.eigvalsh(M)
    lam = lam[lam > 1e-15]
    return float(-(lam * np.log(lam)).sum())


def domi(X, Y, D):
    """Plug-in DOMI of one segment, from the block feature moments directly (no prefix caches)."""
    FX = unit_rff(ranks01(X), D, FEAT_SEED)
    FY = unit_rff(ranks01(Y), D, FEAT_SEED + 1)
    J = np.einsum("ti,tj->tij", FX, FY).reshape(len(FX), D * D)
    m = len(FX)
    return _ent(FX.T @ FX / m) + _ent(FY.T @ FY / m) - _ent(J.T @ J / m)


def regimes(d, seed, rotate):
    """The two MV-S2(d) regimes as one long sample each: independent, then coupled."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d))
    E, E2 = rng.standard_normal((N, d)), rng.standard_normal((N, d))
    if rotate and d > 1:                     # fixed orthogonal mixing of the coordinates
        Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
        Z = X @ Q
    else:
        Z = X
    Y_pre = E
    Y_post = np.sqrt(1 - A * A) * E + A * np.abs(Z) * E2
    return X, Y_pre, Y_post


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {"config": {"n": N, "a": A, "dims": list(DIMS), "feature_seed": FEAT_SEED,
                      "note": "contrast = |I(post) - I(pre)| of the block plug-in at n=20,000"},
           "aligned": [], "rotated": []}

    print(f"{'d':>2} {'D':>3} {'I(pre)':>9} {'I(post)':>9} {'contrast':>9}")
    for D in (8, 16):
        for d in DIMS:
            X, Yp, Yq = regimes(d, 40000 + d, rotate=False)
            a_, b_ = domi(X, Yp, D), domi(X, Yq, D)
            out["aligned"].append({"d": d, "D": D, "I_pre": round(a_, 4),
                                   "I_post": round(b_, 4), "contrast": round(abs(b_ - a_), 4)})
            print(f"{d:>2} {D:>3} {a_:9.4f} {b_:9.4f} {abs(b_-a_):9.4f}")

    print(f"\n{'kind':>8} {'d':>2} {'block':>9} {'best pair':>10}")
    # key "rotated" holds the block vs best-pair comparison for both kinds (field "kind")
    for kind, rot in (("aligned", False), ("rotated", True)):
        for d in (2, 3, 5):
            X, Yp, Yq = regimes(d, 40000 + d, rotate=rot)
            blk = abs(domi(X, Yq, 8) - domi(X, Yp, 8))
            pair = max(abs(domi(X[:, [j]], Yq[:, [j]], 8) - domi(X[:, [j]], Yp[:, [j]], 8))
                       for j in range(d))
            out["rotated"].append({"kind": kind, "d": d, "block": round(blk, 4),
                                   "best_pair": round(pair, 4)})
            print(f"{kind:>8} {d:>2} {blk:9.4f} {pair:10.4f}")

    # MV-M1 at the deployed segment length. The blocks are independent in both halves, so the true
    # DOMI is 0 on each side and the plug-in difference the statistic sees is bias alone. Ranks are
    # global, so after a marginal scale change the two halves occupy different parts of the rank
    # distribution -- the mechanism Supplementary Section B.6 gives for the specificity loss. If the
    # difference grows with d, that mechanism has support; m = 300 is the segment length used throughout.
    M, REPS = 300, 200
    print(f"\n{'d':>2} {'|I_pre - I_post| at m=300':>26} {'(sd)':>9}")
    out["m1_bias_response"] = {"m": M, "reps": REPS, "D": 8, "scale": 2.0, "rows": []}
    for d in DIMS:
        diffs = []
        for r in range(REPS):
            rng = np.random.default_rng(90000 + 100 * d + r)
            X = rng.standard_normal((2 * M, d)); Y = rng.standard_normal((2 * M, d))
            X[M:] *= 2.0                                   # marginal scale change, blocks independent
            UX, UY = ranks01(X), ranks01(Y)                # global ranks over the whole window
            FX = unit_rff(UX, 8, FEAT_SEED); FY = unit_rff(UY, 8, FEAT_SEED + 1)
            J = np.einsum("ti,tj->tij", FX, FY).reshape(len(FX), 64)
            def seg(a, b):
                m = b - a
                return (_ent(FX[a:b].T @ FX[a:b] / m) + _ent(FY[a:b].T @ FY[a:b] / m)
                        - _ent(J[a:b].T @ J[a:b] / m))
            diffs.append(abs(seg(0, M) - seg(M, 2 * M)))
        mu, sd = float(np.mean(diffs)), float(np.std(diffs))
        out["m1_bias_response"]["rows"].append({"d": d, "mean_abs_diff": round(mu, 4),
                                                "sd": round(sd, 4)})
        print(f"{d:>2} {mu:26.4f} {sd:9.4f}")

    p = os.path.join(OUT, "population_contrast.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\nwritten:", p)


if __name__ == "__main__":
    main()
