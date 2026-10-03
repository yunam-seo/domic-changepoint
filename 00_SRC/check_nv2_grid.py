#!/usr/bin/env python
"""Grid-refinement check for [NV-2] (Supplementary Section A.8(iii) and the remark on the
deployed map that follows it).

For the Frank copula (theta = 5) and D = 4 random frequencies per block, computes the largest
entry of M_XY - M_X (x) M_Y over five independent frequency draws, for
  - the deployed map  unit-norm random Fourier features (quoted value 1.5e-2), and
  - the deployed map with its bandwidth held fixed at the population value of the median
    heuristic for uniform inputs, gamma = 1/(1 - 1/sqrt(2))^2 (the deployed map sets gamma by
    the median heuristic on a subsample of its inputs, so on a quadrature grid the bandwidth
    itself moves slightly with the number of grid points),
together with the independence control (copula density = 1). The integrals are evaluated by the midpoint rule on
grids of 300, 400, 600 and 1200 points per axis; Supplementary Section A.8 states the stability of
these values under this refinement. The frequency draws use seeds 1000 + 2s and 1001 + 2s,
s = 0..4.

Run:    python 00_SRC/check_nv2_grid.py
Writes: 04_DAOU/EXPERIMENT/theory/nv2_grid.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.extras import rff  # noqa: E402
from dots.domi import unit_rff  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "theory", "nv2_grid.json")
D, THETA, N_DRAWS = 4, 5.0, 5
GRIDS = (300, 400, 600, 1200)


def frank_density(u, th):
    U, V = np.meshgrid(u, u, indexing="ij")
    e = np.exp(-th)
    return (th * (1 - e) * np.exp(-th * (U + V))
            / ((1 - e) - (1 - np.exp(-th * U)) * (1 - np.exp(-th * V))) ** 2)


def deployed_map(u, D, seed):
    return unit_rff(u, D, seed)


GAMMA_POP = 1.0 / (1.0 - 1.0 / np.sqrt(2.0)) ** 2


def deployed_map_fixed_gamma(u, D, seed):
    F = rff(np.asarray(u, float)[:, None], D=D, seed=seed, gamma=GAMMA_POP)
    return F / np.linalg.norm(F, axis=1, keepdims=True)


def max_mismatch(PX, PY, c, du):
    """max |M_XY - M_X (x) M_Y| for the copula density c on the grid (midpoint rule)."""
    KX = np.einsum("ua,uc->uac", PX, PX)
    KY = np.einsum("vb,vd->vbd", PY, PY)
    MX, MY = KX.sum(0) * du, KY.sum(0) * du
    Tm = np.einsum("uv,uac->vac", c, KX) * du
    d = PX.shape[1]
    MXY = np.einsum("vac,vbd->abcd", Tm, KY).reshape(d * d, d * d) * du
    return float(np.abs(MXY - np.kron(MX, MY)).max())


LABELS = (("deployed", deployed_map),
          ("deployed_fixed_gamma", deployed_map_fixed_gamma))


def main():
    out = {"setting": {"copula": "Frank", "theta": THETA, "D": D, "n_draws": N_DRAWS,
                       "feature_seeds": [[1000 + 2 * s, 1001 + 2 * s] for s in range(N_DRAWS)],
                       "quadrature": "midpoint rule, n points per axis",
                       "fixed_gamma": GAMMA_POP},
           "quoted": {"deployed_min": 1.5e-2},
           "grids": {}}
    for n in GRIDS:
        u = (np.arange(n) + 0.5) / n
        du = 1.0 / n
        c = frank_density(u, THETA)
        row = {}
        for lab, mk in LABELS:
            per_draw = [max_mismatch(mk(u, D, 1000 + 2 * s), mk(u, D, 1001 + 2 * s), c, du)
                        for s in range(N_DRAWS)]
            ind = max_mismatch(mk(u, D, 1000), mk(u, D, 1001), np.ones((n, n)), du)
            row[lab] = {"per_draw": per_draw, "min": min(per_draw), "max": max(per_draw),
                        "independence_control": ind}
        out["grids"][str(n)] = row
        print(f"n={n:5d}: deployed min {row['deployed']['min']:.6e} max {row['deployed']['max']:.6e} | "
              f"fixed-gamma min {row['deployed_fixed_gamma']['min']:.6e} | "
              f"indep ctrl {row['deployed']['independence_control']:.1e}")
    for lab, _ in LABELS:
        vals = np.array([out["grids"][str(n)][lab]["per_draw"] for n in GRIDS])
        out[f"{lab}_max_relative_change_300_to_1200"] = float(
            np.max(np.abs(vals[-1] - vals[0]) / np.abs(vals[-1])))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
