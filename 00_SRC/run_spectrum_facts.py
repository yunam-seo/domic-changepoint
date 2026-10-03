#!/usr/bin/env python
"""Spectral facts about the deployed density operators quoted in the Supplement and Section 4.

(a) Supplementary Section A.9, "Hypotheses and scope": at D = 8, the number of the 64
    eigenvalues of the joint moment rho_XY that lie below 1e-12, for an independent pair of
    standard-normal series (rank inputs, unit-norm random Fourier features), at segment
    lengths 60, 150, 300, 600, 1200, 2000 and five random-Fourier draws (feature seeds
    s and s + 1 for X and Y). One primary table (data seed 2026) plus 20 independent data
    replicates per cell, to show how much of the variation is due to the data draw.
(b) Supplementary Sections A.9 and A.10(iv): smallest eigenvalue of the population marginal
    moments M_X = E[phi(U) phi(U)^T], U ~ Uniform(0,1), for the deployed feature draws
    (seeds 2026 for X, 2027 for Y, D = 8), and of the joint moment at independence,
    M_X (x) M_Y, whose smallest eigenvalue is the product. The population integral is
    evaluated by the midpoint rule on 200,000 points. The same quantities for the other four
    draws of (a) are reported for comparison.
(c) Supplementary Section A.10 remark and Section 4 (discussion of property (i)): von
    Neumann entropy (nats) of the marginal moment rho_X for the rank embedding, series
    length N = 20,000, feature seed 2026 (X) and 2027 (Y), D = 4, 8, 16, 32, 64. Primary data
    seed 31, plus 20 further data seeds for the spread.

Run:    python 00_SRC/run_spectrum_facts.py
Writes: 04_DAOU/EXPERIMENT/spectrum_facts/results.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.qdiv import vn_entropy  # noqa: E402
from dots.domi import ranks01, unit_rff  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "spectrum_facts", "results.json")

# (a)
D_A = 8
THRESH = 1e-12
LENGTHS = (60, 150, 300, 600, 1200, 2000)
FEATURE_SEEDS = (2026, 7, 101, 555, 31337)
N_REP_A = 20
# (b)
N_QUAD = 200_000
# (c)
N_C = 20_000
DS_C = (4, 8, 16, 32, 64)
N_REP_C = 20


def joint_moment(x, y, D, seed):
    FX = unit_rff(ranks01(x[:, None]), D, seed)
    FY = unit_rff(ranks01(y[:, None]), D, seed + 1)
    P = np.einsum("ti,tj->tij", FX, FY).reshape(len(x), D * D)
    return P.T @ P / len(x)


def floor_count(x, y, seed):
    lam = np.linalg.eigvalsh(joint_moment(x, y, D_A, seed))
    return int((lam < THRESH).sum())


def part_a():
    rng = np.random.default_rng(2026)
    primary = {}
    for s in FEATURE_SEEDS:
        primary[str(s)] = [floor_count(rng.standard_normal(m), rng.standard_normal(m), s)
                           for m in LENGTHS]
    reps = {}
    for s in FEATURE_SEEDS:
        reps[str(s)] = {}
        for m in LENGTHS:
            vals = []
            for r in range(N_REP_A):
                rr = np.random.default_rng((r, s, m))
                vals.append(floor_count(rr.standard_normal(m), rr.standard_normal(m), s))
            reps[str(s)][str(m)] = vals
    allp = [c for v in primary.values() for c in v]
    allr = [c for v in reps.values() for w in v.values() for c in w]
    print("(a) primary counts:", primary, "range", [min(allp), max(allp)])
    print("    replicate range", [min(allr), max(allr)])
    return {"D": D_A, "n_eigen": D_A * D_A, "threshold": THRESH,
            "segment_lengths": list(LENGTHS), "feature_seeds": list(FEATURE_SEEDS),
            "data_seed_primary": 2026, "counts": primary, "range": [min(allp), max(allp)],
            "replicates": {"n_rep": N_REP_A, "data_seed": "numpy SeedSequence (rep, feature_seed, length)",
                           "counts": reps, "range": [min(allr), max(allr)]}}


def part_b():
    u = ((np.arange(N_QUAD) + 0.5) / N_QUAD)[:, None]
    out = {"D": D_A, "quadrature_points": N_QUAD, "draws": {}}
    for s in FEATURE_SEEDS:
        lx = float(np.linalg.eigvalsh(unit_rff(u, D_A, s).T @ unit_rff(u, D_A, s) / N_QUAD).min())
        FY = unit_rff(u, D_A, s + 1)
        ly = float(np.linalg.eigvalsh(FY.T @ FY / N_QUAD).min())
        out["draws"][str(s)] = {"lambda_min_MX": lx, "lambda_min_MY": ly,
                                "lambda_min_joint_at_independence": lx * ly}
        print(f"(b) seeds {s}/{s + 1}: lmin M_X {lx:.2e}, M_Y {ly:.2e}, joint {lx * ly:.2e}")
    out["deployed_draw"] = out["draws"]["2026"]
    return out


def marginal_entropies(x):
    U = ranks01(x[:, None])
    res = {}
    for D in DS_C:
        FX = unit_rff(U, D, 2026)
        FY = unit_rff(U, D, 2027)
        res[str(D)] = {"S_X": vn_entropy(FX.T @ FX / len(x)), "S_Y": vn_entropy(FY.T @ FY / len(x)),
                       "log_D": float(np.log(D))}
    return res


def part_c():
    primary = marginal_entropies(np.random.default_rng(31).standard_normal(N_C))
    reps = [marginal_entropies(np.random.default_rng((31, r)).standard_normal(N_C))
            for r in range(N_REP_C)]
    spread = {}
    for D in DS_C:
        for k in ("S_X", "S_Y"):
            v = [rep[str(D)][k] for rep in reps]
            spread.setdefault(str(D), {})[k] = {"min": min(v), "max": max(v), "values": v}
    for D in DS_C:
        p = primary[str(D)]
        print(f"(c) D={D:3d}: S_X {p['S_X']:.4f}  S_Y {p['S_Y']:.4f}  (replicates S_X "
              f"{spread[str(D)]['S_X']['min']:.3f}-{spread[str(D)]['S_X']['max']:.3f})")
    return {"N": N_C, "feature_seeds": {"X": 2026, "Y": 2027}, "data_seed_primary": 31,
            "entropy_unit": "nats", "primary": primary,
            "replicates": {"n_rep": N_REP_C, "data_seed": "numpy SeedSequence (31, rep)",
                           "by_D": spread}}


def main():
    out = {"a_eigenvalue_floor": part_a(), "b_lambda_min": part_b(),
           "c_marginal_entropy": part_c()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
