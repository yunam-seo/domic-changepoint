"""Segmentation and single-break tests on a bivariate series, used by the financial analyses
(run_e6_finance.py, run_e6_monthly.py, run_e6a_block.py; Section 6.8).

pelt_segment    Holevo partitioning with a pair-permutation bias-corrected cost; for each penalty the
                fraction of pair-permuted copies that return any change point (penalty calibration)
gauss_segment   the same with the rank-Gaussian cost (rank-Gaussian PELT)
single_break_test  the most significant single break by permutation-studentized statistics
"""
from __future__ import annotations

import numpy as np

from .domi import DOMIContext, domi_stats, DEP_BASELINES, ranks01, unit_rff
from .encode import MomentCache
from . import pelt as P
from .perm import ge

D = 8
SEED = 20260826


def joint_features(x, y, D, seed=SEED):
    FX = unit_rff(ranks01(x[:, None]), D, seed)
    FY = unit_rff(ranks01(y[:, None]), D, seed + 1)
    return np.einsum("ti,tj->tij", FX, FY).reshape(len(x), -1)


def bias_corrected_cost(J, grid, rng, n_perm=5):
    C = P.cost_matrix_vn(MomentCache(J), grid)
    Cp = np.zeros_like(C)
    for _ in range(n_perm):
        Cp += P.cost_matrix_vn(MomentCache(J[rng.permutation(len(J))]), grid)
    return C - Cp / n_perm


# Per-permutation log for the record files of every caller (weekly, monthly and block analyses).
# Callers set PERM_TAG before a call; helpers append one row per permuted copy / replica.
PERM_TAG = ""
PERM_LOG = []


def flush_perm_log(path):
    import csv as _csv
    if not PERM_LOG:
        return
    keys = ["tag", "kind", "label", "copy", "value"]
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in PERM_LOG])
    PERM_LOG.clear()


def pelt_segment(x, y, step, betas, n_beta_perm=20, rng=None):
    """Returns dict beta -> CPs for observed, and per-beta fraction of permuted copies with any CP."""
    rng = rng or np.random.default_rng(SEED)
    n = len(x)
    grid = np.arange(0, n + 1, step)
    J = joint_features(x, y, D)
    Cobs = bias_corrected_cost(J, grid, rng)
    obs = {b: [int(grid[i]) for i in P.pelt_from_costs(Cobs, b)] for b in betas}
    fa = {b: 0 for b in betas}
    for k in range(n_beta_perm):
        idx = rng.permutation(n)
        Jp = joint_features(x[idx], y[idx], D)
        Cp = bias_corrected_cost(Jp, grid, rng)
        ks = {b: len(P.pelt_from_costs(Cp, b)) for b in betas}
        for b in betas:
            fa[b] += ks[b] > 0
        PERM_LOG.append(dict(tag=PERM_TAG, kind="pelt_beta", label="", copy=k,
                             value=next((b for b in betas if ks[b] == 0), float("inf"))))
    fa = {b: fa[b] / n_beta_perm for b in betas}
    return obs, fa


def gauss_segment(x, y, step, betas, n_beta_perm=20, rng=None):
    rng = rng or np.random.default_rng(SEED + 1)
    n = len(x)
    grid = np.arange(0, n + 1, step)
    U = np.hstack([ranks01(x[:, None]), ranks01(y[:, None])])

    def bc_cost(Ua):
        C = P.cost_matrix_gauss2(MomentCache(Ua), grid)
        Cp = np.zeros_like(C)
        for _ in range(5):
            Cp += P.cost_matrix_gauss2(MomentCache(Ua[rng.permutation(len(Ua))]), grid)
        return C - Cp / 5

    Cobs = bc_cost(U)
    obs = {b: [int(grid[i]) for i in P.pelt_from_costs(Cobs, b)] for b in betas}
    fa = {b: 0 for b in betas}
    for k in range(n_beta_perm):
        Cp = bc_cost(U[rng.permutation(n)])
        ks = {b: len(P.pelt_from_costs(Cp, b)) for b in betas}
        for b in betas:
            fa[b] += ks[b] > 0
        PERM_LOG.append(dict(tag=PERM_TAG, kind="gauss_beta", label="", copy=k,
                             value=next((b for b in betas if ks[b] == 0), float("inf"))))
    return obs, {b: fa[b] / n_beta_perm for b in betas}


def _perm_index(n, rng, block=None):
    """i.i.d. pair permutation (block=None) or permutation of the ORDER of consecutive blocks."""
    if block is None:
        return rng.permutation(n)
    nb = n // block
    order = rng.permutation(nb)
    idx = np.concatenate([np.arange(k * block, (k + 1) * block) for k in order])
    if nb * block < n:
        idx = np.concatenate([idx, np.arange(nb * block, n)])
    return idx


def single_break_test(x, y, w, K=99, seed=SEED, block=None):
    ctx = DOMIContext(x[:, None], y[:, None], w, D=D, seed=2026, n_perm=0)

    def curves(ctxa):
        g = domi_stats(ctxa, "global")
        return {"DOMI-diff": g["DOMI-diff"], "HSIC-diff": DEP_BASELINES["HSIC-diff"](ctxa, "global"),
                "Spearman-diff": DEP_BASELINES["Spearman-diff"](ctxa, "global"),
                "CopulaCvM": DEP_BASELINES["CopulaCvM"](ctxa, "global")}

    obs = curves(ctx)
    rng = np.random.default_rng(seed)
    reps = {m: [] for m in obs}
    for k in range(K):
        idx = _perm_index(len(x), rng, block)
        ctxp = DOMIContext(x[idx][:, None], y[idx][:, None], w, D=D, seed=2026, n_perm=0)
        for m, v in curves(ctxp).items():
            reps[m].append(v)
    res = {}
    for m in obs:
        # Symmetric studentization: per-t moments from ALL K+1 curves, which is the variant
        # Proposition P1 covers exactly (leave-one-out moments are not covered verbatim).
        R = np.array(reps[m])
        A = np.vstack([obs[m][None, :], R])
        mu, sd = A.mean(0), A.std(0) + 1e-12
        Tall = [float(np.max((A[i] - mu) / sd)) for i in range(K + 1)]
        T = (obs[m] - mu) / sd
        mx = Tall[0]
        pval = (1 + sum(ge(t, mx) for t in Tall[1:])) / (K + 1)
        PERM_LOG.append(dict(tag=PERM_TAG, kind="single_break", label=f"{m}|block={block}",
                             copy=-1, value=" ".join(f"{v:.6f}" for v in Tall)))
        res[m] = dict(tau_hat=int(ctx.grid[int(np.argmax(T))]), maxT=mx, p_value=float(pval))
    return res
