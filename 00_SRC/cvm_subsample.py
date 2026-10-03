"""Sequential empirical-copula Cramer-von Mises change-point statistic with subsample ranks.

What it computes
----------------
The copula change-point statistic of Buecher, Kojadinovic, Rohmer and Segers (2014, J. Multivariate
Anal. 132, 111-128), built from its definition. For a candidate split k of a series of length n,
the first k observations and the last n - k observations each get their OWN empirical copula,
computed from ranks taken WITHIN that subsample:

    C_{a:b}(u) = (1/m) sum_{i=a}^{b} prod_j 1{ R^{a:b}_{ij} / m <= u_j },   m = b - a + 1,

where R^{a:b}_{ij} is the rank of X_{ij} among X_{aj}, ..., X_{bj}. The statistic at split k is the
Cramer-von Mises functional of the weighted difference process, integrated against the empirical
copula of the whole series,

    S_n(k) = (k (n-k) / n^{3/2})^2  (1/n) sum_{i=1}^{n} ( C_{1:k}(U_i) - C_{k+1:n}(U_i) )^2,
    U_i = (R^{1:n}_{i1} / n, ..., R^{1:n}_{id} / n),

and the test statistic of that paper is max_k S_n(k). The weight is the square of
sqrt(n) lambda_n(0,s) lambda_n(s,1) with lambda_n(0,s) = k/n, lambda_n(s,1) = (n-k)/n, as in that
paper; integrating against dC_{1:n} is the average over the n whole-series pseudo-observations.

Purpose
-------
The global-rank copula statistic (dots.domi.copula_cvm; Supplementary Table B.6) evaluates the two
segment copulas from GLOBAL pseudo-observations, so a change confined to one margin moves the
segment copulas and the statistic rejects under the marginal-only control M1 (0.81 at s = 2).
The test of Buecher et al. re-ranks within each subsample, which removes the marginal change from
each subsample copula. This module implements that statistic, the copula comparator (CvM) of
Table 1.

Choices that the definition leaves open (stated here so the implementation is unambiguous)
--------------------------------------------------------------------------------------------
* Pseudo-observations are scaled rank / m (the scaling of the paper), not rank / (m + 1).
* Candidate splits are restricted to the candidate grid shared by every statistic in Table 1
  (k = w, ..., n - w, ctx.grid), so that the max over the grid and the localization rule are the
  same as for the other methods. In that paper the maximum runs over k = 1, ..., n - 1; the weight
  vanishes at both ends, so the restriction removes only splits with tiny weight.
* Calibration is NOT the multiplier bootstrap of the paper: in the Monte Carlo study the threshold
  is the Monte Carlo null quantile of the maximum (Section 5 protocol), identical for all methods.
* The raw curve S_n(k) is returned; the studentized variant is formed by the runner with the
  per-candidate null mean and standard deviation, exactly as for the other baselines (the weight
  then cancels).
* Ties: continuous data are assumed (no ties); ranks are computed with a stable sort.

Computation
-----------
With whole-series integer ranks g_ij, the event R^{sub}_{ij} <= floor(m g_lj / n) is the event that
X_ij is among the floor(m g_lj / n) smallest values of the subsample, i.e. g_ij <= t_lj where t_lj
is the corresponding order statistic of the subsample's global ranks. The subsample copula at U_l
is therefore a two-dimensional dominance count, read from an (n+1) x (n+1) table of counts that is
updated in O(n^2) per added observation. The whole curve costs O(n^3) elementary additions for
d = 2 (about 0.1 s at n = 600); `copula_cvm_subsample_bruteforce` is the direct transcription of
the definition for any d and is used when d != 2.

Usage
-----
    from cvm_subsample import copula_cvm_subsample
    curve = copula_cvm_subsample(ctx, "global")     # same signature as dots.domi.copula_cvm
"""
from __future__ import annotations

import numpy as np


def _int_ranks(A: np.ndarray) -> np.ndarray:
    """Column-wise integer ranks 1..n (stable sort; no ties assumed)."""
    A = np.asarray(A, float)
    if A.ndim == 1:
        A = A[:, None]
    n = A.shape[0]
    R = np.empty(A.shape, dtype=np.int64)
    for j in range(A.shape[1]):
        o = np.argsort(A[:, j], kind="mergesort")
        R[o, j] = np.arange(1, n + 1)
    return R


def _weights(grid: np.ndarray, n: int) -> np.ndarray:
    return (grid * (n - grid) / n ** 1.5) ** 2


def copula_cvm_subsample_curve(U: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """S_n(k) for k in `grid`, bivariate fast path. U: (n, 2) array of any values (only the
    column-wise orderings are used)."""
    G = _int_ranks(U)
    n, d = G.shape
    if d != 2:
        return copula_cvm_subsample_bruteforce(U, grid)
    g1, g2 = G[:, 0], G[:, 1]
    grid = np.asarray(grid, dtype=np.int64)
    want = set(int(k) for k in grid)

    def side(order_points, sizes_at):
        """Dominance-count table grown one observation at a time; at each k in `want` the counts
        #{i in subsample: g1_i <= t1_l, g2_i <= t2_l} for every evaluation point l."""
        M = np.zeros((n + 1, n + 1), dtype=np.int32)
        out = {}
        for step, i in enumerate(order_points):
            M[g1[i]:, g2[i]:] += 1
            k = sizes_at(step)
            if k in want:
                idx = order_points[:step + 1]
                m = step + 1
                s1 = np.sort(g1[idx])
                s2 = np.sort(g2[idx])
                m1 = (m * g1) // n            # floor(m * U_l1): within-subsample rank cut-off
                m2 = (m * g2) // n
                t1 = np.where(m1 > 0, s1[np.maximum(m1 - 1, 0)], 0)
                t2 = np.where(m2 > 0, s2[np.maximum(m2 - 1, 0)], 0)
                out[k] = M[t1, t2] / m
        return out

    left = side(np.arange(n), lambda step: step + 1)                 # prefix [0, k): k = step+1
    right = side(np.arange(n - 1, -1, -1), lambda step: n - step - 1)  # suffix [k, n): k = n-1-step
    diff2 = np.array([np.mean((left[int(k)] - right[int(k)]) ** 2) for k in grid])
    return _weights(grid, n) * diff2


def copula_cvm_subsample_bruteforce(U: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Direct transcription of the definition (any dimension d; O(n^2 d) per split)."""
    U = np.asarray(U, float)
    if U.ndim == 1:
        U = U[:, None]
    n = U.shape[0]
    P = _int_ranks(U) / n                                  # whole-series pseudo-observations

    def ecop(rows):
        m = len(rows)
        V = _int_ranks(U[rows]) / m                        # subsample pseudo-observations
        return np.all(V[:, None, :] <= P[None, :, :], axis=2).mean(0)   # C_sub(U_l), l = 1..n

    out = np.empty(len(grid))
    for q, k in enumerate(grid):
        k = int(k)
        out[q] = np.mean((ecop(np.arange(k)) - ecop(np.arange(k, n))) ** 2)
    return _weights(np.asarray(grid), n) * out


def copula_cvm_subsample(ctx, mode: str = "global") -> np.ndarray:
    """Subsample-rank copula CvM curve over ctx.grid (signature and return convention of
    dots.domi.copula_cvm; only the global split form exists for this statistic).

    ctx.UX, ctx.UY are the whole-series rank inputs; the statistic uses only their orderings, so it
    is identical when computed from the raw data."""
    if mode != "global":
        raise ValueError("the subsample-rank copula statistic is defined for the global split only")
    U = np.hstack([ctx.UX, ctx.UY])
    return copula_cvm_subsample_curve(U, ctx.grid)
