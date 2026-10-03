"""Nonparametric multiple change-point baselines: kernel change-point detection (KCP) and e.divisive.

Purpose
-------
The multiple-break comparison of Section 6.3 (Supplementary Table B.7) sets Holevo partitioning against
rank-Gaussian PELT and kernel binary segmentation. Two established nonparametric segmenters are
added here so that the comparison includes the standard references:

  KCP        kernel change-point detection by penalized optimal partitioning with the kernel
             least-squares cost (Arlot, Celisse and Harchaoui, 2019). For a segment s = [a, b),
                 C(s) = sum_{t in s} k(z_t, z_t) - (1 / |s|) sum_{t, t' in s} k(z_t, z_t'),
             the within-segment dispersion of the kernel feature map; the segmentation minimizes
             sum_s C(s) + beta * (number of change points) exactly by dynamic programming over a
             candidate grid (dots.pelt.pelt_from_costs). Kernel: Gaussian,
             k(z, z') = exp(-||z - z'||^2 / m), m the median of the pairwise squared distances
             (median heuristic; the same kernel as the kernel binary segmentation baseline of
             run_experiment.py part e2).
  e.divisive hierarchical divisive segmentation with the energy statistic (Matteson and James,
             2014), alpha = 1. For a segment, the new change point tau and the end point kappa of the
             comparison sample are chosen to maximize
                 Q(tau, kappa) = m k / (m + k) * [ 2/(m k) sum_{A x B} |a - b|
                                  - 1/C(m,2) sum_{A pairs} |a - a'| - 1/C(k,2) sum_{B pairs} |b - b'| ],
             A = z[a:tau), B = z[tau:kappa), m = |A|, k = |B|; the best split over all current
             segments is accepted when its permutation p-value, from R permutations of the
             observations within each current segment, is at most the significance level; the
             procedure stops at the first non-significant split. Split points and kappa are
             searched on the candidate grid with every part at least `min_size` long.

Both work on any (n, p) array; pass ranks (column-wise rank / (n + 1)) for the rank-transformed
variants. All sums over segments are differences of two-dimensional prefix sums of the n x n kernel
or distance matrix, so each cost matrix or split search costs O(n_grid^2) after an O(n^2) pass.

Used by run_seg_baselines.py. No side effects on import.
"""
from __future__ import annotations

import numpy as np

from dots import pelt as P
from dots.perm import ge


# ---------------------------------------------------------------- shared helpers
def sqdist(Z):
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    s = (Z ** 2).sum(1)
    return np.maximum(s[:, None] + s[None, :] - 2.0 * Z @ Z.T, 0.0)


def prefix2d(M):
    """S[i, j] = sum of M[:i, :j]."""
    S = np.zeros((M.shape[0] + 1, M.shape[1] + 1))
    S[1:, 1:] = M.cumsum(0).cumsum(1)
    return S


def box(S, a, b, c, d):
    """Sum of M[a:b, c:d] from the prefix array S (vectorized over array arguments)."""
    return S[b, d] - S[a, d] - S[b, c] + S[a, c]


def gaussian_gram(Z):
    D2 = sqdist(Z)
    med = np.median(D2[np.triu_indices(len(D2), 1)])
    return np.exp(-D2 / (med if med > 0 else 1.0))


# ---------------------------------------------------------------- KCP
def kcp_cost_matrix(Z, grid):
    """C[i, j] = kernel least-squares cost of segment [grid[i], grid[j]) for i < j (NaN otherwise)."""
    S = prefix2d(gaussian_gram(Z))
    g = np.asarray(grid)
    I, J = np.meshgrid(np.arange(len(g)), np.arange(len(g)), indexing="ij")
    a, b = g[I], g[J]
    L = (b - a).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = L - box(S, a, b, a, b) / L          # k(z, z) = 1 for the Gaussian kernel
    C[J <= I] = np.nan
    return C


def kcp_segment(C, grid, beta):
    """Change points (in observation units) of the penalized optimal partition at penalty beta."""
    return [int(grid[i]) for i in P.pelt_from_costs(C, beta)]


# ---------------------------------------------------------------- e.divisive
def _best_split(S, a, b, step, min_size):
    """Best (Q, tau) inside [a, b) from the prefix array S of the distance matrix (global indices).
    tau and kappa run over a + multiples of `step`, kappa also over b; every part >= min_size."""
    taus = np.arange(a + min_size, b - min_size + 1, step)
    if len(taus) == 0:
        return -np.inf, None
    kap = np.unique(np.append(np.arange(a + 2 * min_size, b + 1, step), b))
    T, Kp = np.meshgrid(taus, kap, indexing="ij")
    ok = (Kp - T) >= min_size
    if not ok.any():
        return -np.inf, None
    T, Kp = T[ok], Kp[ok]
    m = (T - a).astype(float)
    k = (Kp - T).astype(float)
    cross = box(S, a, T, T, Kp) / (m * k)
    wa = box(S, a, T, a, T) / (m * (m - 1))      # ordered-pair sum / (m(m-1)) = mean over C(m,2)
    wb = box(S, T, Kp, T, Kp) / (k * (k - 1))
    Q = m * k / (m + k) * (2 * cross - wa - wb)
    i = int(np.argmax(Q))
    return float(Q[i]), int(T[i])


def _best_over_segments(Dm, bounds, step, min_size, perm=None):
    """Best split over all segments; `perm` (global index array) permutes within segments."""
    M = Dm if perm is None else Dm[np.ix_(perm, perm)]
    S = prefix2d(M)
    best = (-np.inf, None, None)
    for (a, b) in bounds:
        q, t = _best_split(S, a, b, step, min_size)
        if q > best[0]:
            best = (q, t, (a, b))
    return best


def e_divisive(Z, sig=0.05, R=99, min_size=60, step=10, rng=None, max_cp=10):
    """Hierarchical energy-statistic segmentation (alpha = 1).
    Returns (sorted change points, list of (tau, Q, p) in the order found, the last entry being the
    rejected split when the procedure stopped on a non-significant p-value)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    Z = np.asarray(Z, float)
    n = len(Z)
    Dm = np.sqrt(sqdist(Z))
    bounds = [(0, n)]
    cps, trail = [], []
    while len(cps) < max_cp:
        q, t, seg = _best_over_segments(Dm, bounds, step, min_size)
        if t is None:
            break
        n_ge = 1
        for _ in range(R):
            perm = np.arange(n)
            for (a, b) in bounds:
                perm[a:b] = a + rng.permutation(b - a)
            n_ge += ge(_best_over_segments(Dm, bounds, step, min_size, perm)[0], q)
        p = n_ge / (R + 1)
        trail.append((t, q, p))
        if p > sig:
            break
        cps.append(t)
        bounds.remove(seg)
        bounds += [(seg[0], t), (t, seg[1])]
        bounds.sort()
    return sorted(cps), trail
