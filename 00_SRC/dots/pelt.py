"""Holevo partitioning: penalized optimal partitioning with the energy-weighted von Neumann cost
(Section 4.4), plus baselines.

Cost (Supplementary Section A.3):
    C([a,b)) = E_[a,b) * S(rho_[a,b)),  E = sum ||z_t||^2,  rho = trace-normalized second moment.
Additive across segments; the single-split gain equals E_total * Holevo chi >= 0.
All costs are computed from a MomentCache prefix sum on a candidate grid (step g), so the
cost matrix over grid pairs is built once and the partitioning and penalty grids reuse it.
"""
from __future__ import annotations

import numpy as np

from .encode import MomentCache


def cost_matrix_vn(cache: MomentCache, grid: np.ndarray) -> np.ndarray:
    """C[i, j] = energy-weighted vN cost of segment [grid[i], grid[j]) for i < j."""
    m = len(grid)
    C = np.full((m, m), np.nan)
    for i in range(m):
        for j in range(i + 1, m):
            a, b = grid[i], grid[j]
            M = cache.second_moment(a, b)
            E = np.trace(M) * (b - a)
            lam = np.clip(np.linalg.eigvalsh(M), 0, None)
            s = lam.sum()
            lam = lam[lam > 0] / s if s > 0 else lam
            C[i, j] = E * float(-(lam * np.log(lam)).sum()) if s > 0 else 0.0
    return C


def cost_matrix_gauss2(cache: MomentCache, grid: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Gaussian log-likelihood cost |seg| * log det(Cov_seg) for low-dim data (baseline)."""
    m = len(grid)
    C = np.full((m, m), np.nan)
    p = cache.p
    for i in range(m):
        for j in range(i + 1, m):
            a, b = grid[i], grid[j]
            cov = cache.covariance(a, b) + ridge * np.eye(p)
            C[i, j] = (b - a) * float(np.linalg.slogdet(cov)[1])
    return C


def pelt_from_costs(C: np.ndarray, beta: float) -> list[int]:
    """Exact optimal-partitioning DP over the candidate grid (no PELT pruning; the name is historical).
    Returns indices (into grid) of interior change points."""
    m = C.shape[0]
    F = np.full(m, np.inf)
    F[0] = -beta
    prev = np.zeros(m, dtype=int)
    for j in range(1, m):
        vals = F[:j] + C[np.arange(j), j] + beta
        k = int(np.argmin(vals))
        F[j] = vals[k]
        prev[j] = k
    cps = []
    j = m - 1
    while j > 0:
        k = prev[j]
        if k > 0:
            cps.append(k)
        j = k
    return sorted(cps)


def binseg_from_stat(stat_fn, n: int, w: int, max_cp: int = 6):
    """Generic binary segmentation: stat_fn(a, b) -> (best_t, best_gain) within [a, b).
    Returns list of (t, gain) sorted by gain descending (model selection done by caller)."""
    found = []
    segs = [(0, n)]
    for _ in range(max_cp):
        best = None
        for (a, b) in segs:
            if b - a < 2 * w + 2:
                continue
            t, g = stat_fn(a, b)
            if t is not None and (best is None or g > best[2]):
                best = (a, b, g, t)
        if best is None:
            break
        a, b, g, t = best
        found.append((t, g))
        segs.remove((a, b))
        segs += [(a, t), (t, b)]
    return found


# ---------------------------------------------------------------- evaluation helpers
def hausdorff(true_cps, est_cps, n):
    if not true_cps and not est_cps:
        return 0.0
    if not true_cps or not est_cps:
        return float(n)
    t = np.asarray(true_cps, float)
    e = np.asarray(est_cps, float)
    d1 = max(min(abs(x - y) for y in e) for x in t)
    d2 = max(min(abs(x - y) for y in t) for x in e)
    return float(max(d1, d2))


def seg_labels(cps, n):
    lab = np.zeros(n, dtype=int)
    for i, c in enumerate(sorted(cps)):
        lab[c:] = i + 1
    return lab


def rand_index_adj(true_cps, est_cps, n):
    a = seg_labels(true_cps, n)
    b = seg_labels(est_cps, n)
    ka, kb = a.max() + 1, b.max() + 1
    cont = np.zeros((ka, kb))
    for i, j in zip(a, b):
        cont[i, j] += 1
    sum_comb = lambda x: (x * (x - 1) / 2).sum()  # noqa: E731
    sij = sum_comb(cont)
    si = sum_comb(cont.sum(1))
    sj = sum_comb(cont.sum(0))
    sn = n * (n - 1) / 2
    exp = si * sj / sn
    mx = 0.5 * (si + sj)
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0
