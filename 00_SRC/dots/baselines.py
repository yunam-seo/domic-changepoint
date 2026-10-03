"""Gaussian likelihood-ratio baseline on the candidate grid (Supplementary Table B.6).

For a split t the segments are [0, t) and [t, n); the statistic is the unweighted Gaussian log-likelihood
ratio n logdet C - n1 logdet C1 - n2 logdet C2 of the pooled and segment covariances.
"""
from __future__ import annotations

import numpy as np

from .detect import Context


def _segments(ctx: Context, mode: str):
    n, w = ctx.n, ctx.w
    for i, t in enumerate(ctx.grid):
        yield i, t, (0, t), (t, n), t * (n - t) / n


def _logdet_ridge(C: np.ndarray, ridge: float) -> float:
    p = C.shape[0]
    lam = np.linalg.eigvalsh(C)
    lam = np.clip(lam, 0, None) + ridge * max(np.trace(C), 1e-12) / p
    return float(np.log(lam).sum())


def gauss_lr(ctx: Context, mode: str) -> np.ndarray:
    ca = ctx.cache_raw()
    out = np.empty(len(ctx.grid))
    for i, t, (a1, b1), (a2, b2), wt in _segments(ctx, mode):
        n1, n2 = b1 - a1, b2 - a2
        C1 = ca.covariance(a1, b1)
        C2 = ca.covariance(a2, b2)
        # pooled: covariance of the union (includes mean-shift contribution)
        Cp = ca.covariance(a1, b2) if mode == "global" else _pooled_cov(ca, a1, b1, a2, b2)
        out[i] = (n1 + n2) * _logdet_ridge(Cp, ctx.ridge) - n1 * _logdet_ridge(C1, ctx.ridge) \
            - n2 * _logdet_ridge(C2, ctx.ridge)
    return out


def _pooled_cov(ca, a1, b1, a2, b2):
    n1, n2 = b1 - a1, b2 - a2
    M = (n1 * ca.second_moment(a1, b1) + n2 * ca.second_moment(a2, b2)) / (n1 + n2)
    m = (n1 * ca.mean(a1, b1) + n2 * ca.mean(a2, b2)) / (n1 + n2)
    return M - np.outer(m, m)


BASELINES = {
    "GaussLR": gauss_lr,
}
