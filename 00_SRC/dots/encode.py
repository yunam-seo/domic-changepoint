"""Segment moments of an encoded series.

A cumulative sum of outer products gives the second moment of any interval [a, b) in O(p^2).
"""
from __future__ import annotations

import numpy as np


def augment(Z: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Append a constant column c (>0) so that the second moment contains the mean.
    c <= 0 returns Z unchanged (pure covariance-type encoding, scale invariant)."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    if c is None or c <= 0:
        return Z
    return np.hstack([Z, np.full((Z.shape[0], 1), float(c))])


class MomentCache:
    """Prefix sums of outer products: S[k] = sum_{t<k} z_t z_t^T, shape (n+1, p, p)."""

    def __init__(self, Z: np.ndarray):
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z[:, None]
        self.Z = Z
        self.n, self.p = Z.shape
        # chunked prefix sum of outer products (avoids a second (n,p,p) temporary for large p)
        self.S = np.zeros((self.n + 1, self.p, self.p))
        step = max(1, int(2e7 // (self.p * self.p)))
        for a in range(0, self.n, step):
            b = min(self.n, a + step)
            np.cumsum(np.einsum("ti,tj->tij", Z[a:b], Z[a:b]), axis=0, out=self.S[a + 1:b + 1])
            self.S[a + 1:b + 1] += self.S[a]
        # first moments (for baselines that need means/covariances)
        self.S1 = np.zeros((self.n + 1, self.p))
        np.cumsum(Z, axis=0, out=self.S1[1:])

    def second_moment(self, a: int, b: int) -> np.ndarray:
        """M_[a,b) = (1/(b-a)) sum z z^T."""
        return (self.S[b] - self.S[a]) / (b - a)

    def mean(self, a: int, b: int) -> np.ndarray:
        return (self.S1[b] - self.S1[a]) / (b - a)

    def covariance(self, a: int, b: int, ddof: int = 0) -> np.ndarray:
        m = self.mean(a, b)
        M = self.second_moment(a, b)
        C = M - np.outer(m, m)
        if ddof:
            C *= (b - a) / max(b - a - ddof, 1)
        return C

    def rho(self, a: int, b: int) -> np.ndarray:
        """Density operator of interval [a, b): trace-normalized second moment."""
        M = self.second_moment(a, b)
        tr = np.trace(M)
        if tr <= 0:
            return np.eye(self.p) / self.p
        return M / tr


