"""Random Fourier features and per-t studentization.

    phi(z) = sqrt(2/D) cos(W z + b),  W ~ N(0, 2*gamma I),  gamma = 1/median||z-z'||^2
The second moment of phi is the kernel density operator of the statistics.
"""
from __future__ import annotations

import numpy as np


def rff(Z: np.ndarray, D: int = 100, seed: int = 0, gamma: float | None = None) -> np.ndarray:
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    n, p = Z.shape
    if gamma is None:
        idx = np.random.default_rng(seed + 1).choice(n, size=min(n, 400), replace=False)
        S = Z[idx]
        sq = (S * S).sum(1)
        D2 = np.clip(sq[:, None] + sq[None, :] - 2 * S @ S.T, 0, None)
        med = np.median(D2[np.triu_indices(len(idx), 1)])
        gamma = 1.0 / (med if med > 0 else 1.0)
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((p, D)) * np.sqrt(2 * gamma)
    b = rng.uniform(0, 2 * np.pi, D)
    return np.sqrt(2.0 / D) * np.cos(Z @ W + b)


def studentize(stat: np.ndarray, mu0: np.ndarray, sd0: np.ndarray) -> np.ndarray:
    return (stat - mu0) / np.where(sd0 > 0, sd0, 1.0)
