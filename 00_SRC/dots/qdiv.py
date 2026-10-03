"""Entropies of density operators and probability vectors.

Eigenvalues are clipped at 0 and 0*log(0) := 0, so rank-deficient operators need no regularization.
"""
from __future__ import annotations

import numpy as np


def eigvals_psd(rho: np.ndarray) -> np.ndarray:
    lam = np.linalg.eigvalsh(rho)
    lam = np.clip(lam, 0.0, None)
    s = lam.sum()
    return lam / s if s > 0 else lam


def eigh_psd(rho: np.ndarray):
    lam, U = np.linalg.eigh(rho)
    lam = np.clip(lam, 0.0, None)
    return lam, U


def shannon(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def vn_entropy(rho: np.ndarray) -> float:
    """von Neumann entropy S(rho) = -tr rho log rho (natural log)."""
    return shannon(eigvals_psd(rho))


def entropy_from_eigs(lam: np.ndarray) -> float:
    return shannon(lam)


