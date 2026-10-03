"""Tie handling for permutation p-values.

Different permutations can give equal test statistics, and floating-point rounding (which depends on the
order of summation) can then place a replica marginally below the observed value. A permutation p-value
counts ties in full (Supplementary Section A.5), so a replica counts as at least the observed value
T_0 when it is not smaller than T_0 - 1e-9 max(1, |T_0|).
"""
import numpy as np

RTOL = 1e-9


def ge(a, b, rtol=RTOL):
    """Elementwise a >= b, counting a as at least b when a >= b - rtol * max(1, |b|)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return a >= b - rtol * np.maximum(1.0, np.abs(b))
