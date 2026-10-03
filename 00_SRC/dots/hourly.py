"""Segment statistics for long hourly series.

At n ~ 70,000 the per-sample prefix sums of psi psi^T would need (n+1) x 64 x 64 doubles, about
2.3 GB. Nothing needs that resolution: candidate change points are placed on a coarse grid, so the
series is first aggregated into per-interval moment sums, which are exact, and every segment moment
is then a difference of cumulative sums over at most a few hundred intervals.
"""
from __future__ import annotations
import numpy as np
from .domi import ranks01, unit_rff


def features(x, y, D=8, sx=2026, sy=2027, tie_seed=None):
    """tie_seed=None ranks tied values in time order (the reported runs); an integer breaks ties at
    random with a data-independent uniform jitter drawn from that seed, which keeps the rank vector
    exchangeable whenever the record is exchangeable (P1 applied to the record augmented by the jitter)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if tie_seed is not None:
        j = np.random.default_rng(tie_seed).uniform(size=(len(x), 2))
        x = np.lexsort((j[:, 0], x)).argsort().astype(float)
        y = np.lexsort((j[:, 1], y)).argsort().astype(float)
    FX = unit_rff(ranks01(x[:, None]), D, sx)
    FY = unit_rff(ranks01(y[:, None]), D, sy)
    J = np.einsum("ti,tj->tij", FX, FY).reshape(len(FX), D * D)
    return FX, FY, J


def block_moments(FX, FY, J, edges):
    """Exact per-interval moment sums and counts for the intervals defined by `edges`."""
    B = len(edges) - 1
    D = FX.shape[1]
    mx = np.empty((B, D, D)); my = np.empty((B, D, D)); mj = np.empty((B, D * D, D * D))
    cnt = np.empty(B, dtype=int)
    for k in range(B):
        a, b = edges[k], edges[k + 1]
        mx[k] = FX[a:b].T @ FX[a:b]
        my[k] = FY[a:b].T @ FY[a:b]
        mj[k] = J[a:b].T @ J[a:b]
        cnt[k] = b - a
    return mx, my, mj, cnt


def prefix(mx, my, mj, cnt):
    z = lambda a: np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])
    return z(mx), z(my), z(mj), np.concatenate([[0], np.cumsum(cnt)])


def _ent(M):
    l = np.clip(np.linalg.eigvalsh(M), 1e-300, None)
    l = l[l > 1e-14]
    return float(-(l * np.log(l)).sum())


def seg_domi(P, i, j):
    """DOMI of the segment spanning intervals [i, j) of the prefix arrays P."""
    cx, cy, cj, cn = P
    m = cn[j] - cn[i]
    if m <= 0:
        return np.nan
    return (_ent((cx[j] - cx[i]) / m) + _ent((cy[j] - cy[i]) / m) - _ent((cj[j] - cj[i]) / m))


def seg_cost(P, i, j):
    """Energy-weighted von Neumann cost of the joint state, E_s * S(rho_XY,s); unit-norm features
    make E_s the number of observations."""
    cx, cy, cj, cn = P
    m = cn[j] - cn[i]
    if m <= 0:
        return 0.0
    return m * _ent((cj[j] - cj[i]) / m)


def domi_diff_curve(P, B, lo=2, hi=None):
    """Weighted |I_left - I_right| over interval-level split points."""
    hi = B - lo if hi is None else hi
    ts = np.arange(lo, hi + 1)
    n = P[3][B]
    out = np.empty(len(ts))
    for k, t in enumerate(ts):
        nl = P[3][t]
        out[k] = np.sqrt(nl * (n - nl) / n) * abs(seg_domi(P, 0, t) - seg_domi(P, t, B))
    return ts, out


def block_perm_order(B, blocks_per_unit, rng):
    """Permute the ORDER of super-blocks of `blocks_per_unit` consecutive intervals."""
    nb = B // blocks_per_unit
    order = rng.permutation(nb)
    idx = np.concatenate([np.arange(k * blocks_per_unit, (k + 1) * blocks_per_unit) for k in order])
    if nb * blocks_per_unit < B:
        idx = np.concatenate([idx, np.arange(nb * blocks_per_unit, B)])
    return idx
