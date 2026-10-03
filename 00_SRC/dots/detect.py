"""Feature density operators on a candidate grid t in [w, n-w] with segments [0, t) and [t, n).

`Context` caches the segment moments shared by the statistics of one run; `qd_global` returns the
Holevo statistic and its spectral (commuting) part used in the ablation (Supplementary B.2).
"""
from __future__ import annotations

import numpy as np

from .encode import MomentCache, augment
from . import qdiv as Q


class Context:
    def __init__(self, Z: np.ndarray, w: int, c_aug: float = 1.0, ridge: float = 1e-2):
        """Z: encoded design (n, p) without augmentation; the statistics use augment(Z, c_aug)."""
        self.Z = np.asarray(Z, float)
        if self.Z.ndim == 1:
            self.Z = self.Z[:, None]
        self.n, self.p = self.Z.shape
        self.w = int(w)
        self.c_aug = c_aug
        self.ridge = ridge
        self.grid = np.arange(self.w, self.n - self.w + 1)  # split positions t
        self._cache = {}

    # ---------------------------------------------------------------- shared lazies
    def cache_raw(self) -> MomentCache:
        if "raw" not in self._cache:
            self._cache["raw"] = MomentCache(self.Z)
        return self._cache["raw"]

    def cache_aug(self) -> MomentCache:
        if "aug" not in self._cache:
            self._cache["aug"] = MomentCache(augment(self.Z, self.c_aug))
        return self._cache["aug"]


    def global_states(self):
        """For every split t in grid: left rho [0,t) and right rho [t,n) with eigen data."""
        if "glob" not in self._cache:
            ca = self.cache_aug()
            L, R = [], []
            for t in self.grid:
                rl = ca.rho(0, t)
                rr = ca.rho(t, self.n)
                ll, Ul = Q.eigh_psd(rl)
                lr, Ur = Q.eigh_psd(rr)
                L.append((rl, ll, Ul))
                R.append((rr, lr, Ur))
            rho_all = ca.rho(0, self.n)
            self._cache["glob"] = dict(L=L, R=R, rho_all=rho_all, ent_all=Q.vn_entropy(rho_all))
        return self._cache["glob"]


# ---------------------------------------------------------------- helpers
def sqdist(Z: np.ndarray) -> np.ndarray:
    sq = (Z * Z).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T
    return np.clip(D2, 0.0, None)


def prefix2d(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    P = np.zeros((n + 1, n + 1))
    P[1:, 1:] = K.cumsum(0).cumsum(1)
    return P


def block_sum(P: np.ndarray, a1: int, b1: int, a2: int, b2: int) -> float:
    return P[b1, b2] - P[a1, b2] - P[b1, a2] + P[a1, a2]


# ---------------------------------------------------------------- density-operator statistics (Holevo: ablation, Supplementary B.2)


def qd_global(ctx: Context):
    """Global (single-break) statistics of the feature density operator: Holevo chi and its spectral
    (commuting) part, the two entropy functionals of the ablation (Supplementary Section B.2)."""
    gs = ctx.global_states()
    ca = ctx.cache_aug()
    n = ctx.n
    out = {k: np.empty(len(ctx.grid)) for k in ["QDg-Holevo", "QDg-SpecHolevo"]}
    S_all = gs["ent_all"]
    # energy (trace) weights: rho_all = q_l rho_l + q_r rho_r exactly, so chi is a true Holevo quantity
    # and the segment cost  C(seg) = E_seg * S(rho_seg),  E_seg = sum ||z||^2,  is additive over segments (optimal partitioning).
    E_all = np.trace(ca.S[n])
    for i, t in enumerate(ctx.grid):
        rl, ll, Ul = gs["L"][i]
        rr, lr, Ur = gs["R"][i]
        El = np.trace(ca.S[t])
        pl = El / E_all
        pr = 1.0 - pl
        Sl, Sr = Q.entropy_from_eigs(ll), Q.entropy_from_eigs(lr)
        out["QDg-Holevo"][i] = n * max(S_all - pl * Sl - pr * Sr, 0.0)
        # spectral (commuting) analog: entropy of mixture of sorted spectra
        mix_spec = pl * np.sort(ll) + pr * np.sort(lr)
        out["QDg-SpecHolevo"][i] = n * max(Q.shannon(mix_spec) - pl * Sl - pr * Sr, 0.0)
    return out
