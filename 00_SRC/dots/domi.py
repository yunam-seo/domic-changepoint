"""Density-operator mutual information (DOMI) statistics for dependence-structure change points
(identifiers in this module use the prefix DOMI).

Blocks X, Y -> rank transform -> RFF features phi_X, phi_Y (unit norm) -> joint pure product
states |phi_X> (x) |phi_Y>. Segment density operators via prefix sums:
    rho_X  = E[phi_X phi_X^T],  rho_Y likewise,  rho_XY = E[(phi_X (x) phi_Y)(...)^T]  (D^2 x D^2)
Unit-norm features make Tr_Y rho_XY = rho_X exactly.  DOMI  I = S(rho_X) + S(rho_Y) - S(rho_XY).

Also: HSIC (RFF cross-covariance Frobenius^2), distance correlation, Spearman, empirical-copula
Cramér–von Mises — all on the same rank inputs — and classical joint-distribution baselines.
"""
from __future__ import annotations

import numpy as np

from .encode import MomentCache
from . import qdiv as Q
from .extras import rff


# ---------------------------------------------------------------- inputs
def ranks01(X: np.ndarray) -> np.ndarray:
    """Column-wise pseudo-observations rank/(n+1) computed on the whole series."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    R = np.empty_like(X)
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="mergesort")
        r = np.empty(n)
        r[order] = np.arange(1, n + 1)
        R[:, j] = r / (n + 1)
    return R


def unit_rff(U: np.ndarray, D: int, seed: int) -> np.ndarray:
    F = rff(U, D=D, seed=seed)
    return F / np.linalg.norm(F, axis=1, keepdims=True)


class DOMIContext:
    """Prefix-sum caches for phi_X, phi_Y and their tensor product; DOMI per segment in O(D^6)."""

    def __init__(self, X, Y, w: int, D: int = 8, seed: int = 2026, n_perm: int = 1):
        self.n = X.shape[0]
        self.w = w
        self.grid = np.arange(w, self.n - w + 1)
        UX, UY = ranks01(X), ranks01(Y)
        self.UX, self.UY = UX, UY
        self.FX = unit_rff(UX, D, seed)
        self.FY = unit_rff(UY, D, seed + 1)
        self.D = D
        self.cx = MomentCache(self.FX)
        self.cy = MomentCache(self.FY)
        J = np.einsum("ti,tj->tij", self.FX, self.FY).reshape(self.n, D * D)  # kron per sample
        self.cj = MomentCache(J)
        self._ent = {}
        # permutation-null copies (Y rows shuffled): bias correction of plug-in DOMI per segment length
        self.perm = []
        prng = np.random.default_rng(seed + 99)
        for _ in range(n_perm):
            idx = prng.permutation(self.n)
            FYp = self.FY[idx]
            Jp = np.einsum("ti,tj->tij", self.FX, FYp).reshape(self.n, D * D)
            self.perm.append((MomentCache(FYp), MomentCache(Jp)))
        self._ent_perm = {}

    def entropies(self, a: int, b: int):
        key = (a, b)
        if key not in self._ent:
            Sx = Q.vn_entropy(self.cx.rho(a, b))
            Sy = Q.vn_entropy(self.cy.rho(a, b))
            Sxy = Q.vn_entropy(self.cj.rho(a, b))
            self._ent[key] = (Sx, Sy, Sxy)
        return self._ent[key]

    def domi(self, a: int, b: int) -> float:
        Sx, Sy, Sxy = self.entropies(a, b)
        return max(Sx + Sy - Sxy, 0.0)

    def domi_perm(self, a: int, b: int) -> float:
        """Mean plug-in DOMI of the permuted-Y copies over [a,b) (estimates the independence-null bias)."""
        key = (a, b)
        if key not in self._ent_perm:
            Sx = self.entropies(a, b)[0]
            vals = []
            for cy, cj in self.perm:
                vals.append(Sx + Q.vn_entropy(cy.rho(a, b)) - Q.vn_entropy(cj.rho(a, b)))
            self._ent_perm[key] = float(np.mean(vals)) if vals else 0.0
        return self._ent_perm[key]

    def domi_bc(self, a: int, b: int) -> float:
        """Bias-corrected DOMI = plug-in DOMI - permutation-null DOMI for the same segment."""
        return self.domi(a, b) - self.domi_perm(a, b)


def _segments(ctx, mode="global"):
    """Candidate split t of the whole window: segments [0, t) and [t, n) with weight t(n - t)/n.

    `mode` is kept for call compatibility; only this whole-window ("global") split is implemented,
    and the statistics below pass it through unchanged."""
    n = ctx.n
    for i, t in enumerate(ctx.grid):
        yield i, t, (0, t), (t, n), t * (n - t) / n


# ---------------------------------------------------------------- DOMI statistics
def domi_stats(ctx: DOMIContext, mode: str = "global") -> dict:
    """DOMI-diff (DOMI difference) and the joint-state Holevo statistic on the tensor density operator."""
    out = {"DOMI-diff": np.empty(len(ctx.grid)), "Holevo-joint": np.empty(len(ctx.grid))}
    S_all = ctx.entropies(0, ctx.n)[2]
    E_all = np.trace(ctx.cj.S[ctx.n])
    n = ctx.n
    for i, t, (a1, b1), (a2, b2), wt in _segments(ctx, mode):
        Il, Ir = ctx.domi(a1, b1), ctx.domi(a2, b2)
        out["DOMI-diff"][i] = np.sqrt(wt) * abs(Il - Ir)
        El = np.trace(ctx.cj.S[t])
        pl = El / E_all
        Sl, Sr = ctx.entropies(a1, b1)[2], ctx.entropies(a2, b2)[2]
        out["Holevo-joint"][i] = n * max(S_all - pl * Sl - (1 - pl) * Sr, 0.0)
    return out


# ---------------------------------------------------------------- classical dependence baselines
def hsic_diff(ctx: DOMIContext, mode: str) -> np.ndarray:
    # cross second moment E[phi_X phi_Y^T] is the (D x D) reshape of the mean kron vector
    out = np.empty(len(ctx.grid))
    for i, t, (a1, b1), (a2, b2), wt in _segments(ctx, mode):
        out[i] = np.sqrt(wt) * abs(_hsic(ctx, a1, b1) - _hsic(ctx, a2, b2))
    return out


def _hsic(ctx, a, b):
    n = b - a
    mx, my = ctx.cx.mean(a, b), ctx.cy.mean(a, b)
    Mxy = (ctx.cj.S1[b] - ctx.cj.S1[a]).reshape(ctx.D, ctx.D) / n  # S1 = prefix of kron vectors = E[phi_X phi_Y^T]
    C = Mxy - np.outer(mx, my)
    return float((C * C).sum())


def spearman_diff(ctx: DOMIContext, mode: str) -> np.ndarray:
    """|rho_S(L) - rho_S(R)| from whole-series ranks; for vector blocks, the mean absolute difference
    over all coordinate pairs."""
    UX, UY = ctx.UX, ctx.UY
    pref = np.zeros((ctx.n + 1, UX.shape[1], UY.shape[1]))
    pref[1:] = np.cumsum(np.einsum("ti,tj->tij", UX - 0.5, UY - 0.5), axis=0)
    pX = np.zeros((ctx.n + 1, UX.shape[1]))
    pX[1:] = np.cumsum(UX - 0.5, axis=0)
    pY = np.zeros((ctx.n + 1, UY.shape[1]))
    pY[1:] = np.cumsum(UY - 0.5, axis=0)
    p2X = np.zeros((ctx.n + 1, UX.shape[1]))
    p2X[1:] = np.cumsum((UX - 0.5) ** 2, axis=0)
    p2Y = np.zeros((ctx.n + 1, UY.shape[1]))
    p2Y[1:] = np.cumsum((UY - 0.5) ** 2, axis=0)

    def corr(a, b):
        n = b - a
        mx, my = (pX[b] - pX[a]) / n, (pY[b] - pY[a]) / n
        cxy = (pref[b] - pref[a]) / n - np.outer(mx, my)
        vx = (p2X[b] - p2X[a]) / n - mx ** 2
        vy = (p2Y[b] - p2Y[a]) / n - my ** 2
        return cxy / np.sqrt(np.outer(vx, vy) + 1e-12)

    out = np.empty(len(ctx.grid))
    for i, t, (a1, b1), (a2, b2), wt in _segments(ctx, mode):
        out[i] = np.sqrt(wt) * np.abs(corr(a1, b1) - corr(a2, b2)).mean()
    return out


def copula_cvm(ctx: DOMIContext, mode: str) -> np.ndarray:
    """Empirical-copula Cramér–von Mises between segments (Bücher–Kojadinovic type), evaluated at the
    sample points; prefix sums over indicator matrix 1{U_i <= U_j} (all coordinates)."""
    U = np.hstack([ctx.UX, ctx.UY])
    n = ctx.n
    Ind = np.all(U[:, None, :] <= U[None, :, :], axis=2).astype(float)  # Ind[i,j] = 1{U_i <= U_j}
    P = np.zeros((n + 1, n))
    P[1:] = np.cumsum(Ind, axis=0)  # P[b,j]-P[a,j] = #{i in [a,b): U_i <= U_j}
    out = np.empty(len(ctx.grid))
    for i, t, (a1, b1), (a2, b2), wt in _segments(ctx, mode):
        CL = (P[b1] - P[a1]) / (b1 - a1)
        CR = (P[b2] - P[a2]) / (b2 - a2)
        out[i] = wt * np.mean((CL - CR) ** 2)
    return out


def dcor_diff(ctx: DOMIContext, mode: str, X=None, Y=None) -> np.ndarray:
    """|dCor(L) - dCor(R)| with distance correlation computed on rank inputs (marginal-free).
    X and Y are unused; the rank inputs come from ctx."""
    UX, UY = ctx.UX, ctx.UY

    def pd(A):
        sq = (A * A).sum(1)
        return np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * A @ A.T, 0, None))

    DX, DY = pd(UX), pd(UY)

    def dcor(a, b):
        A = DX[a:b, a:b]
        B = DY[a:b, a:b]
        A = A - A.mean(0) - A.mean(1)[:, None] + A.mean()
        B = B - B.mean(0) - B.mean(1)[:, None] + B.mean()
        dxy = (A * B).mean()
        dxx = (A * A).mean()
        dyy = (B * B).mean()
        return np.sqrt(max(dxy, 0) / np.sqrt(dxx * dyy + 1e-18))

    out = np.empty(len(ctx.grid))
    for i, t, (a1, b1), (a2, b2), wt in _segments(ctx, mode):
        out[i] = np.sqrt(wt) * abs(dcor(a1, b1) - dcor(a2, b2))
    return out


DEP_BASELINES = {"HSIC-diff": hsic_diff, "dCor-diff": dcor_diff, "Spearman-diff": spearman_diff, "CopulaCvM": copula_cvm}
