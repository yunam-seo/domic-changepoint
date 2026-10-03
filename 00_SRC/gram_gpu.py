#!/usr/bin/env python
"""Gram form of DOMI (matrix-based mutual information) on a CUDA device, full segments, global mode.

What it computes
    For one or many rank series (u_t, v_t), t = 0..n-1, and every candidate t of the grid, the
    Gram-form difference curves of `run_matmi_baseline.matmi_curves` in global mode with every row
    of each segment (MSUB = 10**9):

        a1(t) = sqrt(t (n-t) / n) * | I_1[0,t) - I_1[t,n) |      (von Neumann, alpha = 1)
        r2(t) = sqrt(t (n-t) / n) * | I_2[0,t) - I_2[t,n) |      (order-2 Renyi)

    with I_a(S) = H_a(Gx_S) + H_a(Gy_S) - H_a(Gx_S o Gy_S), Gx_S the Gaussian Gram matrix
    exp(-gamma_x (u_i - u_j)^2) on the rows of segment S, "o" the entrywise product, and H_a the
    order-a entropy of the trace-normalized eigenvalues (eigenvalues clipped at 0, those <= 1e-15
    dropped, exactly as run_matmi_baseline.H / Hr). The bandwidths are supplied by the caller;
    run_matmi_baseline.rff_gamma gives the rule used in the article.

How (two engines)
    gram_curves_lowrank (CPU): every Gram matrix is factorized once per series by pivoted
    Cholesky; a segment's spectrum is taken from the small factor Gram matrix L_S' L_S (see its
    docstring). About 7x faster than the reference on one core.
    gram_curves_gpu (CUDA): segments are grouped by length: a length m carries the left segment [0, m) and the right
    segment [n-m, n), three Gram matrices each, for every series in the batch. Each group is
    reduced to tridiagonal form on the device by batched Householder reflections in float64
    (torch; one reflection step per column, applied to the whole batch at once), and the
    eigenvalues of the tridiagonal matrices are computed on the host with LAPACK dsterf.
    torch.linalg.eigvalsh is not used because on CUDA it processes a batch of matrices larger
    than 32 x 32 one matrix at a time, which is no faster than one CPU core.

Purpose
    It is the numerical engine that makes the Gram form (full segments) affordable inside a
    pair-permutation calibration (K = 99 permuted curves per replicate; run_perm_compare.py) and
    in the off-center break study (run_offcentre.py), which import it; it has no command line.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.linalg import lapack  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover
    torch = None


# ---------------------------------------------------------------- batched tridiagonalization
def tridiagonalise(A):
    """Householder reduction of a batch of real symmetric matrices to tridiagonal form.

    A: torch tensor (B, m, m), float64, symmetric; it is overwritten.
    Returns (d, e): diagonal (B, m) and off-diagonal (B, m-1) of the similar tridiagonal matrices.
    The eigenvalues of each tridiagonal matrix equal those of the input (orthogonal similarity).
    """
    B, m, _ = A.shape
    d = torch.empty((B, m), dtype=A.dtype, device=A.device)
    e = torch.zeros((B, max(m - 1, 1)), dtype=A.dtype, device=A.device)
    for j in range(m - 2):
        d[:, j] = A[:, j, j]
        x = A[:, j + 1:, j]                                   # (B, r)
        nx = torch.linalg.vector_norm(x, dim=1)               # (B,)
        x0 = x[:, 0]
        sgn = torch.where(x0 >= 0, torch.ones_like(x0), -torch.ones_like(x0))
        alpha = -sgn * nx                                     # new sub-diagonal entry
        v = x.clone()
        v[:, 0] = x0 - alpha
        vn2 = (v * v).sum(1)
        ok = vn2 > 0
        tau = torch.where(ok, 2.0 / torch.where(ok, vn2, torch.ones_like(vn2)), torch.zeros_like(vn2))
        e[:, j] = torch.where(ok, alpha, x0)
        S = A[:, j + 1:, j + 1:]                              # trailing block (view)
        p = torch.bmm(S, v.unsqueeze(2)).squeeze(2) * tau[:, None]
        K = 0.5 * tau * (v * p).sum(1)
        w = p - K[:, None] * v
        U = torch.stack([v, w], dim=2)                        # (B, r, 2)
        V = torch.stack([w, v], dim=1)                        # (B, 2, r)
        S.baddbmm_(U, V, alpha=-1.0)                          # S <- S - v w' - w v'
    if m >= 2:
        d[:, m - 2] = A[:, m - 2, m - 2]
        e[:, m - 2] = A[:, m - 1, m - 2]
    d[:, m - 1] = A[:, m - 1, m - 1]
    return d, e[:, :m - 1]


def tridiag_eigvals(d, e):
    """Eigenvalues of a batch of symmetric tridiagonal matrices (numpy (B, m), (B, m-1)) by dsterf."""
    B, m = d.shape
    out = np.empty((B, m))
    for b in range(B):
        w, info = lapack.dsterf(d[b], e[b])
        if info != 0:
            raise RuntimeError(f"dsterf failed (info={info})")
        out[b] = w
    return out


def _tridiag_entropies(de):
    """Worker: (d, e) numpy batch -> (H_1, H_2) per matrix (dsterf, then the entropy rules)."""
    d, e = de
    return _entropies(tridiag_eigvals(d, e))


def batched_entropies(A, pool=None):
    """(H_1, H_2) of the trace-normalized spectra of a batch of symmetric matrices (torch (B, m, m)).

    The tridiagonal reduction runs on A's device. With `pool` (a multiprocessing.Pool) the host
    eigenvalue step is submitted asynchronously and an AsyncResult-like object is returned, so the
    device can reduce the next group meanwhile; call .get() on it. Without a pool the pair of
    arrays is returned directly.
    """
    m = A.shape[-1]
    if m <= 2:
        out = _entropies(torch.linalg.eigvalsh(A).cpu().numpy())
        return _Done(out) if pool is not None else out
    d, e = tridiagonalise(A)
    de = (d.cpu().numpy(), e.cpu().numpy())
    if pool is None:
        return _tridiag_entropies(de)
    B = de[0].shape[0]
    nproc = getattr(pool, "_processes", 1) or 1
    step = max(1, int(np.ceil(B / nproc)))
    parts = [pool.apply_async(_tridiag_entropies, ((de[0][i:i + step], de[1][i:i + step]),))
             for i in range(0, B, step)]
    return _Gather(parts)


class _Done:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v


class _Gather:
    def __init__(self, parts):
        self.parts = parts

    def get(self):
        res = [p.get() for p in self.parts]
        return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


# ---------------------------------------------------------------- entropies (as run_matmi_baseline)
def _entropies(lam):
    """lam: (B, m) raw eigenvalues -> (H_1, H_2) per row, the rules of run_matmi_baseline.eig_unit/H/Hr."""
    lam = np.clip(lam, 0.0, None)
    s = lam.sum(1, keepdims=True)
    p = np.where(s > 0, lam / np.where(s > 0, s, 1.0), lam)
    keep = p > 1e-15
    pl = np.where(keep, p, 1.0)
    h1 = -(np.where(keep, p * np.log(pl), 0.0)).sum(1)
    h2 = -np.log(np.where(keep, p * p, 0.0).sum(1))
    return h1, h2


# ---------------------------------------------------------------- curves
def gram_curves_gpu(UX, UY, gx, gy, w, device="cuda", chunk=None, pool=None):
    """Gram-form difference curves for a batch of series, global mode, full segments.

    UX, UY : arrays (S, n) -- rank inputs (one column per block, as ctx.UX[:, 0] of DOMIContext)
    gx, gy : arrays (S,)   -- Gaussian-kernel bandwidths per series (run_matmi_baseline.rff_gamma)
    w      : candidate window; grid = w..n-w as DOMIContext
    chunk  : series processed per device batch (None = all); lower it if device memory is short
    pool   : optional multiprocessing.Pool for the host eigenvalue step (overlaps with the device)
    Returns (a1, r2), numpy arrays (S, len(grid)).
    """
    UX = np.atleast_2d(np.asarray(UX, float))
    UY = np.atleast_2d(np.asarray(UY, float))
    gx = np.atleast_1d(np.asarray(gx, float))
    gy = np.atleast_1d(np.asarray(gy, float))
    S, n = UX.shape
    chunk = S if chunk is None else chunk
    grid = np.arange(w, n - w + 1)
    a1 = np.empty((S, len(grid)))
    r2 = np.empty((S, len(grid)))
    for s0 in range(0, S, chunk):
        sl = slice(s0, min(S, s0 + chunk))
        a, r = _curves_chunk(UX[sl], UY[sl], gx[sl], gy[sl], grid, n, device, pool)
        a1[sl], r2[sl] = a, r
    return a1, r2


def _curves_chunk(UX, UY, gx, gy, grid, n, device, pool=None):
    S = UX.shape[0]
    dev = torch.device(device)
    ux = torch.as_tensor(UX, dtype=torch.float64, device=dev)
    uy = torch.as_tensor(UY, dtype=torch.float64, device=dev)
    Gx = torch.exp(-torch.as_tensor(gx, dtype=torch.float64, device=dev)[:, None, None]
                   * (ux[:, :, None] - ux[:, None, :]) ** 2)
    Gy = torch.exp(-torch.as_tensor(gy, dtype=torch.float64, device=dev)[:, None, None]
                   * (uy[:, :, None] - uy[:, None, :]) ** 2)
    G3 = torch.stack([Gx, Gy, Gx * Gy], dim=1)                # (S, 3, n, n)
    del Gx, Gy
    gset = set(int(t) for t in grid)
    IL = {}   # t -> (S,) MI of [0,t), alpha 1 and 2
    IR = {}   # t -> (S,) MI of [t,n)
    lengths = sorted(set(int(t) for t in grid) | set(n - int(t) for t in grid), reverse=True)
    pending = []
    for m in lengths:
        blocks, tags = [], []
        if m in gset:                                         # left segment [0, m), t = m
            blocks.append(G3[:, :, :m, :m])
            tags.append(("L", m))
        if (n - m) in gset:                                   # right segment [n-m, n), t = n-m
            blocks.append(G3[:, :, n - m:, n - m:])
            tags.append(("R", n - m))
        A = torch.cat([b.reshape(S * 3, m, m) for b in blocks], 0).contiguous()
        pending.append((tags, batched_entropies(A, pool)))   # (len(blocks)*S*3,) entropies
        del A
    for tags, res in pending:
        h1, h2 = res.get() if pool is not None else res
        h1 = h1.reshape(len(tags), S, 3)
        h2 = h2.reshape(len(tags), S, 3)
        for k, (side, t) in enumerate(tags):
            I1 = h1[k, :, 0] + h1[k, :, 1] - h1[k, :, 2]
            I2 = h2[k, :, 0] + h2[k, :, 1] - h2[k, :, 2]
            (IL if side == "L" else IR)[t] = (I1, I2)
    a1 = np.empty((S, len(grid)))
    r2 = np.empty((S, len(grid)))
    for i, t in enumerate(grid):
        wt = t * (n - t) / n
        a1[:, i] = np.sqrt(wt) * np.abs(IL[int(t)][0] - IR[int(t)][0])
        r2[:, i] = np.sqrt(wt) * np.abs(IL[int(t)][1] - IR[int(t)][1])
    return a1, r2


def pivoted_cholesky(G, tol=1e-15):
    """Low-rank factor L (n x r) with G - L L' positive semidefinite and every diagonal entry of
    the residual <= tol (greedy diagonal pivoting). For the Gaussian Gram matrices here r is
    about 20 for one rank margin and about 300 for the entrywise product of two margins."""
    n = G.shape[0]
    d = np.diag(G).astype(float).copy()
    L = np.zeros((n, n))
    k = 0
    while k < n:
        i = int(np.argmax(d))
        if d[i] <= tol:
            break
        col = (G[:, i] - L[:, :k] @ L[i, :k]) / np.sqrt(d[i])
        L[:, k] = col
        d -= col * col
        k += 1
    return L[:, :k]


def gram_curves_lowrank(UX, UY, gx, gy, w):
    """The Gram-form curves of `run_matmi_baseline.matmi_curves` (global mode, every row of each
    segment) computed through low-rank factors (CPU).

    A Gram matrix of a segment S is the principal submatrix G[S, S]. With G = L L' + E (pivoted
    Cholesky, E positive semidefinite with diagonal <= 1e-15), G[S, S] = L_S L_S' + E[S, S], and
    the nonzero eigenvalues of L_S L_S' are those of the r x r matrix L_S' L_S; by Weyl's
    inequality every eigenvalue moves by at most trace(E[S, S]) <= |S| 1e-15. The margins
    (r ~ 20) then cost almost nothing, and the joint matrix (r ~ 300) is reduced whenever
    |S| > r; a segment shorter than r is decomposed directly.
    """
    UX = np.atleast_2d(np.asarray(UX, float))
    UY = np.atleast_2d(np.asarray(UY, float))
    gx = np.atleast_1d(np.asarray(gx, float))
    gy = np.atleast_1d(np.asarray(gy, float))
    S, n = UX.shape
    grid = np.arange(w, n - w + 1)
    a1 = np.empty((S, len(grid)))
    r2 = np.empty((S, len(grid)))
    for s in range(S):
        Gx = np.exp(-gx[s] * (UX[s][:, None] - UX[s][None, :]) ** 2)
        Gy = np.exp(-gy[s] * (UY[s][:, None] - UY[s][None, :]) ** 2)
        Gxy = Gx * Gy
        facs = [(pivoted_cholesky(Gx), Gx), (pivoted_cholesky(Gy), Gy), (pivoted_cholesky(Gxy), Gxy)]

        def mi(a, b):
            h1 = np.empty(3)
            h2 = np.empty(3)
            for j, (L, G) in enumerate(facs):
                if b - a <= L.shape[1]:
                    lam = np.linalg.eigvalsh(G[a:b, a:b])
                else:
                    Ls = L[a:b]
                    lam = np.linalg.eigvalsh(Ls.T @ Ls)
                e1, e2 = _entropies(lam[None, :])
                h1[j], h2[j] = e1[0], e2[0]
            return h1[0] + h1[1] - h1[2], h2[0] + h2[1] - h2[2]

        for i, t in enumerate(grid):
            L1, L2 = mi(0, t)
            R1, R2 = mi(t, n)
            wt = np.sqrt(t * (n - t) / n)
            a1[s, i] = wt * abs(L1 - R1)
            r2[s, i] = wt * abs(L2 - R2)
    return a1, r2
