"""Asymptotic (permutation-free) calibration of the DOMI difference scan at the independence boundary.

Purpose
-------
Proposition P6, its process-level Corollary, Proposition P6-R and the rank Corollary
(Supplementary Section A.12) give the joint limit, over split fractions
lambda in [eps, 1-eps], of the two segment DOMI plug-ins under the null

    H0^ind : the pairs (X_t, Y_t) are i.i.d. with independent components and continuous margins
             ("no change, and independence throughout").

This module turns that limit into a test.  It never permutes and never simulates length-n series:
the critical value is a quantile of a Brownian functional whose ingredients are read off the
observed features.

The statistic
-------------
Deployed inputs (Section 4.1): pseudo-observations rank/(n+1) over the whole series, D unit-norm
random Fourier features per block (bandwidth by the median heuristic on the ranks, feature seeds
2026 and 2027, exactly as ``dots.domi.DOMIContext``).  On a fixed grid of split fractions
lambda_j = t_j / n (default: t_j = round(n * l), l = 0.100, 0.105, ..., 0.900; 161 points),
let I_L(t) = DOMI of [0, t) and I_R(t) = DOMI of [t, n).  The deployed weighted difference of
Section 4.2 is Q(t) = sqrt(t(n-t)/n) |I_L - I_R|.  At the boundary I_L, I_R = O_p(1/n), so
Q = O_p(n^{-1/2}) is degenerate; the non-degenerate normalization is

    R_n(lambda) := sqrt(n) Q(t) = sqrt(lambda (1-lambda)) * | n I_L(lambda) - n I_R(lambda) |,

    T_raw := max_j R_n(lambda_j)                       (unstudentized scan, the primary statistic)
    T_std := max_j ( R_n(lambda_j) - mu(lambda_j) ) / sd(lambda_j)   (limit-studentized scan)

where mu, sd are the mean and standard deviation of the LIMIT process R(lambda) (below).  Because
mu, sd are fixed continuous functions (given the feature map), T_std is also a continuous
functional of the pair (n I_L, n I_R) and is covered by the Corollary; it is NOT the deployed
studentized statistic of Algorithm 1, whose moments come from the K+1 permutation curves.

The limit
---------
By the Corollary / rank Corollary, (n I_L, n I_R) => (lambda^{-2} Phi(G(lambda)),
(1-lambda)^{-2} Phi(G(1) - G(lambda))) with
Phi(M) = (1/2)[q_rhobar(M) - q_rhoX(Tr_Y M) - q_rhoY(Tr_X M)], rhobar = rhoX (x) rhoY and
q_rho the reciprocal-logarithmic-mean quadratic form.

Reduction (exact algebra).  Write any traceless
symmetric M on R^D (x) R^D as M = A (x) rhoY + rhoX (x) B + Gamma with A = Tr_Y M, B = Tr_X M.
Then Tr_Y Gamma = Tr_X Gamma = 0 and, because k(a mu, b mu) = k(a, b)/mu,
    q_rhobar(A (x) rhoY) = q_rhoX(A),  q_rhobar(rhoX (x) B) = q_rhoY(B),
and every cross term of q_rhobar between the three pieces vanishes.  Hence

    Phi(M) = (1/2) q_rhobar(Gamma):   Phi ignores the marginal directions entirely.

Consequence for ranks.  Writing each summand as
psi psi^T - rhobar = a_t (x) rhoY + rhoX (x) b_t + a_t (x) b_t with a_t = phi_X phi_X^T - rhoX,
b_t = phi_Y phi_Y^T - rhoY, the rank corrections of P6-R (empirical-copula process; at the
independence copula the correction terms are -v alpha(u) - u beta(v)) act only on the marginal
partial sums (they turn the marginal Brownian motions into bridges: whole-series ranks make the
whole-series marginal moments deterministic).  Those directions are annihilated by Phi.  The
interaction partial sums n^{-1/2} sum_{t <= lambda n} a_t (x) b_t converge, for oracle margins and
for ranks alike (combinatorial CLT for two independent uniform permutations), to a Brownian
motion with covariance Sigma_a (x) Sigma_b, Sigma_a = Cov(vec a_t), Sigma_b = Cov(vec b_t).
So the limit of the deployed rank statistic equals the oracle-margin limit, and it is

    Phi(G(lambda)) = sum_k w_k beta_k(lambda)^2,     Phi(G(1)-G(lambda)) = sum_k w_k (beta_k(1)-beta_k(lambda))^2,

with beta_k independent standard Brownian motions and w_k >= 0 the eigenvalues of the quadratic
form (1/2) q_rhobar(L_a Z L_b^T) in the i.i.d. N(0,1) matrix Z (L_a L_a^T = Sigma_a, likewise b),
all in the eigenbases of rhoX and rhoY.  sum_k w_k = c, the bias constant of Lemma 4(iv).

What is estimated, and why the estimate is valid under H0^ind.  rhoX, rhoY, Sigma_a, Sigma_b are
replaced by the whole-series sample moments of the observed rank features.  With ranks these are
deterministic Riemann sums over the grid {k/(n+1)} (given the bandwidth), which converge to the
population values at rate O(1/n): the plug-in is consistent, and in fact distribution-free.
Directions of rhoX / rhoY with eigenvalue <= tau times the largest eigenvalue are dropped, as described
in Supplementary Section B.16 (default tau = 1e-9); ``limit_weights`` also reports the resulting bias
constant c.

Simulation.  The Brownian motions are simulated exactly at the grid (independent Gaussian
increments over [0, lambda_1], [lambda_1, lambda_2], ..., [lambda_G, 1]); weights are truncated to
the leading ones carrying a fraction (1 - 1e-6) of sum_k w_k.  The critical value is the
(1-alpha) empirical quantile of the simulated maximum over ``n_draw`` draws (default 2000).

Exact Monte Carlo reference.  Under H0^ind with continuous margins the rank vectors of X and Y
are independent uniform permutations, so the null law of T_raw depends only on (n, D, feature
seeds, grid): ``exact_mc_null`` simulates it with length-n draws.  That route is exact but costs
n-length curve evaluations per draw; the asymptotic route costs O(n D^4) once for the moments
plus a simulation whose cost does not depend on n.

Scope.  This calibrates the test of H0^ind only.  It does NOT calibrate a change between two
dependent regimes (away from the boundary the limit is P2's Gaussian one, not proved for ranks),
and the deployed studentized scan (moments from the permutation curves) remains
permutation-calibrated.  A rejection means "not (i.i.d. and independent throughout)": a change in
the margins alone is outside H0^ind; with whole-series ranks the test is exactly invariant to a
monotone transformation applied to the whole window, but not to a marginal change part-way.
"""
from __future__ import annotations

import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from dots.domi import ranks01, unit_rff  # noqa: E402
from dots import qdiv as Q  # noqa: E402
from dots.perm import ge  # noqa: E402

D_DEFAULT = 8
SEED = 2026
TAU_SUPPORT = 1e-9


# ---------------------------------------------------------------- grid and features
def fraction_grid(n: int, eps: float = 0.1, step: float = 0.005) -> np.ndarray:
    """Candidate split points t_j = round(n * l_j), l_j = eps, eps+step, ..., 1-eps (unique, sorted)."""
    ls = np.round(np.arange(eps, 1 - eps + 1e-12, step), 10)
    t = np.unique(np.round(n * ls).astype(int))
    return t[(t >= 2) & (t <= n - 2)]


def features(X, Y, D: int = D_DEFAULT, seed: int = SEED):
    """Deployed features: whole-series ranks/(n+1), unit-norm RFF (median-heuristic bandwidth).

    Identical to ``DOMIContext``: FX = unit_rff(ranks01(X), D, seed), FY with seed+1.
    """
    return unit_rff(ranks01(X), D, seed), unit_rff(ranks01(Y), D, seed + 1)


def features_from_ranks(UX, UY, D: int = D_DEFAULT, seed: int = SEED):
    """Same as ``features`` for inputs that are already pseudo-observations (n,1)."""
    return unit_rff(UX, D, seed), unit_rff(UY, D, seed + 1)


# ---------------------------------------------------------------- the scan curve
def _entropy(M):
    tr = np.trace(M)
    return Q.vn_entropy(M / tr)


def segment_curves(FX, FY, grid):
    """n*I_L and n*I_R at every t in ``grid`` from block sums (memory O(G D^4), time O(n D^4)).

    Numerically the same quantities as ``DOMIContext.domi(0, t)`` / ``domi(t, n)`` (trace-normalized
    second moments, eigenvalues clipped at 0, DOMI clipped at 0).
    """
    n, D = FX.shape
    J = np.einsum("ti,tj->tij", FX, FY).reshape(n, D * D)
    edges = np.concatenate([[0], grid, [n]])
    SJ = np.zeros((len(edges), D * D, D * D))
    SX = np.zeros((len(edges), D, D))
    SY = np.zeros((len(edges), D, D))
    for k in range(1, len(edges)):
        a, b = edges[k - 1], edges[k]
        SJ[k] = SJ[k - 1] + J[a:b].T @ J[a:b]
        SX[k] = SX[k - 1] + FX[a:b].T @ FX[a:b]
        SY[k] = SY[k - 1] + FY[a:b].T @ FY[a:b]
    nIL = np.empty(len(grid))
    nIR = np.empty(len(grid))
    for j in range(len(grid)):
        k = j + 1
        IL = _entropy(SX[k]) + _entropy(SY[k]) - _entropy(SJ[k])
        IR = (_entropy(SX[-1] - SX[k]) + _entropy(SY[-1] - SY[k]) - _entropy(SJ[-1] - SJ[k]))
        nIL[j], nIR[j] = n * max(IL, 0.0), n * max(IR, 0.0)
    return nIL, nIR


def scan_process(nIL, nIR, lam):
    """R_n(lambda) = sqrt(lambda(1-lambda)) |n I_L - n I_R|  (= sqrt(n) * deployed Q(t))."""
    return np.sqrt(lam * (1 - lam)) * np.abs(nIL - nIR)


# ---------------------------------------------------------------- the limit
def _kmat(lam):
    """Reciprocal-logarithmic-mean kernel k(a,b) = (log a - log b)/(a - b), k(a,a) = 1/a."""
    L = lam[:, None] - lam[None, :]
    G = np.log(lam)[:, None] - np.log(lam)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(np.abs(L) > 1e-12 * np.maximum(lam[:, None], lam[None, :]),
                     G / np.where(L == 0, 1, L), 2.0 / (lam[:, None] + lam[None, :]))
    return K


def _marginal(F, tau):
    n = len(F)
    rho = F.T @ F / n
    lam, W = np.linalg.eigh(rho)
    keep = lam > tau * lam.max()
    lam, W = lam[keep], W[:, keep]
    Ft = F @ W                                     # coordinates in the eigenbasis (support)
    s = len(lam)
    a = np.einsum("ti,tj->tij", Ft, Ft).reshape(n, s * s) - np.diag(lam).reshape(1, -1)
    Sig = a.T @ a / n                              # Cov(vec a_t) (mean of a_t is 0 exactly)
    ev, V = np.linalg.eigh(Sig)
    pos = ev > 1e-14 * ev.max()
    La = V[:, pos] * np.sqrt(ev[pos])              # Sigma = La La^T
    return lam, La


def limit_weights(FX, FY, tau: float = TAU_SUPPORT):
    """Weights w_k >= 0 of the limit Phi(G(1)) = sum_k w_k Z_k^2 from the observed features.

    Returns (w sorted descending, info) with info = dict(c=sum w, support dims, ranks).
    """
    lx, La = _marginal(FX, tau)
    ly, Lb = _marginal(FY, tau)
    sx, sy = len(lx), len(ly)
    # Kw[(i,j),(b,c)] = k(lx_i ly_b, lx_j ly_c) on the product eigenbasis
    prod = np.multiply.outer(lx, ly).reshape(-1)             # index (i,b)
    K = _kmat(prod).reshape(sx, sy, sx, sy)                  # [i,b,j,c]
    Kw = K.transpose(0, 2, 1, 3).reshape(sx * sx, sy * sy)   # [(i,j),(b,c)]
    ra, rb = La.shape[1], Lb.shape[1]
    Ta = np.einsum("pr,pR->prR", La, La).reshape(sx * sx, ra * ra)
    Tb = np.einsum("qs,qS->qsS", Lb, Lb).reshape(sy * sy, rb * rb)
    M = (Ta.T @ Kw @ Tb).reshape(ra, ra, rb, rb).transpose(0, 2, 1, 3).reshape(ra * rb, ra * rb)
    M = 0.5 * (M + M.T)
    w = np.linalg.eigvalsh(0.5 * M)[::-1]
    w = np.clip(w, 0.0, None)
    return w, dict(c=float(w.sum()), support=(sx, sy), ranks=(ra, rb), n_weights=len(w))


def simulate_limit(w, lam, n_draw: int = 2000, seed: int = 0, mass: float = 1 - 1e-6, chunk: int = 250):
    """Draws of the limit process R(lambda_j) = sqrt(l(1-l)) |l^-2 Phi_L - (1-l)^-2 Phi_R|.

    Phi_L(l) = sum_k w_k beta_k(l)^2, Phi_R(l) = sum_k w_k (beta_k(1) - beta_k(l))^2 with beta_k
    independent standard Brownian motions, simulated exactly at the grid.  Returns an array
    (n_draw, G).  Weights beyond the leading fraction ``mass`` of sum w are dropped.
    """
    cw = np.cumsum(w) / w.sum()
    J = int(np.searchsorted(cw, mass) + 1)
    wk = w[:J]
    dl = np.diff(np.concatenate([[0.0], lam, [1.0]]))
    rng = np.random.default_rng(seed)
    out = np.empty((n_draw, len(lam)))
    wt = np.sqrt(lam * (1 - lam))
    for s in range(0, n_draw, chunk):
        m = min(chunk, n_draw - s)
        inc = rng.standard_normal((m, len(dl), J)) * np.sqrt(dl)[None, :, None]
        Bm = np.cumsum(inc, axis=1)                    # beta at lam_1..lam_G, 1
        BL = Bm[:, :-1, :]
        BR = Bm[:, -1:, :] - BL
        PL = (BL * BL) @ wk
        PR = (BR * BR) @ wk
        out[s:s + m] = wt * np.abs(PL / lam ** 2 - PR / (1 - lam) ** 2)
    return out


# ---------------------------------------------------------------- the test
def asymptotic_test(X, Y, D: int = D_DEFAULT, alpha: float = 0.05, n_draw: int = 2000,
                    seed: int = 0, grid=None, tau: float = TAU_SUPPORT, return_curve=False):
    """Permutation-free test of H0^ind by the limit law of the Corollary (A.12).

    Returns dict with T_raw, crit_raw, p_raw (Monte Carlo p-value against the limit draws),
    T_std, crit_std, p_std, tau_hat_raw, tau_hat_std, c (bias constant), n_weights_used.
    """
    n = len(X)
    grid = fraction_grid(n) if grid is None else grid
    lam = grid / n
    FX, FY = features(X, Y, D)
    nIL, nIR = segment_curves(FX, FY, grid)
    R = scan_process(nIL, nIR, lam)
    w, info = limit_weights(FX, FY, tau)
    sims = simulate_limit(w, lam, n_draw=n_draw, seed=seed)
    out = dict(info)
    out.update(_decide(R, sims, alpha, grid))
    if return_curve:
        out["R"] = R
    return out


def _decide(R, sims, alpha, grid):
    mu, sd = sims.mean(0), sims.std(0) + 1e-12
    Traw_sim = sims.max(1)
    Tstd_sim = ((sims - mu) / sd).max(1)
    Traw = float(R.max())
    Zs = (R - mu) / sd
    Tstd = float(Zs.max())
    return dict(T_raw=Traw, crit_raw=float(np.quantile(Traw_sim, 1 - alpha)),
                p_raw=float((1 + ge(Traw_sim, Traw).sum()) / (len(Traw_sim) + 1)),
                T_std=Tstd, crit_std=float(np.quantile(Tstd_sim, 1 - alpha)),
                p_std=float((1 + ge(Tstd_sim, Tstd).sum()) / (len(Tstd_sim) + 1)),
                tau_hat_raw=int(grid[int(np.argmax(R))]), tau_hat_std=int(grid[int(np.argmax(Zs))]))


# ---------------------------------------------------------------- exact Monte Carlo reference
def exact_mc_draw(n, rng, D: int = D_DEFAULT, grid=None, oracle: bool = False):
    """One draw of the scan process R_n under H0^ind (ranks of independent uniforms, deployed features).

    With ``oracle=True`` the features are applied to the uniforms themselves (P6 setting).
    Also returns n*I-hat of the whole series (for the bias constant c).
    """
    grid = fraction_grid(n) if grid is None else grid
    U, V = rng.random((n, 1)), rng.random((n, 1))
    if not oracle:
        U, V = ranks01(U), ranks01(V)
    FX, FY = features_from_ranks(U, V, D)
    nIL, nIR = segment_curves(FX, FY, grid)
    J = np.einsum("ti,tj->tij", FX, FY).reshape(n, -1)
    Iall = _entropy(FX.T @ FX) + _entropy(FY.T @ FY) - _entropy(J.T @ J)
    return scan_process(nIL, nIR, grid / n), n * max(Iall, 0.0)


# ---------------------------------------------------------------- permutation test (Section 4.3)
def permutation_test(X, Y, D: int = D_DEFAULT, K: int = 99, alpha: float = 0.05, seed: int = 0, grid=None):
    """Pair-permutation test on the same grid (Algorithm 1 with b = 1), two variants:

    'perm_std' : deployed rule -- symmetric studentization over the K+1 curves, p-value (9);
    'perm_raw' : the unstudentized T_raw with the same permutation p-value.
    Each replica recomputes ranks, bandwidth and features from the permuted pairs, as deployed.
    """
    n = len(X)
    grid = fraction_grid(n) if grid is None else grid
    lam = grid / n
    rng = np.random.default_rng(seed)
    curves = []
    for k in range(K + 1):
        idx = np.arange(n) if k == 0 else rng.permutation(n)
        FX, FY = features(X[idx], Y[idx], D)
        curves.append(scan_process(*segment_curves(FX, FY, grid), lam))
    A = np.array(curves)
    mu, sd = A.mean(0), A.std(0) + 1e-12
    Ts = ((A - mu) / sd).max(1)
    Tr = A.max(1)
    p_std = (1 + ge(Ts[1:], Ts[0]).sum()) / (K + 1)
    p_raw = (1 + ge(Tr[1:], Tr[0]).sum()) / (K + 1)
    return dict(p_perm_std=float(p_std), p_perm_raw=float(p_raw),
                tau_hat_perm_std=int(grid[int(np.argmax((A[0] - mu) / sd))]),
                tau_hat_perm_raw=int(grid[int(np.argmax(A[0]))]))
