"""BMCTC: moving-window matrix-based Renyi total correlation + Bernaola-Galvan segmentation.

Our reimplementation of the method of Qian, Jin, Wang, Liu, Yang and Wang (2024, Stoch. Environ.
Res. Risk Assess. 38, 467-488; reference [4] of the article), written from the description in
[4] (its Section 2) and the standard definitions, without access to the original code:

  1. Moving-window total correlation. For every window of L consecutive observations the
     matrix-based Renyi alpha-entropy (Sanchez Giraldo, Rao and Principe, 2015)
         S_alpha(A) = (1 - alpha)^{-1} log2 tr(A^alpha),   A = K / tr(K),
     is computed from the Gaussian Gram matrix K of each variable; the joint entropy uses the
     normalized Hadamard product A_X o A_Y / tr(A_X o A_Y) (Yu et al., 2020), and the total
     correlation of two variables is TC = S(A_X) + S(A_Y) - S(A_XY) (their mutual information).
     The window value is attached to the window center, giving a TC sequence of length n - L + 1.
  2. Bernaola-Galvan (BG) segmentation of that sequence (Bernaola-Galvan et al., 2001, Phys. Rev.
     Lett. 87, 168105): at every admissible split j the Student-type statistic
         T(j) = |mean_L - mean_R| / s_D,
         s_D = sqrt( ((N_L-1) s_L^2 + (N_R-1) s_R^2) / (N_L+N_R-2) ) * sqrt(1/N_L + 1/N_R),
     is computed; the maximum T_max is assessed by the BG max-t approximation
         P(T_max) = { 1 - I_{nu/(nu+T_max^2)}(delta*nu, delta) }^eta,
         nu = N - 2, delta = 0.40, eta = 4.19 ln N - 11.54,
     (I = regularized incomplete beta), the segment is split if P > P0 (P0 = 0.95), and the
     procedure recurses on both parts while each part keeps at least l0 points.

Implementation choices that the description in [4] leaves open (our reimplementation):
  * alpha = 1.01 (primary; the near-Shannon order recommended by Yu et al. and the closest to the
    Gram form of DOMI at alpha = 1) and alpha = 2 (secondary, reported alongside).
  * Gaussian kernel exp(-(u-v)^2 / (2 sigma^2)) on each variable z-scored within the window,
    sigma by Silverman's rule sigma = 1.06 * L^(-1/5) (unit variance after z-scoring, d = 1).
  * Window length L = 60 (our minimum segment length w), step 1, raw (not rank) values.
  * Minimum BG segment l0 = 30 TC points; for n = 600 the admissible splits of the TC sequence are
    times 60..541 (split j <-> time j + L/2); run_bmctc.py drops the last one to use the candidate
    grid 60..540.
  * The BG probability is evaluated on the log scale, q = -log P = -eta * log1p(-I), so that
    "P > P0" is "q < -log P0"; q is kept for calibration.
  * Each variable block is univariate (true of every design this is run on).
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import betainc

L_WIN = 60
ALPHAS = (1.01, 2.0)
L0 = 30
BG_DELTA, BG_A, BG_B = 0.40, 4.19, 11.54


# ---------------------------------------------------------------- moving-window total correlation
def _gram_windows(W, sigma):
    """W: (n_windows, L) z-scored windows -> (n_windows, L, L) Gaussian Gram matrices."""
    d = W[:, :, None] - W[:, None, :]
    return np.exp(-d * d / (2.0 * sigma * sigma))


def _renyi(lam, alpha):
    """Row-wise Renyi alpha-entropy (bits) of unit-trace spectra lam (n_windows, L)."""
    lam = np.clip(lam, 0.0, None)
    lam = lam / lam.sum(1, keepdims=True)
    if abs(alpha - 1.0) < 1e-12:
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.nansum(np.where(lam > 0, lam * np.log2(lam), 0.0), 1)
    return np.log2((lam ** alpha).sum(1)) / (1.0 - alpha)


def tc_series(x, y, L=L_WIN, alphas=ALPHAS, chunk=256):
    """Moving-window matrix-based Renyi total correlation of two univariate series.

    Returns (centers, {alpha: TC array}) with centers[i] = i + L // 2 (window [i, i+L))."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    Wx = sliding_window_view(x, L)
    Wy = sliding_window_view(y, L)
    sigma = 1.06 * L ** (-0.2)
    out = {a: np.empty(len(Wx)) for a in alphas}
    for s in range(0, len(Wx), chunk):
        zx = Wx[s:s + chunk]
        zy = Wy[s:s + chunk]
        zx = (zx - zx.mean(1, keepdims=True)) / zx.std(1, ddof=1, keepdims=True)
        zy = (zy - zy.mean(1, keepdims=True)) / zy.std(1, ddof=1, keepdims=True)
        Kx = _gram_windows(zx, sigma)
        Ky = _gram_windows(zy, sigma)
        lx = np.linalg.eigvalsh(Kx)            # tr(K) = L; normalization inside _renyi
        ly = np.linalg.eigvalsh(Ky)
        lxy = np.linalg.eigvalsh(Kx * Ky)
        for a in alphas:
            out[a][s:s + chunk] = _renyi(lx, a) + _renyi(ly, a) - _renyi(lxy, a)
    return np.arange(len(Wx)) + L // 2, out


# ---------------------------------------------------------------- Bernaola-Galvan
def bg_tstat(s, l0=L0):
    """T(j) for every split j = l0..N-l0 of the sequence s (left = s[:j]). Returns (js, T)."""
    s = np.asarray(s, float)
    N = len(s)
    js = np.arange(l0, N - l0 + 1)
    if len(js) == 0:
        return js, np.array([])
    c1 = np.concatenate([[0.0], np.cumsum(s)])
    c2 = np.concatenate([[0.0], np.cumsum(s * s)])
    nL, nR = js.astype(float), (N - js).astype(float)
    mL, mR = c1[js] / nL, (c1[N] - c1[js]) / nR
    ssL = np.clip(c2[js] - nL * mL ** 2, 0, None)       # (N_L - 1) s_L^2
    ssR = np.clip(c2[N] - c2[js] - nR * mR ** 2, 0, None)
    sp = np.sqrt((ssL + ssR) / (N - 2)) * np.sqrt(1 / nL + 1 / nR)
    return js, np.abs(mL - mR) / np.where(sp > 0, sp, np.inf)


def bg_q(tmax, N):
    """q = -log P(T_max) under the BG max-t approximation (small q = significant)."""
    nu = N - 2.0
    eta = max(BG_A * np.log(N) - BG_B, 1.0)
    I = betainc(BG_DELTA * nu, BG_DELTA, nu / (nu + tmax * tmax))
    return float(-eta * np.log1p(-min(I, 1.0 - 1e-16)))


def bg_top(s, l0=L0):
    """Top-level BG split of s: (j_hat, T_max, q)."""
    js, T = bg_tstat(s, l0)
    if len(T) == 0:
        return None, np.nan, np.inf
    i = int(np.argmax(T))
    return int(js[i]), float(T[i]), bg_q(T[i], len(s))


def bg_segment(s, q0, l0=L0, offset=0):
    """Recursive BG segmentation: split while q < q0 (i.e. P > P0 = exp(-q0)). Returns sorted
    split indices of the sequence (absolute, left count)."""
    s = np.asarray(s, float)
    if len(s) < 2 * l0:
        return []
    j, _, q = bg_top(s, l0)
    if j is None or not q < q0:
        return []
    return (bg_segment(s[:j], q0, l0, offset) + [offset + j]
            + bg_segment(s[j:], q0, l0, offset + j))


Q_PUBLISHED = float(-np.log(0.95))       # P0 = 0.95
