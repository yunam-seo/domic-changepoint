"""Synthetic scenarios: auxiliary S1, S3 and N1-N3 (ablation, Supplementary B.2), article S1-S4 (codes D1-D4), M1, M2, MB (code P1).

Each generator returns dict(Z, x_raw, tau, p, info[, blocks]). Z is the design matrix on which
ALL methods operate. `null=True` keeps the pre-change structure throughout. The argument
`m_embed` of generate() is not used by the scenarios of the article.
"""
from __future__ import annotations

import numpy as np


SCENARIOS = {
    "S1": dict(name="mean shift (d=5)", levels=[0.15, 0.3, 0.5], param="delta"),
    "S3": dict(name="correlation rotation, fixed spectrum (d=5)", levels=[8, 15, 25], param="theta_deg"),
    # nonparametric scenarios: mean 0 and covariance I in BOTH regimes (moment-based methods blind)
    "N1": dict(name="shape: Gaussian -> symmetric 2-Gaussian mixture, same mean/cov (d=2)", levels=[0.6, 0.8, 0.9], param="a"),
    "N2": dict(name="tails: Gaussian -> Student-t scaled to unit cov (d=2)", levels=[5.0, 3.0, 2.5], param="nu"),
    "N3": dict(name="dependence w/o correlation: y = sqrt(1-a^2) e1 + a |x| e2 (d=2)", levels=[0.5, 0.7, 0.9], param="a"),
    # dependence-structure scenarios (blocks X, Y; marginals N(0,1) fixed unless noted)
    "D1": dict(name="Gaussian correlation 0 -> r (linear dependence)", levels=[0.2, 0.35, 0.5], param="r"),
    "D2": dict(name="independent -> dependence w/o correlation (y = sqrt(1-a^2) e1 + a |x| e2)", levels=[0.5, 0.7, 0.9], param="a"),
    "D3": dict(name="copula shape: Gaussian(tau) -> Clayton(tau), same Kendall tau & marginals", levels=[0.3, 0.5, 0.7], param="tau"),
    "D4": dict(name="sign-mixed dependence (rho=+-r mixture, corr 0) -> independent", levels=[0.5, 0.7, 0.85], param="r"),
    "M1": dict(name="SPECIFICITY: marginal variance of X 1 -> s^2, X,Y independent throughout", levels=[1.3, 1.6, 2.0], param="s"),
    "M2": dict(name="SPECIFICITY: margin of Y changes as in D2 (y = sqrt(1-a^2) e1 + a |x'| e2, x' independent of x), X,Y independent throughout", levels=[0.5, 0.7, 0.9], param="a"),
    # E2 multi-change-point scenario (n=1800): segments [indep | |x|-dep(a) | indep | corr(r=0.5)],
    # breaks at 450/900/1350
    "P1": dict(name="multi-CP dependence: indep | |x|-dep(a) | indep | corr(r=0.5)", levels=[0.8, 0.9], param="a"),
}

DEFAULT_W = {"S1": 60, "S3": 60, "N1": 60, "N2": 60, "N3": 60,
             "D1": 60, "D2": 60, "D3": 60, "D4": 60, "M1": 60, "M2": 60, "P1": 60}


def rng_for(base_seed: int, scen: str, level_idx: int, rep: int, null: bool):
    sid = int(scen[1:]) + {"S": 0, "N": 100, "D": 200, "M": 300, "P": 400}[scen[0]]
    return np.random.default_rng([base_seed, sid, level_idx, rep, int(null)])


def _gauss_two_regimes(rng, n, tau, mu1, S1, mu2, S2, null):
    d = len(mu1)
    L1 = np.linalg.cholesky(S1)
    L2 = L1 if null else np.linalg.cholesky(S2)
    E = rng.standard_normal((n, d))
    X = np.empty((n, d))
    X[:tau] = E[:tau] @ L1.T + mu1
    X[tau:] = E[tau:] @ L2.T + (mu1 if null else mu2)
    return X


def givens(d, i, j, theta):
    R = np.eye(d)
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R


def generate(scen: str, level: float, rep: int, n: int = 600, tau: int = 300, null: bool = False,
             base_seed: int = 20260820, level_idx: int = 0, m_embed: int = 10):
    rng = rng_for(base_seed, scen, level_idx, rep, null)
    info = dict(scen=scen, level=level, n=n, tau=tau, null=null)

    if scen == "S1":
        d = 5
        mu2 = np.full(d, level / np.sqrt(d))
        X = _gauss_two_regimes(rng, n, tau, np.zeros(d), np.eye(d), mu2, np.eye(d), null)
        return dict(Z=X, x_raw=None, tau=tau, p=d, info=info)


    if scen == "S3":
        d = 5
        lam = np.array([4.0, 1.0, 2.0, 0.5, 1.0])
        th = np.deg2rad(level)
        Q1 = np.eye(d)
        Q2 = givens(d, 0, 1, th) @ givens(d, 2, 3, th)
        S1 = Q1 @ np.diag(lam) @ Q1.T
        S2 = Q2 @ np.diag(lam) @ Q2.T
        X = _gauss_two_regimes(rng, n, tau, np.zeros(d), S1, np.zeros(d), S2, null)
        return dict(Z=X, x_raw=None, tau=tau, p=d, info=info)


    if scen == "N1":
        a = level
        X = rng.standard_normal((n, 2))
        if not null:
            k = n - tau
            sign = np.where(rng.random(k) < 0.5, 1.0, -1.0)
            # component cov Sigma0 = I - mu mu^T with mu = (a, 0): x ~ sign*a + sqrt(1-a^2) e
            X[tau:, 0] = sign * a + np.sqrt(1 - a * a) * X[tau:, 0]
        return dict(Z=X, x_raw=None, tau=tau, p=2, info=info)

    if scen == "N2":
        nu = level
        X = rng.standard_normal((n, 2))
        if not null:
            k = n - tau
            g = rng.chisquare(nu, size=k) / nu
            X[tau:] = X[tau:] / np.sqrt(g)[:, None] * np.sqrt((nu - 2) / nu)
        return dict(Z=X, x_raw=None, tau=tau, p=2, info=info)

    if scen == "N3":
        a = level
        E = rng.standard_normal((n, 3))
        X = E[:, :2].copy()
        if not null:
            x = E[tau:, 0]
            X[tau:, 1] = np.sqrt(1 - a * a) * E[tau:, 1] + a * np.abs(x) * E[tau:, 2]
            # standardize |x| e2 to variance 1: Var(|x| e2) = E[x^2] = 1 already; mean 0; corr(x, |x|e2)=0
        return dict(Z=X, x_raw=None, tau=tau, p=2, info=info)

    from scipy.stats import norm as _norm

    def _clayton(rng_, k, tau):
        theta = 2 * tau / (1 - tau)
        u1 = rng_.random(k)
        v = rng_.random(k)
        u2 = (u1 ** (-theta) * (v ** (-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)
        return u1, u2

    def _gauss_cop(rng_, k, tau):
        rho = np.sin(np.pi * tau / 2)
        e = rng_.standard_normal((k, 2))
        x = e[:, 0]
        y = rho * e[:, 0] + np.sqrt(1 - rho ** 2) * e[:, 1]
        return _norm.cdf(x), _norm.cdf(y)

    if scen == "D1":
        r = level
        E = rng.standard_normal((n, 2))
        X = E.copy()
        if not null:
            X[tau:, 1] = r * E[tau:, 0] + np.sqrt(1 - r * r) * E[tau:, 1]
        return dict(Z=X, x_raw=None, tau=tau, p=2, info=info, blocks=([0], [1]))

    if scen == "D2":
        a = level
        E = rng.standard_normal((n, 3))
        X = E[:, :2].copy()
        if not null:
            X[tau:, 1] = np.sqrt(1 - a * a) * E[tau:, 1] + a * np.abs(E[tau:, 0]) * E[tau:, 2]
        return dict(Z=X, x_raw=None, tau=tau, p=2, info=info, blocks=([0], [1]))

    if scen == "D3":
        tau_k = level
        u1, u2 = _gauss_cop(rng, n, tau_k)
        if not null:
            v1, v2 = _clayton(rng, n - tau, tau_k)
            u1[tau:], u2[tau:] = v1, v2
        X = np.column_stack([_norm.ppf(np.clip(u1, 1e-9, 1 - 1e-9)), _norm.ppf(np.clip(u2, 1e-9, 1 - 1e-9))])
        return dict(Z=X, x_raw=None, tau=tau, p=2, info=info, blocks=([0], [1]))

    if scen == "D4":
        r = level
        E = rng.standard_normal((n, 2))
        sgn = np.where(rng.random(n) < 0.5, 1.0, -1.0)
        X = E.copy()
        X[:, 1] = sgn * r * E[:, 0] + np.sqrt(1 - r * r) * E[:, 1]  # mixed-sign dependence, corr 0
        if not null:
            X[tau:, 1] = E[tau:, 1]  # independent after tau
        return dict(Z=X, x_raw=None, tau=tau, p=2, info=info, blocks=([0], [1]))

    if scen == "P1":
        a = level
        n4 = 1800
        taus = [450, 900, 1350]
        E = rng.standard_normal((n4, 3))
        X = E[:, :2].copy()
        if not null:
            X[450:900, 1] = np.sqrt(1 - a * a) * E[450:900, 1] + a * np.abs(E[450:900, 0]) * E[450:900, 2]
            r = 0.5
            X[1350:, 1] = r * E[1350:, 0] + np.sqrt(1 - r * r) * E[1350:, 1]
        info["taus"] = taus if not null else []
        return dict(Z=X, x_raw=None, tau=taus[0], p=2, info=info, blocks=([0], [1]))

    if scen == "M1":
        s_ = level
        E = rng.standard_normal((n, 2))
        X = E.copy()
        if not null:
            X[tau:, 0] *= s_
        return dict(Z=X, x_raw=None, tau=tau, p=2, info=info, blocks=([0], [1]))

    if scen == "M2":
        # Marginal-shape control for D2: after tau, Y has exactly the D2 post-change margin, but the
        # multiplier |x'| is an independent draw, so X and Y stay independent throughout.
        a = level
        E = rng.standard_normal((n, 4))
        X = E[:, :2].copy()
        if not null:
            X[tau:, 1] = np.sqrt(1 - a * a) * E[tau:, 1] + a * np.abs(E[tau:, 3]) * E[tau:, 2]
        return dict(Z=X, x_raw=None, tau=tau, p=2, info=info, blocks=([0], [1]))

    raise ValueError(scen)
