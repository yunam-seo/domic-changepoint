"""Non-Gaussian dependence-change scenarios (G-series) for the non-Gaussian suite (Table 1, lower block).

Every scenario: bivariate (X, Y), independent N(0,1) before tau, a non-Gaussian dependence after tau.
Marginals are N(0,1) in both segments -- exactly via a copula construction (G1-G3), and up to the
Monte Carlo error of a fixed 400,000-draw reference by mapping the post-change Y through its own
marginal CDF estimated from that reference (G4-G6) -- so only the dependence changes. The null process is the pre-change regime (independence).

    G1  symmetric tail dependence, zero correlation: t-copula with rho = 0, nu = level
    G2  upper-tail dependence: Gumbel copula, Kendall tau = level
    G3  lower-tail dependence: Clayton copula, Kendall tau = level
    G4  U-shaped (quadratic), zero correlation: y* = sqrt(1-a^2) e + a (x^2-1)/sqrt(2)
    G5  periodic, zero correlation: y* = sqrt(1-a^2) e + a c cos(pi x)   (c: unit variance)
    G6  asymmetric volatility coupling (leverage), zero correlation: y* = e * exp(b x) / sd
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as tdist

SCENARIOS_NG = {
    "G1": dict(name="tail dependence, zero corr: indep -> t-copula(rho=0, nu)", levels=[8.0, 4.0, 2.0], param="nu"),
    "G2": dict(name="upper-tail: indep -> Gumbel(tau)", levels=[0.2, 0.35, 0.5], param="tau"),
    "G3": dict(name="lower-tail: indep -> Clayton(tau)", levels=[0.2, 0.35, 0.5], param="tau"),
    "G4": dict(name="U-shaped (quadratic), zero corr", levels=[0.5, 0.7, 0.9], param="a"),
    "G5": dict(name="periodic cos(pi x), zero corr", levels=[0.5, 0.7, 0.9], param="a"),
    "G6": dict(name="asymmetric volatility coupling (leverage), zero corr", levels=[0.3, 0.5, 0.8], param="b"),
}
DEFAULT_W_NG = {k: 60 for k in SCENARIOS_NG}
_GID = {k: 500 + int(k[1:]) for k in SCENARIOS_NG}
_REF = {}  # (scen, level) -> sorted MC reference sample of y*


def _rng(base_seed, scen, level_idx, rep, null):
    return np.random.default_rng([base_seed, _GID[scen], level_idx, rep, int(null)])


def _ystar(scen, level, x, e, rng):
    if scen == "G4":
        a = level
        return np.sqrt(1 - a * a) * e + a * (x * x - 1) / np.sqrt(2)
    if scen == "G5":
        a = level
        c = 1.0 / np.sqrt(0.5 * (1 + np.exp(-2 * np.pi ** 2)) - np.exp(-np.pi ** 2))  # 1/sd(cos(pi x))
        m = np.exp(-np.pi ** 2 / 2)                                                  # E cos(pi x)
        return np.sqrt(1 - a * a) * e + a * c * (np.cos(np.pi * x) - m)
    if scen == "G6":
        b = level
        return e * np.exp(b * x - b * b)          # Var = E exp(2bx - 2b^2) = 1
    raise ValueError(scen)


def _to_normal(scen, level, ys):
    key = (scen, float(level))
    if key not in _REF:
        r = np.random.default_rng([987654, _GID[scen], int(round(level * 1000))])
        x, e = r.standard_normal(400_000), r.standard_normal(400_000)
        _REF[key] = np.sort(_ystar(scen, level, x, e, r))
    ref = _REF[key]
    u = (np.searchsorted(ref, ys) + 0.5) / (len(ref) + 1)
    return norm.ppf(np.clip(u, 1e-9, 1 - 1e-9))


def _gumbel_u(rng, k, tau):
    theta = 1.0 / (1.0 - tau)
    al = 1.0 / theta                                  # positive-stable index
    U = rng.uniform(0, np.pi, k)
    W = rng.exponential(1.0, k)
    V = (np.sin(al * U) / np.sin(U) ** (1 / al)) * (np.sin((1 - al) * U) / W) ** ((1 - al) / al)
    E = rng.exponential(1.0, (k, 2))
    return np.exp(-(E / V[:, None]) ** al)


def _clayton_u(rng, k, tau):
    theta = 2 * tau / (1 - tau)
    u1, v = rng.random(k), rng.random(k)
    u2 = (u1 ** (-theta) * (v ** (-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)
    return np.column_stack([u1, u2])


def generate_ng(scen, level, rep, n=600, tau=300, null=False, base_seed=20260825, level_idx=0):
    rng = _rng(base_seed, scen, level_idx, rep, null)
    X = rng.standard_normal((n, 2))
    info = dict(scen=scen, level=level, n=n, tau=tau, null=null)
    if not null:
        k = n - tau
        if scen == "G1":
            nu = level
            z = rng.standard_normal((k, 2))
            w = rng.chisquare(nu, k) / nu
            X[tau:] = norm.ppf(np.clip(tdist.cdf(z / np.sqrt(w)[:, None], nu), 1e-12, 1 - 1e-12))
        elif scen in ("G2", "G3"):
            U = _gumbel_u(rng, k, level) if scen == "G2" else _clayton_u(rng, k, level)
            X[tau:] = norm.ppf(np.clip(U, 1e-12, 1 - 1e-12))
        else:
            x, e = X[tau:, 0], rng.standard_normal(k)
            X[tau:, 1] = _to_normal(scen, level, _ystar(scen, level, x, e, rng))
    return dict(Z=X, x_raw=None, tau=tau, p=2, info=info, blocks=([0], [1]))
