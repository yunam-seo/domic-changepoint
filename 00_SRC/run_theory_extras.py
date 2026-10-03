#!/usr/bin/env python
"""Numbers quoted in Sections 3.5, 4.1 and 6.2, in Supplementary Sections A.6 and A.10, and the left
panel of Figure B.2.

X1  Degeneracy of P2 at the independence null: the influence function h vanishes identically, so
    sigma^2 = 0, and the correct normalization is m rather than sqrt(m).
X2  Sign of the bias constant c of Lemma 4(iv) as dependence strengthens: c > 0 near independence,
    changing sign between Gaussian correlation 0.95 and 0.99.
X3  Specificity under a marginal change superimposed on unchanged dependence (the interaction the
    M1 design cannot expose, since there X and Y are independent throughout).

Writes 04_DAOU/EXPERIMENT/theory_extras/{results.json, records_extras_reps.csv}
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy.linalg import logm

SRC = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.domi import ranks01, unit_rff  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "theory_extras")
D = 8


def ent(M):
    l = np.clip(np.linalg.eigvalsh(M), 1e-300, None); l = l[l > 1e-14]
    return float(-(l * np.log(l)).sum())


def feats(x, y):
    return (unit_rff(ranks01(np.asarray(x, float)[:, None]), D, 2026),
            unit_rff(ranks01(np.asarray(y, float)[:, None]), D, 2027))


def domi(x, y):
    fx, fy = feats(x, y)
    P = np.einsum("ti,tj->tij", fx, fy).reshape(len(fx), -1)
    return ent(fx.T @ fx / len(fx)) + ent(fy.T @ fy / len(fy)) - ent(P.T @ P / len(P))


REC = []


def x1_influence_and_rate(rng):
    global REC
    n = 200_000
    fx, fy = feats(rng.standard_normal(n), rng.standard_normal(n))
    MX, MY = fx.T @ fx / n, fy.T @ fy / n
    lMX, lMY, lMXY = logm(MX).real, logm(MY).real, logm(np.kron(MX, MY)).real
    k = 5000
    psi = np.einsum("ti,tj->tij", fx[:k], fy[:k]).reshape(k, D * D)
    h = np.array([-np.trace(lMX @ np.outer(fx[t], fx[t])) - np.trace(lMY @ np.outer(fy[t], fy[t]))
                  + np.trace(lMXY @ np.outer(psi[t], psi[t])) for t in range(k)])
    rate = {}
    for m in (150, 600, 2400):
        v = np.array([domi(rng.standard_normal(m), rng.standard_normal(m)) for _ in range(600)])
        REC += [dict(part="X1_rate", config=f"m{m}", rep=i, value=float(x)) for i, x in enumerate(v)]
        rate[m] = dict(m_times_mean=round(float(m * v.mean()), 3), m_times_sd=round(float(m * v.std()), 3),
                       sqrtm_times_sd=round(float(np.sqrt(m) * v.std()), 5),
                       skew=round(float(((v - v.mean()) ** 3).mean() / v.std() ** 3), 3))
    return dict(max_abs_h=float(np.abs(h).max()), sigma2=float(h.var()), rate=rate)


def x2_bias_sign(rng):
    def beta(F):
        N = len(F); M = F.T @ F / N
        lam, U = np.linalg.eigh(M); lam = np.clip(lam, 1e-14, None)
        Q = F @ U; Q2 = Q * Q
        Var = np.clip((Q2.T @ Q2) / N - ((Q.T @ Q) / N) ** 2, 0, None)
        a, b = lam[:, None], lam[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            L = (a - b) / (np.log(a) - np.log(b))
        L = np.where(np.abs(a - b) < 1e-13, a * np.ones_like(b), L)
        return 0.5 * float((Var / L).sum())
    N = 150_000; out = {}
    for tag, r in (("independent", 0.0), ("r=0.7", 0.7), ("r=0.95", 0.95), ("r=0.99", 0.99), ("Y=X", None)):
        a = rng.standard_normal(N)
        b = a.copy() if r is None else r * a + np.sqrt(1 - r * r) * rng.standard_normal(N)
        fa, fb = feats(a, b)
        ps = np.einsum("ti,tj->tij", fa, fb).reshape(N, D * D)
        out[tag] = round(beta(ps) - beta(fa) - beta(fb), 4)
    return out


def x3_drift_specificity():
    global REC
    n, tau, REPS = 600, 300, 400

    def contrast(x, y):
        fx, fy = feats(x, y)                       # ranks pooled over the WHOLE window

        def I(sl):
            a, b = fx[sl], fy[sl]
            P = np.einsum("ti,tj->tij", a, b).reshape(len(a), -1)
            return ent(a.T @ a / len(a)) + ent(b.T @ b / len(b)) - ent(P.T @ P / len(P))
        return abs(I(slice(0, tau)) - I(slice(tau, n)))

    def gen(rng, r, drift):
        x = rng.standard_normal(n)
        y = r * x + np.sqrt(1 - r * r) * rng.standard_normal(n) if r > 0 else rng.standard_normal(n)
        x = x.copy()
        if drift == "scale8":
            x[tau:] *= 8
        elif drift == "tanh":
            x[tau:] = np.tanh(x[tau:])
        return x, y

    out = {}
    for r, lab in ((0.7, "dependence_r0.7"), (0.0, "independent_M1_control")):
        rng = np.random.default_rng(21)
        thr = float(np.quantile([contrast(*gen(rng, r, "none")) for _ in range(REPS)], 0.95))
        d = {}
        for drift in ("scale8", "tanh"):
            rng = np.random.default_rng(22)
            v = np.array([contrast(*gen(rng, r, drift)) for _ in range(REPS)])
            REC += [dict(part="X3_drift", config=f"{lab}|{drift}", rep=i, value=float(x))
                    for i, x in enumerate(v)]
            d[drift] = round(float(np.mean(v > thr)), 3)
        out[lab] = d
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20260906)
    res = dict(config=dict(D=D, note="paper S1-S4 == code D1-D4"),
               X1_independence_degeneracy=x1_influence_and_rate(rng),
               X2_bias_sign=x2_bias_sign(rng),
               X3_drift_specificity=x3_drift_specificity())
    import csv as _csv
    with open(os.path.join(OUT, "records_extras_reps.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["part", "config", "rep", "value"])
        w.writeheader(); w.writerows(REC)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
