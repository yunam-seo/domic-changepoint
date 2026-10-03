#!/usr/bin/env python
"""Computed numbers the proofs of Supplementary Section A rest on: [NV-1] and [NV-4].

NV-1  A.7, condition (A3) for the S2 family: the population change-point criterion
       sqrt(s(1-s)) |I_left(s) - I_right(s)| has its unique maximum at the true split theta = 0.5
       (population moments from 2 x 10^5 samples, grid step 0.01).
NV-4  A.10(iv): the closed-form plug-in bias constant c at independence (D = 8, rank features,
       deployed draws 2026/2027), against the measured m E[I_hat] at m = 200 and 800
       (200 replicates each).

Run:  python 00_SRC/proof_numbers.py  ->  04_DAOU/EXPERIMENT/theory/proof_numbers.json

The NV-4 replicates use seed 4242 after a fixed burn-in of that stream (advance_stream()), so the
printed values (c = 3.070; 3.117 and 3.122) are reproduced exactly.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.domi import ranks01, unit_rff  # noqa: E402
from dots.synth import generate  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "theory", "proof_numbers.json")
D = 8


def entropy(M, floor=1e-15):
    lam = np.clip(np.linalg.eigvalsh(M), 1e-300, None)
    lam = lam[lam > floor]
    return float(-(lam * np.log(lam)).sum())


def nv1():
    big = generate("D2", 0.9, 1, n=200000, tau=100000)
    Zb = big["Z"]
    UX, UY = ranks01(Zb[:, [0]]), ranks01(Zb[:, [1]])
    FX, FY = unit_rff(UX, D, 2026), unit_rff(UY, D, 2027)
    PSI = np.einsum("ti,tj->tij", FX, FY).reshape(len(FX), -1)
    h = 100000
    M1, M2 = PSI[:h].T @ PSI[:h] / h, PSI[h:].T @ PSI[h:] / h
    X1, X2 = FX[:h].T @ FX[:h] / h, FX[h:].T @ FX[h:] / h
    Y1, Y2 = FY[:h].T @ FY[:h] / h, FY[h:].T @ FY[h:] / h
    theta, ss = 0.5, np.linspace(0.1, 0.9, 81)

    def domi(MX, MY, MXY):
        return entropy(MX) + entropy(MY) - entropy(MXY)

    curve = []
    for s in ss:
        a = min(s, theta) / s
        b = max(0.0, (theta - s) / (1 - s))
        L = domi(a * X1 + (1 - a) * X2, a * Y1 + (1 - a) * Y2, a * M1 + (1 - a) * M2)
        R = domi(b * X1 + (1 - b) * X2, b * Y1 + (1 - b) * Y2, b * M1 + (1 - b) * M2)
        curve.append(np.sqrt(s * (1 - s)) * abs(L - R))
    curve = np.array(curve)
    imax = int(np.argmax(curve))
    outside = curve[np.abs(ss - theta) > 0.1]
    return dict(argmax_s=float(ss[imax]), theta=theta, peak=float(curve[imax]),
                max_outside_pm_0_1=float(outside.max()), grid_step=0.01, samples=200000)


def domi_rank(x, y):
    fx = unit_rff(ranks01(np.asarray(x, float)[:, None]), D, 2026)
    fy = unit_rff(ranks01(np.asarray(y, float)[:, None]), D, 2027)
    P = np.einsum("ti,tj->tij", fx, fy).reshape(len(fx), -1)
    e = lambda M: entropy(M, 1e-14)  # noqa: E731
    return e(fx.T @ fx / len(fx)) + e(fy.T @ fy / len(fy)) - e(P.T @ P / len(P))


def bias_coeff(F):
    """(1/2) sum_ij Var[(U^T v v^T U)_ij] / L(l_i, l_j): the O(1/m) bias coefficient of -tr M log M."""
    n = len(F)
    lam, U = np.linalg.eigh(F.T @ F / n)
    lam = np.clip(lam, 1e-14, None)
    Q = F @ U
    Q2 = Q * Q
    V = np.clip((Q2.T @ Q2) / n - ((Q.T @ Q) / n) ** 2, 0, None)
    a, b = lam[:, None], lam[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        Lm = (a - b) / (np.log(a) - np.log(b))
    Lm = np.where(np.abs(a - b) < 1e-13, a * np.ones_like(b), Lm)
    return 0.5 * float((V / Lm).sum())


def advance_stream(r):
    """Advance the NV-4 stream past the draws that preceded these replicates when the printed
    values were first computed (the same sequence of calls, in the same order), so that those values
    are reproduced exactly."""
    for t in range(40):
        r.standard_normal(400)
        r.uniform(0, 1)
        r.standard_normal(400)
    r.standard_normal(400)
    r.standard_normal(400)
    r.standard_normal((D, D))


def nv4():
    r = np.random.default_rng(4242)
    advance_stream(r)
    n = 100000
    x, y = r.standard_normal(n), r.standard_normal(n)
    fx = unit_rff(ranks01(x[:, None]), D, 2026)
    fy = unit_rff(ranks01(y[:, None]), D, 2027)
    psi = np.einsum("ti,tj->tij", fx, fy).reshape(n, D * D)
    c = bias_coeff(psi) - bias_coeff(fx) - bias_coeff(fy)
    meas = {}
    for m in (200, 800):
        v = [domi_rank(r.standard_normal(m), r.standard_normal(m)) for _ in range(200)]
        meas[str(m)] = m * float(np.mean(v))
    return dict(c_closed_form=c, m_times_mean_I_hat=meas, replicates=200, D=D)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res = {"NV-1": nv1(), "NV-4": nv4()}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
