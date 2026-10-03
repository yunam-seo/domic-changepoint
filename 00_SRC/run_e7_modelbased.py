#!/usr/bin/env python
"""Model-based and log-determinant dependence baselines (Supplementary Section B.4):
(1) Pearson correlation CUSUM in the spirit of Wied, Kraemer and Dehling (2012),
    (t/sqrt n)|rho_t - rho_n| without a variance normalizer, calibrated by Monte Carlo,
(2) DCC(1,1)-GARCH fitted correlation path + CUSUM on the path (two-stage QMLE, local implementation),
(3) Bach-Jordan log-determinant mutual information from regularized canonical correlations on the
    same rank features (KCCA-logdet).

Power study on D1 (their natural alternative), D2/D4 (correlation-zero dependence), M1 (specificity),
Monte Carlo calibrated at a false-alarm rate of 0.05 against the level-0 null of each scenario
(200 replicates). The level-0 null is the matching null for S1, S2 and M1, whose no-change regime does not
depend on the level; for S4 it does, and the S4 numbers of Supplementary Section B.4 come from
run_e7_matched_d4.py, which draws the S4 null at every level and re-scores the alternatives stored here. Plus the 2021-22 S&P 500 - 10-year yield daily window with pair-permutation
calibration (K=99) for the correlation CUSUM and KCCA-logdet (DCC refit per permutation, K=49).

Writes 04_DAOU/EXPERIMENT/e7/{results.csv, records_reps.csv, finance.json}
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.synth import SCENARIOS, generate  # noqa: E402
from dots.domi import ranks01, unit_rff  # noqa: E402
from dots.perm import ge  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e7")
CFG = dict(n=600, tau=300, tol=30, fpr=0.05, base_seed=20260901, reps=200, w=60)


# ---------------------------------------------------------------- statistics
def corr_cusum(x, y, w):
    """Wied-type statistic: successive-vs-full Pearson correlation difference, weighted."""
    n = len(x)
    xs, ys = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    cxy = np.cumsum(xs * ys)
    cx2 = np.cumsum(xs * xs)
    cy2 = np.cumsum(ys * ys)
    ts = np.arange(w, n - w + 1)
    rho_j = cxy[ts - 1] / np.sqrt(cx2[ts - 1] * cy2[ts - 1])
    rho_n = cxy[-1] / np.sqrt(cx2[-1] * cy2[-1])
    return ts, (ts / np.sqrt(n)) * np.abs(rho_j - rho_n)


def garch11_fit(r):
    """QMLE of GARCH(1,1) on returns r; returns conditional sd path."""
    r = r - r.mean()
    v0 = r.var()

    def nll(theta):
        om, al, be = np.exp(theta[0]), 1 / (1 + np.exp(-theta[1])) * 0.3, 1 / (1 + np.exp(-theta[2])) * 0.98
        if al + be >= 0.999:
            be = 0.999 - al
        h = np.empty(len(r))
        h[0] = v0
        for t in range(1, len(r)):
            h[t] = om + al * r[t - 1] ** 2 + be * h[t - 1]
        h = np.clip(h, 1e-12, None)
        return 0.5 * np.sum(np.log(h) + r**2 / h)

    res = minimize(nll, x0=[np.log(v0 * 0.05), 0.0, 2.0], method="Nelder-Mead",
                   options=dict(maxiter=400, xatol=1e-4, fatol=1e-4))
    om, al, be = np.exp(res.x[0]), 1 / (1 + np.exp(-res.x[1])) * 0.3, 1 / (1 + np.exp(-res.x[2])) * 0.98
    if al + be >= 0.999:
        be = 0.999 - al
    h = np.empty(len(r))
    h[0] = v0
    for t in range(1, len(r)):
        h[t] = om + al * r[t - 1] ** 2 + be * h[t - 1]
    return r / np.sqrt(np.clip(h, 1e-12, None))


def dcc_rho_path(x, y):
    """Two-stage DCC(1,1): GARCH(1,1) margins -> standardized residuals -> DCC(a,b) QMLE -> rho_t."""
    e1, e2 = garch11_fit(x), garch11_fit(y)
    E = np.vstack([e1, e2]).T
    Qbar = np.corrcoef(E.T)

    def rho_of(a, b):
        Q = Qbar.copy()
        rho = np.empty(len(E))
        for t in range(len(E)):
            if t > 0:
                et = E[t - 1][:, None]
                Q = (1 - a - b) * Qbar + a * (et @ et.T) + b * Q
            rho[t] = Q[0, 1] / np.sqrt(Q[0, 0] * Q[1, 1])
        return np.clip(rho, -0.999, 0.999)

    def nll(theta):
        a = 1 / (1 + np.exp(-theta[0])) * 0.2
        b = 1 / (1 + np.exp(-theta[1])) * 0.97
        if a + b >= 0.999:
            b = 0.999 - a
        rho = rho_of(a, b)
        z1, z2 = E[:, 0], E[:, 1]
        return 0.5 * np.sum(np.log(1 - rho**2) + (z1**2 + z2**2 - 2 * rho * z1 * z2) / (1 - rho**2))

    res = minimize(nll, x0=[-2.0, 2.0], method="Nelder-Mead", options=dict(maxiter=200, xatol=1e-3, fatol=1e-3))
    a = 1 / (1 + np.exp(-res.x[0])) * 0.2
    b = 1 / (1 + np.exp(-res.x[1])) * 0.97
    if a + b >= 0.999:
        b = 0.999 - a
    return rho_of(a, b)


def dcc_cusum(x, y, w):
    """CUSUM of the fitted DCC correlation path (partial mean vs global mean, weighted)."""
    rho = dcc_rho_path(np.asarray(x, float), np.asarray(y, float))
    n = len(rho)
    cs = np.cumsum(rho - rho.mean())
    ts = np.arange(w, n - w + 1)
    return ts, np.abs(cs[ts - 1]) / np.sqrt(n)


def kcca_logdet_diff(x, y, w, D=8, reg=1e-2):
    """Kernel-CCA / Bach-Jordan log-det mutual information on the SAME rank-RFF features as DOMI:
    I_ld(seg) = -0.5 * sum log(1 - rho_i^2), rho_i = regularized canonical correlations of
    (phi_X, phi_Y) over the segment; statistic = weighted |I_ld(L) - I_ld(R)| (global form)."""
    FX = unit_rff(ranks01(np.asarray(x)[:, None]), D, 2026)
    FY = unit_rff(ranks01(np.asarray(y)[:, None]), D, 2027)
    n = len(FX)
    cXX = np.zeros((n + 1, D, D)); cYY = np.zeros((n + 1, D, D)); cXY = np.zeros((n + 1, D, D))
    np.cumsum(np.einsum("ti,tj->tij", FX, FX), axis=0, out=cXX[1:])
    np.cumsum(np.einsum("ti,tj->tij", FY, FY), axis=0, out=cYY[1:])
    np.cumsum(np.einsum("ti,tj->tij", FX, FY), axis=0, out=cXY[1:])

    def ild(a, b):
        m = b - a
        Sxx = (cXX[b] - cXX[a]) / m + reg * np.eye(D)
        Syy = (cYY[b] - cYY[a]) / m + reg * np.eye(D)
        Sxy = (cXY[b] - cXY[a]) / m
        lx, Ux = np.linalg.eigh(Sxx); ly, Uy = np.linalg.eigh(Syy)
        Wx = (Ux / np.sqrt(np.clip(lx, 1e-10, None))) @ Ux.T
        Wy = (Uy / np.sqrt(np.clip(ly, 1e-10, None))) @ Uy.T
        rho = np.clip(np.linalg.svd(Wx @ Sxy @ Wy, compute_uv=False), 0, 0.999)
        return float(-0.5 * np.sum(np.log(1 - rho**2)))

    ts = np.arange(w, n - w + 1)
    v = np.array([np.sqrt(t * (n - t) / n) * abs(ild(0, t) - ild(t, n)) for t in ts])
    return ts, v


# ---------------------------------------------------------------- power study
def _job(a):
    scen, li, rep, null, cfg = a
    lv = SCENARIOS[scen]["levels"][li]
    smp = generate(scen, lv, rep, n=cfg["n"], tau=cfg["tau"], null=null, base_seed=cfg["base_seed"], level_idx=li)
    Z = smp["Z"]
    x, y = Z[:, 0], Z[:, 1]
    out = {}
    for name, fn in [("CorrCUSUM", corr_cusum), ("DCC-CUSUM", dcc_cusum), ("KCCA-logdet", kcca_logdet_diff)]:
        ts, v = fn(x, y, cfg["w"])
        out[name] = (float(np.nanmax(v)), int(ts[int(np.nanargmax(v))]))
    return out


def run_power(cfg, procs):
    rows = []
    rec = []
    with Pool(procs, maxtasksperchild=40) as pool:
        for scen in ["D1", "D2", "D4", "M1"]:
            t0 = time.time()
            nulls = pool.map(_job, [(scen, 0, r, True, cfg) for r in range(cfg["reps"])], chunksize=2)
            thr = {m: float(np.quantile([d[m][0] for d in nulls], 1 - cfg["fpr"])) for m in nulls[0]}
            for li, lv in enumerate(SCENARIOS[scen]["levels"]):
                alts = pool.map(_job, [(scen, li, r, False, cfg) for r in range(cfg["reps"])], chunksize=2)
                for ri, d in enumerate(alts):
                    for m in thr:
                        rec.append(dict(scen=scen, level=lv, method=m, rep=ri,
                                        stat=float(d[m][0]), tau_hat=int(d[m][1]),
                                        detect=int(d[m][0] > thr[m]), threshold=thr[m]))
                for m in thr:
                    if scen == "M1":
                        det = float(np.mean([d[m][0] > thr[m] for d in alts]))
                        rows.append(dict(scen=scen, level=lv, method=m, power="", detect_rate=det))
                    else:
                        pw = float(np.mean([d[m][0] > thr[m] and abs(d[m][1] - cfg["tau"]) <= cfg["tol"] for d in alts]))
                        det = float(np.mean([d[m][0] > thr[m] for d in alts]))
                        rows.append(dict(scen=scen, level=lv, method=m, power=pw, detect_rate=det))
            print(f"[{time.strftime('%H:%M:%S')}] {scen} done {time.time()-t0:.0f}s | " +
                  ", ".join(f"{r['method']} {r['scen']} {r['level']}: P={r['power']} det={r['detect_rate']:.2f}" for r in rows[-6:]), flush=True)
    with open(os.path.join(OUT, "records_reps.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rec[0].keys()))
        wr.writeheader()
        wr.writerows(rec)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


# ---------------------------------------------------------------- finance window
def run_finance():
    from run_e6_finance import load, align_pair
    spx, tnx = load("yahoo_gspc.csv"), load("yahoo_tnx.csv")
    dates, x, y = align_pair(spx, tnx, ("logret", "diff"))
    sel = [i for i, d in enumerate(dates) if "20210101" <= d <= "20221231"]
    xs, ys = x[sel], y[sel]
    res = {}
    rng = np.random.default_rng(20260901)
    for name, fn, K in [("CorrCUSUM", corr_cusum, 99), ("DCC-CUSUM", dcc_cusum, 49), ("KCCA-logdet", kcca_logdet_diff, 99)]:
        ts, v = fn(xs, ys, 60)
        obs = float(np.nanmax(v))
        tau_hat = int(ts[int(np.nanargmax(v))])
        perms = []
        for k in range(K):
            idx = rng.permutation(len(xs))
            _, vp = fn(xs[idx], ys[idx], 60)
            perms.append(float(np.nanmax(vp)))
        p = (1 + sum(ge(t, obs) for t in perms)) / (K + 1)
        res[name] = dict(tau_date=dates[sel[0] + tau_hat], p_value=p, K=K)
        print(f"finance {name}: tau_hat={res[name]['tau_date']} p={p:.3f}", flush=True)
    json.dump(res, open(os.path.join(OUT, "finance.json"), "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--procs", type=int, default=24)
    a = ap.parse_args()
    cfg = dict(CFG)
    cfg["reps"] = a.reps
    run_power(cfg, a.procs)
    run_finance()
    print("E7 done", flush=True)
