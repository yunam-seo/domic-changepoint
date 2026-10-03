#!/usr/bin/env python
"""DOMI under the same bandwidth sweep applied to the full-Gram HSIC baseline.

`run_hsic_fullgram.py` shows that a standard HSIC on the same global ranks becomes far more
powerful when its Gaussian kernel is narrowed to a quarter of the median-heuristic bandwidth. That
result is only interpretable next to the same sweep for DOMI: if DOMI moves the same way, the two
methods were simply being compared at an uninformative operating point; if it does not, the
narrower kernel is a genuine advantage of the full Gram representation.

DOMI's random Fourier features use gamma = 1 / median squared distance. Narrowing the kernel by a
factor c on the *distance* scale means gamma / c^2, so the multipliers here are the squared
reciprocals of the HSIC ones: sigma x {4, 2, 1, 1/2, 1/4} <-> gamma x {1/16, 1/4, 1, 4, 16}.

Protocol is Section 5 throughout: same generators and seeds as `run_experiment.py --part e1`,
same candidate grid, thresholds at a false-alarm rate of 0.05 from 500 null replicates (first half
for the pointwise studentizing moments, second half for the studentized threshold), detection only
if |tau_hat - tau| <= 30.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_domi_bandwidth.py --scens D2
Writes 04_DAOU/EXPERIMENT/domi_bandwidth/{results_<scens>.csv, records_alt.csv.gz} (results.csv in
that folder is the concatenation of the per-scenario files).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from dots.domi import ranks01  # noqa: E402
from dots.extras import rff, studentize  # noqa: E402
from dots.synth import SCENARIOS, DEFAULT_W, generate  # noqa: E402
from dots.persist import save_records  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "domi_bandwidth")
CFG = dict(n=600, tau=300, tol=30, base_seed=20260825, fpr=0.05, D=8, w=DEFAULT_W)
SCENS = ["D1", "D2", "D3", "D4", "M1"]
# gamma multipliers; the labels are the equivalent sigma multipliers used by the HSIC sweep
GAM = {4.0: 1 / 16, 2.0: 1 / 4, 1.0: 1.0, 0.5: 4.0, 0.25: 16.0}


def _ent(M):
    lam = np.linalg.eigvalsh(M)
    lam = lam[lam > 1e-15]
    return float(-(lam * np.log(lam)).sum())


def _median_gamma(U, seed):
    idx = np.random.default_rng(seed + 1).choice(len(U), size=min(len(U), 400), replace=False)
    S = U[idx]
    sq = (S * S).sum(1)
    D2 = np.clip(sq[:, None] + sq[None, :] - 2 * S @ S.T, 0, None)
    med = np.median(D2[np.triu_indices(len(idx), 1)])
    return 1.0 / (med if med > 0 else 1.0)


def _unit(F):
    return F / np.linalg.norm(F, axis=1, keepdims=True)


def curve(x, y, w, n, D, gmult, seed=2026):
    """|DOMI(left) - DOMI(right)| over the candidate grid, at gamma scaled by `gmult`."""
    UX, UY = ranks01(x), ranks01(y)
    gx, gy = _median_gamma(UX, seed) * gmult, _median_gamma(UY, seed + 1) * gmult
    FX = _unit(rff(UX, D=D, seed=seed, gamma=gx))
    FY = _unit(rff(UY, D=D, seed=seed + 1, gamma=gy))
    J = np.einsum("ti,tj->tij", FX, FY).reshape(n, D * D)
    PX = np.concatenate([np.zeros((1, D, D)), np.einsum("ti,tj->tij", FX, FX).cumsum(0)])
    PY = np.concatenate([np.zeros((1, D, D)), np.einsum("ti,tj->tij", FY, FY).cumsum(0)])
    PJ = np.concatenate([np.zeros((1, D * D, D * D)), np.einsum("ti,tj->tij", J, J).cumsum(0)])
    grid = np.arange(w, n - w + 1)

    def domi(a, b):
        m = b - a
        return (_ent((PX[b] - PX[a]) / m) + _ent((PY[b] - PY[a]) / m)
                - _ent((PJ[b] - PJ[a]) / m))

    return np.array([abs(domi(0, t) - domi(t, n)) for t in grid]), grid


def _job(a):
    s, li, r, null = a
    smp = generate(s, SCENARIOS[s]["levels"][li], r, null=null,
                   base_seed=CFG["base_seed"], level_idx=li)
    Z = smp["Z"]
    X, Y = Z[:, smp["blocks"][0]], Z[:, smp["blocks"][1]]
    out = {sig: curve(X, Y, CFG["w"][s], CFG["n"], CFG["D"], g)[0] for sig, g in GAM.items()}
    grid = np.arange(CFG["w"][s], CFG["n"] - CFG["w"][s] + 1)
    return out, grid, int(smp["tau"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--scens", default=",".join(SCENS))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for s in a.scens.split(","):
        t0 = time.time()
        with Pool(a.procs) as pool:
            nulls = pool.map(_job, [(s, 0, r, True) for r in range(a.reps)], chunksize=2)
        half = a.reps // 2
        cal = {}
        for sig in GAM:
            N = np.array([d[0][sig] for d in nulls])
            mu0, sd0 = N[:half].mean(0), N[:half].std(0)
            cal[sig] = dict(mu0=mu0, sd0=sd0,
                            gs=float(np.quantile([studentize(c, mu0, sd0).max() for c in N[half:]],
                                                 1 - CFG["fpr"])))
        print(f"[{s}] null {a.reps} reps {time.time()-t0:.0f}s", flush=True)
        for li, level in enumerate(SCENARIOS[s]["levels"]):
            with Pool(a.procs) as pool:
                alts = pool.map(_job, [(s, li, r, False) for r in range(a.reps)], chunksize=2)
            recs = []
            for sig in GAM:
                c = cal[sig]
                hit = det = 0
                for out, grid, tau in alts:
                    cur = studentize(out[sig], c["mu0"], c["sd0"])
                    mx = float(np.nanmax(cur)); tau_hat = int(grid[int(np.nanargmax(cur))])
                    d_ = mx > c["gs"]; h_ = d_ and abs(tau_hat - tau) <= CFG["tol"]
                    det += d_; hit += h_
                    recs.append({"sigma_mult": sig, "max_gs": mx, "tau_hat": tau_hat,
                                 "detected": bool(d_), "power_hit": bool(h_)})
                rows.append(dict(scen=s, level=level, method="DOMI", sigma_mult=sig, mode="gs",
                                 power=round(hit / len(alts), 3),
                                 detect_rate=round(det / len(alts), 3)))
            save_records(OUT, "records_alt.csv", recs, {"scen": s, "level": level})
            sel = [r for r in rows if r["scen"] == s and r["level"] == level]
            print("   " + f"{s} level={level}: " +
                  ", ".join(f"x{r['sigma_mult']}={r['power']:.3f}" for r in sel), flush=True)
    with open(os.path.join(OUT, f"results_{a.scens.replace(',','_')}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print("written:", OUT)


if __name__ == "__main__":
    main()
