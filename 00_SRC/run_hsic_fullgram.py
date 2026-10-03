#!/usr/bin/env python
"""Standard HSIC as a change-point baseline (full Gram matrices, median-heuristic bandwidth).

The HSIC comparator of Section 5 is built from the *same* D = 8 random Fourier features as DOMI and
differs from it only in the functional. Standard HSIC instead uses the full Gram matrix of a
Gaussian kernel; this runner computes that baseline.

Everything except the statistic follows Section 5: the same scenario generators and seeds as
`run_experiment.py --part e1`, the same candidate grid, thresholds set to a false-alarm rate
of 0.05 from 500 null replicates (first half for the pointwise studentizing moments, second half
for the studentized threshold quantile), and a detection counted only if |tau_hat - tau| <= 30.
One exception: the null is drawn at level index 0 for every level. This is exact for D1, D2 and
M1; the S3 and S4 (codes D3, D4) columns of Table B.2 come from run_bandwidth_matched_null.py, which recalibrates
them against the matched null.

The segment statistic is the biased (V-statistic) HSIC on global ranks,

    HSIC(S) = tr(Kx H Ky H) / m^2 = [ A - (2/m) B + Sx Sy / m^2 ] / m^2,

with A = sum_{i,j in S} Kx_ij Ky_ij, B = sum_{i in S} rx_i ry_i, rx_i = sum_{j in S} Kx_ij, and
Sx = sum_{i,j in S} Kx_ij. Two-dimensional prefix sums give A, Sx, Sy in O(1) per segment and the
row sums in O(1) each, so the whole candidate grid costs O(n |grid|) rather than O(n^3).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_hsic_fullgram.py --procs 24
Writes 04_DAOU/EXPERIMENT/hsic_fullgram/{results.csv,records_null.csv.gz,records_alt.csv.gz}.
records_null holds the studentized null maxima of all 500 replicates; the raw (unstudentized)
maxima behind the "g" threshold are not recorded.
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
from dots.synth import SCENARIOS, DEFAULT_W, generate  # noqa: E402
from dots.extras import studentize  # noqa: E402
from dots.persist import save_records  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "hsic_fullgram")
CFG = dict(n=600, tau=300, tol=30, base_seed=20260825, fpr=0.05, w=DEFAULT_W)
SCENS = ["D1", "D2", "D3", "D4", "M1"]


MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)   # multiples of the median-heuristic bandwidth swept in
                                     # Supplementary Section B.7 (Table B.2)


def _gram(u, mult=1.0):
    """Gaussian Gram matrix on one margin's ranks, median-heuristic bandwidth times `mult`."""
    d = np.abs(u[:, None] - u[None, :])
    s = np.median(d[np.triu_indices_from(d, 1)]) * mult
    return np.exp(-(d ** 2) / (2.0 * max(s, 1e-12) ** 2))


def _prep(x, y, mult=1.0):
    """Gram matrices and the prefix sums every segment statistic needs."""
    u, v = ranks01(x).ravel(), ranks01(y).ravel()
    Kx, Ky = _gram(u, mult), _gram(v, mult)
    P = {}
    for name, M in (("x", Kx), ("y", Ky), ("xy", Kx * Ky)):
        P[name] = np.pad(M.cumsum(0).cumsum(1), ((1, 0), (1, 0)))     # 2-D prefix
    P["rx"] = np.pad(Kx.cumsum(1), ((0, 0), (1, 0)))                  # row-wise prefix
    P["ry"] = np.pad(Ky.cumsum(1), ((0, 0), (1, 0)))
    return P


def _block(P2, a, b):
    """Sum of a Gram submatrix over the contiguous index block [a, b)."""
    return P2[b, b] - P2[a, b] - P2[b, a] + P2[a, a]


def _hsic(P, a, b):
    m = b - a
    if m < 3:
        return 0.0
    A = _block(P["xy"], a, b)
    Sx, Sy = _block(P["x"], a, b), _block(P["y"], a, b)
    rx = P["rx"][a:b, b] - P["rx"][a:b, a]
    ry = P["ry"][a:b, b] - P["ry"][a:b, a]
    B = float(rx @ ry)
    return float((A - 2.0 * B / m + Sx * Sy / m ** 2) / m ** 2)


def curve(x, y, w, n, mult=1.0):
    """|HSIC(left) - HSIC(right)| over the same candidate grid the other baselines use."""
    P = _prep(x, y, mult)
    grid = np.arange(w, n - w + 1)
    return np.array([abs(_hsic(P, 0, t) - _hsic(P, t, n)) for t in grid]), grid


def _job(a):
    s, li, r, null = a
    smp = generate(s, SCENARIOS[s]["levels"][li], r, null=null,
                   base_seed=CFG["base_seed"], level_idx=li)
    Z = smp["Z"]
    X, Y = Z[:, smp["blocks"][0]], Z[:, smp["blocks"][1]]
    out = {m: curve(X.ravel(), Y.ravel(), CFG["w"][s], CFG["n"], m)[0] for m in MULTS}
    grid = np.arange(CFG["w"][s], CFG["n"] - CFG["w"][s] + 1)
    return out, grid, int(smp["tau"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--scens", default=",".join(SCENS))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for s in a.scens.split(","):
        t0 = time.time()
        with Pool(a.procs) as pool:
            nulls = pool.map(_job, [(s, 0, r, True) for r in range(a.reps)], chunksize=4)
        half = a.reps // 2
        cal = {}
        for mlt in MULTS:                                   # each bandwidth calibrated on its own
            N = np.array([d[0][mlt] for d in nulls])
            mu0, sd0 = N[:half].mean(0), N[:half].std(0)
            cal[mlt] = dict(mu0=mu0, sd0=sd0,
                            raw=float(np.quantile(N.max(1), 1 - CFG["fpr"])),
                            gs=float(np.quantile([studentize(c, mu0, sd0).max() for c in N[half:]],
                                                 1 - CFG["fpr"])))
        save_records(OUT, "records_null.csv",
                     [{f"max_gs_x{mlt}": float(studentize(d[0][mlt], cal[mlt]["mu0"],
                                                          cal[mlt]["sd0"]).max()) for mlt in MULTS}
                      for d in nulls], {"scen": s})
        print(f"[{s}] null {a.reps} reps {time.time()-t0:.0f}s", flush=True)
        for li, level in enumerate(SCENARIOS[s]["levels"]):
            with Pool(a.procs) as pool:
                alts = pool.map(_job, [(s, li, r, False) for r in range(a.reps)], chunksize=4)
            recs, cell = [], {}
            for mlt in MULTS:
                c = cal[mlt]
                for mode in ("g", "gs"):
                    hit = det = 0
                    for out, grid, tau in alts:
                        cur = out[mlt] if mode == "g" else studentize(out[mlt], c["mu0"], c["sd0"])
                        th = c["raw"] if mode == "g" else c["gs"]
                        mx = float(np.nanmax(cur)); tau_hat = int(grid[int(np.nanargmax(cur))])
                        d_ = mx > th; h_ = d_ and abs(tau_hat - tau) <= CFG["tol"]
                        det += d_; hit += h_
                        recs.append({"mult": mlt, "mode": mode, "max": mx, "tau_hat": tau_hat,
                                     "detected": bool(d_), "power_hit": bool(h_)})
                    cell[(mlt, mode)] = (hit / len(alts), det / len(alts))
            save_records(OUT, "records_alt.csv", recs, {"scen": s, "level": level})
            (bm, bmode), (bp, bd) = max(cell.items(), key=lambda kv: kv[1][0])
            for (mlt, mode), (pw, dt) in cell.items():
                rows.append(dict(scen=s, level=level, method="HSIC-fullgram", mult=mlt, mode=mode,
                                 key=f"HSIC-fullgram|x{mlt}|{mode}",
                                 power=round(pw, 3), detect_rate=round(dt, 3),
                                 best=int((mlt, mode) == (bm, bmode))))
            print(f"   {s} level={level}: best power={bp:.3f} at bandwidth x{bm} ({bmode}), "
                  f"detect={bd:.3f}", flush=True)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print("written:", os.path.join(OUT, "results.csv"))


if __name__ == "__main__":
    main()
