#!/usr/bin/env python
"""Stage two (re-test) for the financial change points of Section 6.8.

Purpose
-------
Section 6.8 reports three segmentation breaks under calibrations the exchangeability diagnostic
accepts: S&P 500 vs 10-year-yield changes, weekly, 2020-02-27 (block-calibrated Holevo partitioning);
the same pair, monthly, 2021-08 (pair-calibrated); KOSPI vs USD/KRW, monthly, 2018-01
(block-calibrated). These are stage-one outputs. This script applies the paper's stage two
(Section 4.7, Algorithm 2 steps 5-6) to each of them exactly as it is applied to the weather
candidates (run_e8_retest.py, run_e8_marginal.py):

  * dependence test: the weighted DOMI-difference curve on a window centered on the candidate,
    ranks and random features (D = 8, frequency seeds 2026 / 2027) computed within the window,
    candidate split points on the stage-one grid, K = 999 block (or pair) permutations of the
    observation order, jointly in X and Y, symmetric studentization over all K + 1 curves, and
    the permutation p-value (1 + #{T_k >= T_0}) / (K + 1);
  * marginal test: the same design applied to the weighted |log sd_L - log sd_R| curve of each
    variable separately (raw values, not ranks);
  * classification: dependence change at Benjamini-Hochberg q <= 0.10 across the candidates
    tested together, marginal change at min(p_X, p_Y) <= 0.05, giving coupling change / change in
    both / marginal-driven / undetermined, as in run_e8_marginal.py.

Geometry rules (the weather rules, transferred to weekly and monthly resolution)
-------------------------------------------------------------------------------
  grid unit     the stage-one candidate-grid step (weather: two weeks; here 4 weeks for the weekly
                record and 2 months for the monthly record, the steps used by run_e6a_block.py and
                run_e6_monthly.py);
  block length  the exchangeability diagnostic's recommendation b_hat = ceil(5 tau_int) on the
                full record when the diagnostic rejects (Algorithm 2, step 1; the weather re-blocking
                of Supplementary Section B.8), b = 1 (pair permutation) when it does not. Full-record
                recommendations: 44 weeks (S&P/yield weekly), 12 months (KOSPI/KRW monthly);
                the monthly S&P/yield record is pair-exchangeable by the diagnostic;
  window        the weather window holds about 17 six-week blocks (+/- 1 year). At the finance
                block lengths that many blocks do not fit in the record, so the window is the
                largest window centered on the candidate that the record allows, its half-width
                rounded down to a whole number of blocks (to a whole number of grid units under pair
                permutation). This maximizes the number of blocks and hence the richness of the
                permutation group;
  minimum segment  one block, and at least 6% of the window (weather: one six-week super-block,
                3 of 52 intervals).
Sensitivity variants: half the recommended block length, the recommendation of the diagnostic computed on the window itself, and, for the
pair-calibrated monthly S&P/yield candidate, the diagnostic's b_hat = 12 months although the
diagnostic does not reject. The dependence test is also repeated over 20 further feature draws
(seeds 10000 + 2j, 10001 + 2j), holding the permutation stream fixed.

Run
---
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_e6_retest.py --procs 12

Inputs:  01_ORG/FINANCE/yahoo_{gspc,tnx,kospi,usdkrw}.csv; 04_DAOU/EXPERIMENT/e6/
         {results_block.json, results_monthly.json} (stage-one breaks and diagnostics)
Outputs: 04_DAOU/EXPERIMENT/e6/retest.json
         04_DAOU/EXPERIMENT/e6/records_retest_perm.csv.gz  (per test: observed statistic and all
         K replica maxima; every p-value regenerates from these rows)
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys
import time
from math import ceil
from multiprocessing import Pool

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.hourly import features, domi_diff_curve  # noqa: E402
from dots.perm import ge  # noqa: E402
from run_e6_finance import load, align_pair, weekly_sum  # noqa: E402
from run_e6_monthly import msum  # noqa: E402
from run_e8_marginal import curve_from  # noqa: E402
from run_exch_diag import exch_diag  # noqa: E402
from run_e8_summary import step_up_q  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e6")
K = 999
N_DRAWS = 20

# (label, pair, resolution, stage-one break date, grid unit in observations, diagnostic lag L)
CANDIDATES = [
    ("SPX-TNX weekly 2020-02-27", "SPX-TNX", "weekly", "20200227", 4, 20),
    ("SPX-TNX monthly 2021-08", "SPX-TNX", "monthly", "20210830", 2, 12),
    ("KOSPI-USDKRW monthly 2018-01", "KOSPI-USDKRW", "monthly", "20180110", 2, 12),
]


def series(pair, resolution):
    if pair == "SPX-TNX":
        dates, x, y = align_pair(load("yahoo_gspc.csv"), load("yahoo_tnx.csv"), ("logret", "diff"))
    else:
        dates, x, y = align_pair(load("yahoo_kospi.csv"), load("yahoo_usdkrw.csv"),
                                 ("logret", "logret"))
    if resolution == "weekly":
        xa, ya = weekly_sum(x), weekly_sum(y)
        da = dates[::5][: len(xa)]
    else:
        xa, ya = msum(x), msum(y)
        da = dates[::21][: len(xa)]
    return da, xa, ya


def perm_index(n, b, rng):
    """Permutation of the ORDER of consecutive blocks of b observations (b = 1: pair permutation).
    A remainder shorter than b stays in place at the end, as in dots.hourly.block_perm_order."""
    if b <= 1:
        return rng.permutation(n)
    nb = n // b
    order = rng.permutation(nb)
    idx = (order[:, None] * b + np.arange(b)[None, :]).ravel()
    return np.concatenate([idx, np.arange(nb * b, n)])


def unit_sums(A, unit):
    """Sums of the rows of A over consecutive grid units of `unit` observations."""
    edges = np.arange(0, len(A), unit)
    return np.add.reduceat(A, edges, axis=0), np.diff(np.append(edges, len(A)))


def outer_products(FX, FY, J):
    """Per-observation second moments phi_X phi_X^T, phi_Y phi_Y^T, psi psi^T."""
    return (np.einsum("ti,tj->tij", FX, FX), np.einsum("ti,tj->tij", FY, FY),
            np.einsum("ti,tj->tij", J, J))


def dep_curve(O, unit, lo):
    """Weighted DOMI-difference curve on the grid of units, from per-observation moments O."""
    mx, cnt = unit_sums(O[0], unit)
    my, _ = unit_sums(O[1], unit)
    mj, _ = unit_sums(O[2], unit)
    z = lambda a: np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])  # noqa: E731
    P = (z(mx), z(my), z(mj), np.concatenate([[0], np.cumsum(cnt)]))
    return domi_diff_curve(P, len(cnt), lo=lo)


def marg_curve(v, unit, lo):
    s1, cnt = unit_sums(v, unit)
    s2, _ = unit_sums(v ** 2, unit)
    return curve_from(s1, s2, cnt, lo=lo)


def perm_pvalue(obs, R):
    A = np.vstack([obs[None, :], R])
    mu, sd = A.mean(0), A.std(0) + 1e-12
    T = np.nanmax((A - mu) / sd, axis=1)
    return (1 + int(np.sum(ge(T[1:], T[0])))) / (len(T)), T


def window(n, c, unit, b):
    """Largest window centered on index c; half-width a whole number of blocks (units if b = 1)."""
    step = unit if b <= 1 else int(np.lcm(unit, b))
    h = (min(c, n - c) // step) * step
    return c - h, c + h


def _job(args):
    """One stage-two test: kind = 'dep' (with feature draw j) or 'marg_x' / 'marg_y'."""
    label, pair, res, date, unit, b, variant, kind, j, seed = args
    da, x, y = series(pair, res)
    c = da.index(date)
    lo_i, hi_i = window(len(x), c, unit, b)
    xw, yw = x[lo_i:hi_i], y[lo_i:hi_i]
    n = len(xw)
    Bunits = int(ceil(n / unit))
    lo = max(int(ceil(max(b, 1) / unit)), int(ceil(0.06 * Bunits)))
    rng = np.random.default_rng(seed)
    if kind == "dep":
        sx, sy = (2026, 2027) if j == 0 else (10000 + 2 * j, 10001 + 2 * j)
        FX, FY, J = features(xw, yw, sx=sx, sy=sy)
        O = outer_products(FX, FY, J)
        ts, obs = dep_curve(O, unit, lo)
        R = np.empty((K, len(ts)))
        for k in range(K):
            o = perm_index(n, b, rng)
            R[k] = dep_curve(tuple(M[o] for M in O), unit, lo)[1]
    else:
        v = xw if kind == "marg_x" else yw
        ts, obs = marg_curve(v, unit, lo)
        R = np.empty((K, len(ts)))
        for k in range(K):
            R[k] = marg_curve(v[perm_index(n, b, rng)], unit, lo)[1]
    p, T = perm_pvalue(obs, R)
    return dict(label=label, variant=variant, block=b, kind=kind, draw=j,
                window=f"{da[lo_i]}-{da[hi_i - 1]}", n_window=n,
                n_blocks=(n // b if b > 1 else n), min_seg_units=lo,
                p=round(float(p), 4), T_obs=float(T[0]),
                T_rep=" ".join(f"{t:.6f}" for t in T[1:]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    a_ = ap.parse_args()
    rb = json.load(open(os.path.join(OUT, "results_block.json")))
    rm = json.load(open(os.path.join(OUT, "results_monthly.json")))
    stage1 = {CANDIDATES[0][0]: dict(source="results_block.json", **{k: rb["e6a_block_SPX-TNX"][k] for k in
                                     ("diag_decision", "block", "cps_dates")}),
              CANDIDATES[1][0]: dict(source="results_monthly.json", **{k: rm["SPX-TNX"][k] for k in
                                     ("diag_decision", "b_hat", "calibration", "cps_dates")}),
              CANDIDATES[2][0]: dict(source="results_monthly.json", **{k: rm["KOSPI-USDKRW"][k] for k in
                                     ("diag_decision", "b_hat", "calibration", "cps_dates")})}
    jobs, meta = [], {}
    for ci, (label, pair, res, date, unit, L) in enumerate(CANDIDATES):
        da, x, y = series(pair, res)
        c = da.index(date)
        # same seeds as the full-record diagnostics of run_e6a_block.py (weekly, 3000 + len(name))
        # and run_e6_monthly.py (monthly, 4000 + len(name)); the conditional covers the whole sum
        dfull = exch_diag(x, y, L=L, seed=3000 + len(pair) if res == "weekly" else 4000 + len(pair))
        b_primary = dfull["b_hat"] if dfull["decision"] == "block" else 1
        lo_i, hi_i = window(len(x), c, unit, b_primary)
        dwin = exch_diag(x[lo_i:hi_i], y[lo_i:hi_i], L=L, seed=5000 + ci)
        variants = {"primary": b_primary}
        if b_primary > 1:
            variants["half_block"] = max(unit, int(round(b_primary / 2 / unit)) * unit)
        else:
            variants["block_bhat_despite_pair_decision"] = dfull["b_hat"]
        bw = dwin["b_hat"] if dwin["decision"] == "block" else 1
        if bw not in variants.values():
            variants["window_diagnostic"] = bw
        meta[label] = dict(pair=pair, resolution=res, date=date, index=c, n_record=len(x),
                           grid_unit_obs=unit, diag_full=dfull, diag_window=dwin,
                           variants=variants, stage1=stage1[label])
        for vi, (vname, b) in enumerate(variants.items()):
            base = 610000 + 1000 * ci + 100 * vi
            jobs.append((label, pair, res, date, unit, b, vname, "dep", 0, base + 1))
            jobs.append((label, pair, res, date, unit, b, vname, "marg_x", 0, base + 2))
            jobs.append((label, pair, res, date, unit, b, vname, "marg_y", 0, base + 3))
            if vname == "primary":
                for j in range(1, N_DRAWS + 1):          # same permutation stream as draw 0
                    jobs.append((label, pair, res, date, unit, b, vname, "dep", j, base + 1))
    print(f"{len(jobs)} permutation tests at K={K}", flush=True)
    t0 = time.time()
    with Pool(a_.procs, maxtasksperchild=8) as pool:
        out = pool.map(_job, jobs, chunksize=1)
    wall = time.time() - t0
    with gzip.open(os.path.join(OUT, "records_retest_perm.csv.gz"), "wt", newline="") as f:
        keys = ["label", "variant", "block", "kind", "draw", "window", "n_window", "n_blocks",
                "min_seg_units", "p", "T_obs", "T_rep"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    get = lambda lab, v, kind, j=0: next(r for r in out if r["label"] == lab and  # noqa: E731
                                         r["variant"] == v and r["kind"] == kind and r["draw"] == j)
    table = {}
    for vname in ("primary",):
        pd = np.array([get(lab, vname, "dep")["p"] for lab, *_ in CANDIDATES])
        q = step_up_q(pd)
        for i, (lab, *_) in enumerate(CANDIDATES):
            pm = min(get(lab, vname, "marg_x")["p"], get(lab, vname, "marg_y")["p"])
            dep, marg = q[i] <= 0.10, pm <= 0.05
            table[lab] = dict(p_dep=float(pd[i]), bh_q_dep=round(float(q[i]), 4),
                              p_marg_x=get(lab, vname, "marg_x")["p"],
                              p_marg_y=get(lab, vname, "marg_y")["p"],
                              p_marg_min=pm,
                              cls=("coupling change" if dep and not marg else "change in both"
                                   if dep and marg else "marginal-driven" if marg else "undetermined"))
    for lab, *_ in CANDIDATES:
        m = meta[lab]
        r0 = get(lab, "primary", "dep")
        m["window"] = r0["window"]; m["n_window"] = r0["n_window"]
        m["n_blocks_primary"] = r0["n_blocks"]; m["min_seg_units"] = r0["min_seg_units"]
        m["retest"] = table[lab]
        m["sensitivity"] = {v: dict(block=b, p_dep=get(lab, v, "dep")["p"],
                                    p_marg_x=get(lab, v, "marg_x")["p"],
                                    p_marg_y=get(lab, v, "marg_y")["p"],
                                    window=get(lab, v, "dep")["window"],
                                    n_blocks=get(lab, v, "dep")["n_blocks"])
                            for v, b in m["variants"].items()}
        pdraw = np.array([get(lab, "primary", "dep", j)["p"] for j in range(1, N_DRAWS + 1)])
        m["feature_draws"] = dict(p=[float(v) for v in pdraw], min=float(pdraw.min()),
                                  median=float(np.median(pdraw)), max=float(pdraw.max()),
                                  frac_le_05=float(np.mean(pdraw <= 0.05)))
    res = dict(config=dict(K=K, D=8, feature_seeds=[2026, 2027], n_extra_draws=N_DRAWS,
                           multiplicity="BH across the three candidates, dependence passed at q<=0.10",
                           marginal_rule="min(p_X, p_Y) <= 0.05"),
               candidates=meta, wall_clock_s=round(wall, 1))
    json.dump(res, open(os.path.join(OUT, "retest.json"), "w"), indent=1, default=float)
    for lab, *_ in CANDIDATES:
        m = meta[lab]
        print(f"\n{lab}: window {m['window']} (n={m['n_window']}, blocks={m['n_blocks_primary']}, "
              f"b={m['variants']['primary']}) diag_full={m['diag_full']['decision']} "
              f"b_hat={m['diag_full']['b_hat']} | diag_window={m['diag_window']['decision']} "
              f"b_hat={m['diag_window']['b_hat']}")
        print(f"   primary: {m['retest']}")
        for v, s in m["sensitivity"].items():
            print(f"   {v}: {s}")
        print(f"   draws: {m['feature_draws']['min']:.3f}/{m['feature_draws']['median']:.3f}/"
              f"{m['feature_draws']['max']:.3f} frac<=.05 {m['feature_draws']['frac_le_05']:.2f}")
    print(f"wall clock {wall:.0f}s")


if __name__ == "__main__":
    main()
