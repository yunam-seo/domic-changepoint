#!/usr/bin/env python
"""Stability of the weather stage-two results under the random-feature draw (Supplementary B.8).

Purpose
-------
Every real-data p-value in Section 6.7 is computed from a single draw of D random Fourier
frequencies per variable block, and Supplementary Section B.5 shows that real-data p-values can move
with that draw. For every weather candidate whose stage-two p-value under the unrestricted scheme
is at most 0.05 (Jeonju temperature-humidity 2019-10-21, Suwon humidity-wind 2019-08-26 and three
borderline candidates), the stage-two test is repeated under 20 further, independent feature draws.

Everything except the feature draw is held exactly as in stage two of run_e8_retest.py: the same
+/- 1 year window (26 two-week intervals on each side of the candidate), six-week super-blocks
(three intervals) whose order is permuted jointly in X and Y, K = 999 replicas, the same permutation
stream (seed 4242 + candidate interval, so the replica permutations are identical across draws and
only the features differ), symmetric studentization over all K + 1 curves and the permutation
p-value (1 + #{T_k >= T_0}) / (K + 1). Draw 0 is the deployed draw (frequency seeds 2026 / 2027),
whose p-values are those of run_e8_retest.py; draws 1..20 use seeds (10000 + 2j, 10001 + 2j).

The script also restates, from e8/retest_summary.json, the Benjamini-Yekutieli reading of the 27
stage-two p-values (valid under arbitrary dependence), and recomputes the BH and BY q-values under
each draw (the draw applied to all five re-tested candidates, the other 22 held at their K = 999
values from run_e8_retest.py), to show whether any draw changes that reading.

Run
---
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_e8_feature_draws.py --procs 12
    python 00_SRC/run_e8_feature_draws.py --summarize-only   # rebuild the JSON from the records

Inputs:  02_MART/WEATHER_HOURLY_ANOM.npz; 04_DAOU/EXPERIMENT/e8/{retest_K999.json,
         retest_summary.json}
Outputs: 04_DAOU/EXPERIMENT/e8/feature_draws.json            (per candidate p-value distribution)
         04_DAOU/EXPERIMENT/e8/records_feature_draws_perm.csv.gz (per candidate x draw: observed
         statistic and all K replica maxima, from which every p-value regenerates)
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.hourly import features, block_moments, prefix, domi_diff_curve, block_perm_order  # noqa: E402
from dots.perm import ge  # noqa: E402
from run_e8_retest import UNIT, K, SUPER  # noqa: E402
from run_e8_summary import step_up_q  # noqa: E402

E8 = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8")
ANOM = os.path.join(ROOT, "02_MART", "WEATHER_HOURLY_ANOM.npz")
N_DRAWS = 20
HALF = 26          # +/- 26 two-week intervals = +/- 1 year, as in run_e8_retest.py
LO = 3             # minimum segment = one super-block, as in run_e8_retest.py


def draw_seeds(j):
    """Feature seeds of draw j; j = 0 is the deployed draw."""
    return (2026, 2027) if j == 0 else (10000 + 2 * j, 10001 + 2 * j)


def _job(args):
    stn, pair, c, date, j = args
    u, v = pair.split("-")
    z = np.load(ANOM)
    a, b = z[f"{stn}|{u}"], z[f"{stn}|{v}"]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    ef = np.arange(0, len(a) + 1, UNIT)
    if ef[-1] != len(a):
        ef = np.append(ef, len(a))
    B = len(ef) - 1
    lo, hi = max(0, c - HALF), min(B, c + HALF)
    xa, ya = a[ef[lo]:ef[hi]], b[ef[lo]:ef[hi]]
    ew = np.arange(0, len(xa) + 1, UNIT)
    if ew[-1] != len(xa):
        ew = np.append(ew, len(xa))
    sx, sy = draw_seeds(j)
    FX, FY, J = features(xa, ya, sx=sx, sy=sy)
    mx, my, mj, cnt = block_moments(FX, FY, J, ew)
    Bw = len(ew) - 1
    ts, obs = domi_diff_curve(prefix(mx, my, mj, cnt), Bw, lo=LO)
    rng = np.random.default_rng(4242 + c)          # identical permutation stream across draws
    R = np.empty((K, len(ts)))
    for k in range(K):
        o = block_perm_order(Bw, SUPER, rng)
        R[k] = domi_diff_curve(prefix(mx[o], my[o], mj[o], cnt[o]), Bw, lo=LO)[1]
    A = np.vstack([obs[None, :], R])
    mu, sd = A.mean(0), A.std(0) + 1e-12
    T = np.nanmax((A - mu) / sd, axis=1)
    p = (1 + int(np.sum(ge(T[1:], T[0])))) / (K + 1)
    return dict(stn=stn, pair=pair, date=date, interval=c, draw=j, seed_x=sx, seed_y=sy,
                p=round(float(p), 4), T_obs=float(T[0]),
                T_rep=" ".join(f"{t:.6f}" for t in T[1:]))


def summarise(res, retest_rows, summ, wall):
    """Per-candidate p-value distribution across draws, and the BH / BY reading per draw.

    For draw j the five re-tested candidates all take their draw-j p-values (one draw is one set of
    frequency seeds, applied to every candidate) and the other 22 keep their K = 999 values."""
    cands = [r for r in retest_rows if r["p_block"] <= 0.05]
    all_p = {(r["stn"], r["pair"], r["interval"]): r["p_block"] for r in retest_rows}
    keys27 = list(all_p)
    m = len(keys27)
    H = float(np.sum(1.0 / np.arange(1, m + 1)))
    pdraw = {(x["stn"], x["pair"], int(x["interval"]), int(x["draw"])): float(x["p"]) for x in res}
    qbh, qby = {}, {}
    for j in range(1, N_DRAWS + 1):
        pv = np.array([pdraw.get((k[0], k[1], k[2], j), all_p[k]) for k in keys27])
        qbh[j], qby[j] = step_up_q(pv), step_up_q(pv, H)
    n_by = [int(np.sum(qby[j] <= 0.10)) for j in range(1, N_DRAWS + 1)]
    n_bh = [int(np.sum(qbh[j] <= 0.10)) for j in range(1, N_DRAWS + 1)]
    out_c = []
    for r in cands:
        key = (r["stn"], r["pair"], r["interval"])
        i = keys27.index(key)
        pd = np.array([pdraw[key + (j,)] for j in range(1, N_DRAWS + 1)])
        bh = np.array([qbh[j][i] for j in range(1, N_DRAWS + 1)])
        by = np.array([qby[j][i] for j in range(1, N_DRAWS + 1)])
        out_c.append(dict(
            stn=r["stn"], pair=r["pair"], date=r["date"], interval=r["interval"],
            p_unrestricted=r["p_block"],
            p_draws=[float(v) for v in pd],
            p_min=float(pd.min()), p_median=float(np.median(pd)), p_max=float(pd.max()),
            frac_le_05=float(np.mean(pd <= 0.05)), frac_le_01=float(np.mean(pd <= 0.01)),
            bh_q_min=float(bh.min()), bh_q_median=float(np.median(bh)), bh_q_max=float(bh.max()),
            frac_bh_q_le_10=float(np.mean(bh <= 0.10)),
            by_q_min=float(by.min()), by_q_median=float(np.median(by)), by_q_max=float(by.max()),
            frac_by_q_le_10=float(np.mean(by <= 0.10))))
    out = dict(
        config=dict(K=K, unit_hours=UNIT, super_block_intervals=SUPER, half_window_intervals=HALF,
                    min_segment_intervals=LO, D=8, n_draws=N_DRAWS,
                    draw_seeds={str(j): list(draw_seeds(j)) for j in range(N_DRAWS + 1)},
                    permutation_seed="4242 + interval (identical across draws)",
                    candidates="stage-two p <= 0.05 in e8/retest_K999.json",
                    per_draw_multiplicity="draw j applied to all five candidates; other 22 at their K = 999 p-values"),
        by_reading_unrestricted=dict(
            n_candidates=summ["n_candidates"], harmonic_number=H,
            n_by_q_le_10=summ["n_by_q_le_10"], strongest=summ["strongest"],
            by_q_second=sorted(summ["per_candidate"], key=lambda c: c["p"])[1]["by_q"],
            by_q_floor_single_candidate_at_K=m * H / (K + 1),
            by_q_floor_two_candidates_at_K=m * H / (2 * (K + 1)),
            reading="Under Benjamini-Yekutieli no candidate has q <= 0.10: the strongest (Jeonju "
                    "ta-hm) has q = 0.105 and the second (Suwon hm-ws) q = 0.210. 0.105 is the "
                    "smallest BY q-value a lone top candidate can reach at K = 999 "
                    "(27 H_27 / 1000), so the BY reading is limited by the permutation resolution."),
        per_draw_counts=dict(n_bh_q_le_10=n_bh, n_by_q_le_10=n_by),
        candidates=out_c, wall_clock_s=wall)
    json.dump(out, open(os.path.join(E8, "feature_draws.json"), "w"), indent=1)
    print(f"\n  {'stn':<5}{'pair':<7}{'date':<10}{'p_K999':>7}{'min':>7}{'med':>7}"
          f"{'max':>7}{'fr<=.05':>9}{'BYq_med':>9}")
    for c in out_c:
        print(f"  {c['stn']:<5}{c['pair']:<7}{c['date']:<10}{c['p_unrestricted']:>7.3f}"
              f"{c['p_min']:>7.3f}{c['p_median']:>7.3f}"
              f"{c['p_max']:>7.3f}{c['frac_le_05']:>9.2f}{c['by_q_median']:>9.3f}")
    print(f"  per draw: #BH q<=0.10 {n_bh}; #BY q<=0.10 {n_by}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--summarise-only", action="store_true",
                    help="rebuild feature_draws.json from the stored per-draw records")
    a_ = ap.parse_args()
    retest_rows = json.load(open(os.path.join(E8, "retest_K999.json")))["results"]
    summ = json.load(open(os.path.join(E8, "retest_summary.json")))
    rec = os.path.join(E8, "records_feature_draws_perm.csv.gz")
    if a_.summarise_only:
        with gzip.open(rec, "rt", newline="") as f:
            res = list(csv.DictReader(f))
        wall = json.load(open(os.path.join(E8, "feature_draws.json"))).get("wall_clock_s")
        return summarise(res, retest_rows, summ, wall)
    cands = [r for r in retest_rows if r["p_block"] <= 0.05]
    jobs = [(r["stn"], r["pair"], r["interval"], r["date"], j)
            for r in cands for j in range(1, N_DRAWS + 1)]
    print(f"{len(cands)} candidates x {N_DRAWS} draws = {len(jobs)} stage-two tests at K={K}",
          flush=True)
    t0 = time.time()
    with Pool(a_.procs, maxtasksperchild=4) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    wall = round(time.time() - t0, 1)
    with gzip.open(rec, "wt", newline="") as f:
        keys = ["stn", "pair", "date", "interval", "draw", "seed_x", "seed_y", "p", "T_obs", "T_rep"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(res)
    summarise(res, retest_rows, summ, wall)
    print(f"  wall clock {wall:.0f}s")


if __name__ == "__main__":
    main()
