#!/usr/bin/env python
"""Weather stage two at K = 9999 for the candidates with K = 999 stage-two p <= 0.05, under the unrestricted
(all-year) block permutation of Supplementary Section B.8. Its output retest_K9999.json is read by the
whole-procedure check of Supplementary Section B.14 (run_e8_fullperm.py). The weather numbers of Section 6.7
come from the season-restricted stage two of run_e8_season.py, not from this script.

Purpose
-------
The 27 stage-two p-values are read under Benjamini-Hochberg (BH) and Benjamini-Yekutieli (BY), the
latter valid under arbitrary dependence. At K = 999 the smallest attainable permutation p-value is
0.001, and 27 * H_27 * 0.001 = 0.105 is the smallest BY q-value a lone top candidate can reach, so
at that K the BY reading is limited by the permutation resolution. This script lowers the floor to
0.0001 by repeating the stage-two tests at K = 9999 for the candidates with K = 999 p <= 0.05, and
recomputes the BH and BY q-values over all 27 candidates, the others held at their K = 999
p-values (which are larger than 0.05, so the ranks that matter are unaffected by their resolution).

Machinery: the dependence test is run_e8_retest._job and the marginal-scale test is
run_e8_marginal._job, unchanged, with the module constant K raised to 9999: the same +/- 1 year
windows, six-week super-blocks, feature draw (seeds 2026 / 2027) and permutation seeds as the
K = 999 run. numpy's Generator draws the permutations sequentially from the same seed, so the
first 999 replicas are exactly those of the K = 999 run and the K = 9999 run extends them.
Classification as in run_e8_marginal.py: dependence passes at BH q <= 0.10, marginal change at
min(p_X, p_Y) <= 0.05.

Run
---
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_e8_retest_hiK.py --procs 6

Inputs:  02_MART/WEATHER_HOURLY_ANOM.npz; 04_DAOU/EXPERIMENT/e8/{results.json, retest_K999.json,
         marginal_diagnostic.json}
Outputs: 04_DAOU/EXPERIMENT/e8/retest_K9999.json
         04_DAOU/EXPERIMENT/e8/records_retest_K9999_perm.csv.gz (per test: observed statistic
         and all 9999 replica maxima; every p-value regenerates from these rows)
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
import run_e8_retest as RETEST  # noqa: E402
import run_e8_marginal as MARG  # noqa: E402
from run_e8_summary import step_up_q  # noqa: E402

E8 = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8")
K_HI = 9999
RETEST.K = K_HI        # module constants read by the unchanged _job functions (inherited by workers)
MARG.K = K_HI


def _run(args):
    kind, job = args
    return kind, (RETEST._job(job) if kind == "dep" else MARG._job(job))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    a_ = ap.parse_args()
    retest_rows = json.load(open(os.path.join(E8, "retest_K999.json")))["results"]
    marg = {(r["stn"], r["pair"], r["interval"]): r
            for r in json.load(open(os.path.join(E8, "marginal_diagnostic.json")))["results"]}
    sel = [r for r in retest_rows if r["p_block"] <= 0.05]
    jobs = [(k, (r["stn"], r["pair"], r["interval"], r["date"])) for r in sel for k in ("dep", "marg")]
    print(f"{len(sel)} candidates, {len(jobs)} tests at K={K_HI}", flush=True)
    t0 = time.time()
    with Pool(a_.procs, maxtasksperchild=1) as pool:
        out = pool.map(_run, jobs, chunksize=1)
    wall = round(time.time() - t0, 1)

    dep = {(r["stn"], r["pair"], r["interval"]): r for k, r in out if k == "dep"}
    mg = {(r["stn"], r["pair"], r["interval"]): r for k, r in out if k == "marg"}
    with gzip.open(os.path.join(E8, "records_retest_K9999_perm.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stn", "pair", "date", "interval", "test", "p", "T"])
        w.writeheader()
        for key, r in dep.items():
            w.writerow(dict(stn=r["stn"], pair=r["pair"], date=r["date"], interval=r["interval"],
                            test="dependence", p=r["p_block"],
                            T=f"{r['T_obs']:.6f} " + r["T_rep"]))
        for key, r in mg.items():
            u, v = r["pair"].split("-")
            for name, pv in ((u, r["p_marg_x"]), (v, r["p_marg_y"])):
                w.writerow(dict(stn=r["stn"], pair=r["pair"], date=r["date"], interval=r["interval"],
                                test=f"marginal:{name}", p=pv, T=r["_T"][name]))

    keys = [(r["stn"], r["pair"], r["interval"]) for r in retest_rows]
    p27 = np.array([dep[k]["p_block"] if k in dep else r["p_block"] for k, r in zip(keys, retest_rows)])
    m = len(p27)
    H = float(np.sum(1.0 / np.arange(1, m + 1)))
    bh, by = step_up_q(p27), step_up_q(p27, H)
    rows = []
    for i, (k, r) in enumerate(zip(keys, retest_rows)):
        hi = k in dep
        pm = (min(mg[k]["p_marg_x"], mg[k]["p_marg_y"]) if hi else marg[k]["p_marg_min"])
        d_ok, m_ok = bh[i] <= 0.10, pm <= 0.05
        rows.append(dict(stn=k[0], pair=k[1], date=r["date"], interval=k[2],
                         K=K_HI if hi else 999, p_dep_K999=r["p_block"],
                         p_dep=float(p27[i]), bh_q=round(float(bh[i]), 4), by_q=round(float(by[i]), 4),
                         p_marg_x=(mg[k]["p_marg_x"] if hi else marg[k]["p_marg_x"]),
                         p_marg_y=(mg[k]["p_marg_y"] if hi else marg[k]["p_marg_y"]),
                         p_marg_min=pm,
                         p_marg_min_K999=marg[k]["p_marg_min"],
                         cls=("dependence change" if d_ok and not m_ok else "both change"
                              if d_ok and m_ok else "marginal-driven" if m_ok else "undetermined")))
    rows.sort(key=lambda r: r["p_dep"])
    res = dict(config=dict(K=K_HI, retested="stage-two p <= 0.05 at K=999 (5 candidates)",
                           others="K=999 p-values of run_e8_retest.py", unit_hours=RETEST.UNIT,
                           super_blocks=RETEST.SUPER, feature_seeds=[2026, 2027],
                           dep_perm_seed="4242 + interval", harmonic_number=H,
                           by_floor_single_candidate=m * H / (K_HI + 1)),
               n_bh_q_le_10=int(np.sum(bh <= 0.10)), n_bh_q_le_05=int(np.sum(bh <= 0.05)),
               n_by_q_le_10=int(np.sum(by <= 0.10)), n_by_q_le_05=int(np.sum(by <= 0.05)),
               results=rows, wall_clock_s=wall)
    json.dump(res, open(os.path.join(E8, "retest_K9999.json"), "w"), indent=1)
    print(f"  {'stn':<5}{'pair':<7}{'date':<10}{'K':>6}{'p999':>8}{'p':>8}{'BH q':>8}{'BY q':>8}"
          f"{'p_marg':>8}  class")
    for r in rows[:8]:
        print(f"  {r['stn']:<5}{r['pair']:<7}{r['date']:<10}{r['K']:>6}{r['p_dep_K999']:>8.4f}"
              f"{r['p_dep']:>8.4f}{r['bh_q']:>8.4f}{r['by_q']:>8.4f}{r['p_marg_min']:>8.4f}  {r['cls']}")
    print(f"  BH<=.10: {res['n_bh_q_le_10']}  BH<=.05: {res['n_bh_q_le_05']}  "
          f"BY<=.10: {res['n_by_q_le_10']}  BY<=.05: {res['n_by_q_le_05']}   ({wall:.0f}s)")


if __name__ == "__main__":
    main()
