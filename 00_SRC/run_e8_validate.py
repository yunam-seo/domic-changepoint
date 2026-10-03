#!/usr/bin/env python
"""Random-date rejection frequencies of the weather stage-two tests (Section 6.7; Supplementary B.8).

Purpose
-------
Under the unrestricted scheme the stage-two block-permutation tests permute six-week super-blocks
freely within the +/- 1 year window. Block exchangeability is doubtful for these records (Section
6.7 reports it rejected for the log block variance in 32 of 36 series), so the rejection frequency
of stage two is measured at dates chosen without any stage one.

  For each of the 36 station-pair series, 20 random dates are
      drawn uniformly without replacement (seed [20260926, series index]) from the grid intervals at least 26
      intervals (one year) from either end, no stage one involved. At each date the stage-two
      dependence test (run_e8_retest._job) and marginal-scale test (run_e8_marginal._job) are
      run, K = 999. Under a valid null and no change at the random date the dependence p-values
      are uniform; because genuine dependence changes may lie inside some windows, the
      random-date rejection frequency is a diagnostic of calibration, not a verified size.

The same 720 dates are used by run_e8_season.py for the season-restricted calibration.

Run
---
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_e8_validate.py --procs 20

Inputs:  02_MART/WEATHER_HOURLY_ANOM.npz
Outputs (04_DAOU/EXPERIMENT/e8_validate/):
  results.json               summaries (rejection frequencies at 0.10, 0.05, 0.01; quantiles)
  records_tests.csv.gz       one row per test: part, series, date, interval, test kind, p, observed
                             statistic and all K replica maxima (every p-value regenerates);
                             about 7 MB, so it is not shipped with the stored outputs -- rerunning
                             this script regenerates it
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
from scipy import stats

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
import run_e8_retest as RETEST  # noqa: E402
import run_e8_marginal as MARG  # noqa: E402
from run_e8_hourly import PAIRS  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8_validate")
ANOM = os.path.join(ROOT, "02_MART", "WEATHER_HOURLY_ANOM.npz")
K = 999
UNIT, SUPER, HALF, LO = 336, 3, 26, 3        # stage-two geometry of run_e8_retest.py
N_RANDOM_DATES = 20


def load(stn, u, v):
    z = np.load(ANOM)
    a, b = z[f"{stn}|{u}"], z[f"{stn}|{v}"]
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok], z[f"{stn}|tm"][ok]


def window(a, c):
    ef = np.arange(0, len(a) + 1, UNIT)
    if ef[-1] != len(a):
        ef = np.append(ef, len(a))
    B = len(ef) - 1
    lo, hi = max(0, c - HALF), min(B, c + HALF)
    return slice(ef[lo], ef[hi]), B


# ---------------------------------------------------------------- dispatcher
def _run(job):
    part, stn, pair, c, date, kind = job
    t0 = time.time()
    if kind == "dep":
        o = RETEST._job((stn, pair, c, date))
        r = dict(p=float(o["p_block"]), T=[o["T_obs"]] + [float(v) for v in o["T_rep"].split()])
    else:
        o = MARG._job((stn, pair, c, date))
        r = dict(p=float(o["p_marg_min"]), p_marg_x=o["p_marg_x"], p_marg_y=o["p_marg_y"],
                 T=list(o["_T"].values())[0].split() + ["|"] + list(o["_T"].values())[1].split())
    r.update(part=part, stn=stn, pair=pair, interval=c, date=date, kind=kind,
             secs=round(time.time() - t0, 1))
    return r


def randomdate_dates():
    z = np.load(ANOM)
    stns = sorted({k.split("|")[0] for k in z.files})
    out = []
    for si, (stn, (u, v)) in enumerate([(s, pr) for s in stns for pr in PAIRS]):
        a, b, tm = load(stn, u, v)
        B = int(np.ceil(len(a) / UNIT))
        rng = np.random.default_rng([20260926, si])
        cs = rng.choice(np.arange(HALF, B - HALF + 1), size=N_RANDOM_DATES, replace=False)
        out += [(stn, f"{u}-{v}", int(c), str(tm[c * UNIT])[:8], j) for j, c in enumerate(cs)]
    return out


def binom_ref(k, n, p0):
    return dict(k=int(k), n=int(n), rate=k / n, expected=p0,
                binom_sf=float(stats.binom.sf(k - 1, n, p0)))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=20)
    a_ = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rdates = randomdate_dates()
    jobs = []
    for stn, pair, c, date, j in rdates:
        jobs += [("RD", stn, pair, c, date, "dep"), ("RD", stn, pair, c, date, "marg")]
    print(f"{len(jobs)} tests at K={K}", flush=True)
    t0 = time.time()
    with Pool(a_.procs, maxtasksperchild=10) as pool:
        res = pool.map(_run, jobs, chunksize=1)
    wall = time.time() - t0

    with gzip.open(os.path.join(OUT, "records_tests.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["part", "stn", "pair", "date", "interval", "kind", "p",
                                          "p_marg_x", "p_marg_y", "T"])
        w.writeheader()
        for r in res:
            T = r.get("T", [])
            w.writerow(dict(part=r["part"], stn=r["stn"], pair=r["pair"], date=r["date"],
                            interval=r["interval"], kind=r["kind"], p=r["p"],
                            p_marg_x=r.get("p_marg_x", ""), p_marg_y=r.get("p_marg_y", ""),
                            T=" ".join(str(t) if isinstance(t, str) else f"{t:.6f}" for t in T)))
    def randomdate_summary(part, kind):
        rr = [r for r in res if r["part"] == part and r["kind"] == kind]
        p = np.array([r["p"] for r in rr])
        n = len(p)
        per = {}
        for r in rr:
            per.setdefault(f"{r['stn']}|{r['pair']}", []).append(r["p"])
        qs = [0.01, 0.05, 0.10, 0.25, 0.5]
        return dict(
            n=n, le_05=binom_ref(int((p <= 0.05).sum()), n, 0.05),
            le_01=binom_ref(int((p <= 0.01).sum()), n, 0.01),
            le_10=binom_ref(int((p <= 0.10).sum()), n, 0.10),
            qq=dict(uniform_quantiles=qs, empirical_quantiles=[float(np.quantile(p, q)) for q in qs],
                    ks_uniform_p=float(stats.kstest(p, "uniform").pvalue),
                    mean_p=float(p.mean())),
            per_series={s: dict(n=len(v), le_05=int(np.sum(np.array(v) <= 0.05)),
                                le_01=int(np.sum(np.array(v) <= 0.01)),
                                binom_sf_le_05=float(stats.binom.sf(np.sum(np.array(v) <= 0.05) - 1,
                                                                    len(v), 0.05)))
                        for s, v in sorted(per.items())},
            n_series_with_ge3_of_20_le_05=int(sum(np.sum(np.array(v) <= 0.05) >= 3
                                                 for v in per.values())))

    rd_dep, rd_marg = randomdate_summary("RD", "dep"), randomdate_summary("RD", "marg")
    out = dict(config=dict(K=K, unit_hours=UNIT, super_blocks=SUPER, half_window=HALF, lo=LO,
                           n_randomdate_per_series=N_RANDOM_DATES,
                           randomdate_seed="[20260926, series index]",
                           note_marginal="the marginal p is min(p_X, p_Y); its null reference rate "
                                         "at 0.05 is at most 1-(0.95)^2 = 0.0975"),
               dependence=rd_dep, marginal=rd_marg, wall_clock_s=round(wall, 1),
               cpu_s=round(sum(r["secs"] for r in res), 1))
    json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    for nm, s in (("dependence", rd_dep), ("marginal", rd_marg)):
        print(nm, s["le_05"], s["le_01"], s["qq"], "series>=3/20:", s["n_series_with_ge3_of_20_le_05"])
    print(f"wall {wall:.0f}s, cpu {out['cpu_s']:.0f}s")


if __name__ == "__main__":
    main()
