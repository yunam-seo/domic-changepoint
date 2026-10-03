#!/usr/bin/env python
"""Re-check of the level at the highest tie share of run_tie_check.py (about 91% of observations tied).

The reported run (400 replicates, replicates 0-399) gave rejection rates of 7.25-7.75% for all three
implementations at this tie share, including 'random' and 'recompute', which Proposition P1 covers (see
run_tie_check.py). This run draws 2000 NEW independent replicates (replicate indices 400-2399, the same
generator, rounding step, permutations and statistic as run_tie_check._job) and reports, per
implementation, the number and rate of rejections at 0.05 with an exact (Clopper-Pearson) 95% interval.
The 400-replicate results are not replaced.

Run:    OMP_NUM_THREADS=1 python 00_SRC/run_tie_check_extreme.py --procs 22
Writes 04_DAOU/EXPERIMENT/tie_check_extreme/{config.json, shards/, records.csv.gz, results.json}
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import beta  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
import run_tie_check as TC  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "tie_check_extreme")
CFG = dict(TC.CFG, level="extreme", step=TC.STEPS["extreme"], rep_first=400, rep_count=2000)
MODES = ("order", "random", "recompute")


def job(rep):
    p = os.path.join(OUT, "shards", f"{rep:04d}.json")
    if os.path.exists(p):
        return
    rec = TC._job(("extreme", rep))
    tmp = p + ".tmp"
    json.dump(rec, open(tmp, "w"))
    os.replace(tmp, p)


def cp_interval(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def aggregate():
    reps = range(CFG["rep_first"], CFG["rep_first"] + CFG["rep_count"])
    recs = [json.load(open(os.path.join(OUT, "shards", f"{r:04d}.json"))) for r in reps
            if os.path.exists(os.path.join(OUT, "shards", f"{r:04d}.json"))]
    with gzip.open(os.path.join(OUT, "records.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]), lineterminator="\n"); w.writeheader(); w.writerows(recs)
    n = len(recs)
    res = {"replicates": n, "tie_share_mean": float(np.mean([(r["tie_x"] + r["tie_y"]) / 2 for r in recs]))}
    for m in MODES:
        k = int(sum(r[f"p_{m}"] <= CFG["alpha"] for r in recs))
        lo, hi = cp_interval(k, n)
        res[m] = dict(rejections=k, rate=k / n, ci95=[lo, hi])
    json.dump(dict(config=CFG, results=res), open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "shards"), exist_ok=True)
    cp = os.path.join(OUT, "config.json")
    if not os.path.exists(cp):
        json.dump(CFG, open(cp, "w"), indent=1)
    elif json.load(open(cp)) != json.loads(json.dumps(CFG)):
        sys.exit("config.json differs from CFG: use a new output folder")
    if not a.aggregate:
        with Pool(a.procs, maxtasksperchild=20) as pool:
            for _ in pool.imap_unordered(job, range(CFG["rep_first"], CFG["rep_first"] + CFG["rep_count"]), chunksize=2):
                pass
    aggregate()


if __name__ == "__main__":
    main()
