#!/usr/bin/env python
"""Permutation-draw variability of the 2021-2022 stock-bond window p-values (Sections 6.8 and B.18).

With few blocks the studentized maxima of Algorithm 1 depend on the drawn replicas (their pointwise moments enter
the statistic), so a single draw of K replicas gives one realization of the p-value. This runs the test of
run_combined_block.py for 20 independent permutation draws (seeds 1..20, K = 999 each) under pair permutation and
block permutation at b = 20 and 50, and reports per statistic the median and range over the draws.

Run:    python 00_SRC/run_combined_block_seeds.py --procs 20
Writes 04_DAOU/EXPERIMENT/combined_block_seeds20/{config.json, draws.jsonl, summary.json}
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
import run_combined_block as CB  # noqa: E402
from run_e6_finance import load, align_pair  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "combined_block_seeds20")
CFG = dict(seeds=list(range(1, 21)), K=999, calibrations=["pair", 20, 50], window=CB.CFG["window"])


def job(a):
    seed, b = a
    dates, x, y = align_pair(load("yahoo_gspc.csv"), load("yahoo_tnx.csv"), ("logret", "diff"))
    sel = [i for i, d in enumerate(dates) if CFG["window"][0] <= d <= CFG["window"][1]]
    CB.CFG["seed"] = seed
    p, _, _ = CB.scan_test(x[sel], y[sel], None if b == "pair" else b)
    return dict(seed=seed, calibration=str(b), p=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    json.dump(CFG, open(os.path.join(OUT, "config.json"), "w"), indent=1)
    path = os.path.join(OUT, "draws.jsonl")
    done = {(r["seed"], r["calibration"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    jobs = [(s, b) for b in CFG["calibrations"] for s in CFG["seeds"] if (s, str(b)) not in done]
    with Pool(a.procs) as pool:
        for rec in pool.imap_unordered(job, jobs):
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
    rows = [json.loads(l) for l in open(path)]
    summ = {}
    for b in map(str, CFG["calibrations"]):
        rr = [r for r in rows if r["calibration"] == b]
        summ[b] = {m: dict(median=float(np.median([r["p"][m] for r in rr])), min=float(min(r["p"][m] for r in rr)),
                           max=float(max(r["p"][m] for r in rr)), n_draws=len(rr),
                           share_le_005=float(np.mean([r["p"][m] <= 0.05 for r in rr])))
                   for m in rr[0]["p"]}
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
