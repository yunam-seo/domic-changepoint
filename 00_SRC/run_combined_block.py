#!/usr/bin/env python
"""The combined statistic of Section 4.5 on the daily 2021-2022 stock-bond window of Section 6.8, under pair
permutation and under block permutation at b = 20 and 50 (Supplementary Section B.18).

DOMI, HSIC, Spearman and the copula statistic from global ranks are computed on the observed window and on
K = 999 permuted copies (the same copies for every statistic); each curve is studentized by its pointwise
moments over all K + 1 curves; the combined statistic is the larger of the studentized DOMI and Spearman
maxima, calibrated by the same copies. p = (1 + #{T_k >= T_0}) / (K + 1) with the tie rule of dots/perm.py.

Run:    python 00_SRC/run_combined_block.py
Writes 04_DAOU/EXPERIMENT/combined_block/{config.json, window.json, records_T.csv.gz}
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.domi import DOMIContext, domi_stats, DEP_BASELINES  # noqa: E402
from dots.perm import ge  # noqa: E402
from dots.segtests import _perm_index  # noqa: E402
from run_e6_finance import load, align_pair  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "combined_block")
CFG = dict(D=8, w=60, K=999, blocks=[20, 50], seed=20260928, window=["20210101", "20221231"])
STATS = ["DOMI-diff", "HSIC-diff", "Spearman-diff", "CopulaCvM"]


def curves(x, y):
    ctx = DOMIContext(x[:, None], y[:, None], CFG["w"], D=CFG["D"], seed=2026, n_perm=0)
    return ctx.grid, {"DOMI-diff": domi_stats(ctx, "global")["DOMI-diff"],
                      "HSIC-diff": DEP_BASELINES["HSIC-diff"](ctx, "global"),
                      "Spearman-diff": DEP_BASELINES["Spearman-diff"](ctx, "global"),
                      "CopulaCvM": DEP_BASELINES["CopulaCvM"](ctx, "global")}


def scan_test(x, y, block):
    grid, obs = curves(x, y)
    rng = np.random.default_rng(CFG["seed"])
    reps = {m: [] for m in STATS}
    for _ in range(CFG["K"]):
        idx = _perm_index(len(x), rng, block)
        _, c = curves(x[idx], y[idx])
        for m in STATS:
            reps[m].append(c[m])
    Tall, tau = {}, {}
    for m in STATS:
        A = np.vstack([obs[m][None, :], np.array(reps[m])])
        S = (A - A.mean(0)) / (A.std(0) + 1e-12)
        Tall[m] = S.max(1)
        tau[m] = int(grid[int(np.argmax(S[0]))])
    Tall["combined"] = np.maximum(Tall["DOMI-diff"], Tall["Spearman-diff"])
    tau["combined"] = tau["DOMI-diff"] if Tall["DOMI-diff"][0] >= Tall["Spearman-diff"][0] else tau["Spearman-diff"]
    p = {m: (1 + int(sum(ge(t, T[0]) for t in T[1:]))) / (CFG["K"] + 1) for m, T in Tall.items()}
    return p, tau, Tall


def main():
    os.makedirs(OUT, exist_ok=True)
    json.dump(CFG, open(os.path.join(OUT, "config.json"), "w"), indent=1)
    dates, x, y = align_pair(load("yahoo_gspc.csv"), load("yahoo_tnx.csv"), ("logret", "diff"))
    sel = [i for i, d in enumerate(dates) if CFG["window"][0] <= d <= CFG["window"][1]]
    xs, ys = x[sel], y[sel]
    out, rows = {}, []
    for calib, b in [("pair", None)] + [(f"block b={b}", b) for b in CFG["blocks"]]:
        t0 = time.time()
        p, tau, Tall = scan_test(xs, ys, b)
        out[calib] = dict(p=p, tau_date={m: dates[sel[0] + t] for m, t in tau.items()}, n=len(sel))
        rows += [dict(calibration=calib, stat=m, replica=i, T=repr(float(t))) for m, T in Tall.items() for i, t in enumerate(T)]
        print(calib, {m: round(v, 3) for m, v in p.items()}, f"{time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(OUT, "window.json"), "w"), indent=1)
    with gzip.open(os.path.join(OUT, "records_T.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["calibration", "stat", "replica", "T"], lineterminator="\n")
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
