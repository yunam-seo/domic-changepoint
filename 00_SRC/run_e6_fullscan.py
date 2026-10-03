#!/usr/bin/env python
"""Financial pairs on the raw returns: whole-record single-break scans and the combined statistic under
block permutation (exploratory scans that preceded the protocol of Supplementary Section B.22).

part window   the daily 2021-2022 stock-bond window of Section 6.8: DOMI, Spearman and the combined
              statistic T = max(T_DOMI, T_Spearman) from the same replicas, under pair permutation and block
              permutation at b = 20 and 50; K = 999, a single permutation draw, superseded in B.18 and
              Table B.19 by run_combined_block_seeds.py
part full     each pair over the whole daily record 2011-2026: one single-break scan (candidates n/10 ..
              9n/10), calibrated by block permutation at the block length the diagnostic of Section 4.3
              recommends for the whole record (run_exch_diag.exch_diag, L = 20, K = 199); DOMI, HSIC,
              Spearman, the copula statistic and the combined statistic from the same replicas; K = 199.
              The scan answers whether the dependence changed anywhere in the record; the location is an
              estimate. Exploratory: the block length is chosen from the data (Section 4.3).
Pairs as in run_e6_finance (SPX-TNX, KOSPI-USDKRW) and KOSPI-SPXlag (KOSPI on Korean date d against the S&P
500 return of the last US trading date before d).

Run:    python 00_SRC/run_e6_fullscan.py --part window
        python 00_SRC/run_e6_fullscan.py --part full --procs 3
Writes 04_DAOU/EXPERIMENT/e6_fullscan/{config.json, window.json, full.json, records_T.csv.gz}
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
import time  # noqa: E402

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.domi import DOMIContext, domi_stats, DEP_BASELINES  # noqa: E402
from dots.perm import ge  # noqa: E402
from dots.segtests import _perm_index  # noqa: E402
from run_e6_finance import load  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e6_fullscan")
CFG = dict(D=8, window_w=60, window_K=999, window_blocks=[20, 50], full_K=199, full_w_frac=0.1, diag_L=20,
           diag_K=199, seed=20260928)
STATS = ["DOMI-diff", "HSIC-diff", "Spearman-diff", "CopulaCvM"]


def returns(series, kind):
    d = sorted(series)
    x = np.array([series[k] for k in d])
    return d[1:], (np.diff(np.log(x)) if kind == "logret" else np.diff(x))


def pairs():
    spx, tnx = load("yahoo_gspc.csv"), load("yahoo_tnx.csv")
    ksp, fx = load("yahoo_kospi.csv"), load("yahoo_usdkrw.csv")
    R = {"SPX": returns(spx, "logret"), "TNX": returns(tnx, "diff"), "KOSPI": returns(ksp, "logret"),
         "USDKRW": returns(fx, "logret")}
    out = {}
    for name, (a, b) in (("SPX-TNX", ("SPX", "TNX")), ("KOSPI-USDKRW", ("KOSPI", "USDKRW"))):
        (da, ra), (db, rb) = R[a], R[b]
        ib = {d: i for i, d in enumerate(db)}
        ia = {d: i for i, d in enumerate(da)}
        common = [d for d in da if d in ib]
        out[name] = (common, ra[[ia[d] for d in common]], rb[[ib[d] for d in common]])
    (dk, rk), (du, ru) = R["KOSPI"], R["SPX"]
    j, keep, lag = 0, [], []
    for i, d in enumerate(dk):
        while j + 1 < len(du) and du[j + 1] < d:
            j += 1
        if du[j] < d:
            keep.append(i); lag.append(j)
    out["KOSPI-SPXlag"] = ([dk[i] for i in keep], rk[keep], ru[lag])
    return out


def curves(x, y, w):
    ctx = DOMIContext(x[:, None], y[:, None], w, D=CFG["D"], seed=2026, n_perm=0)
    return ctx.grid, {"DOMI-diff": domi_stats(ctx, "global")["DOMI-diff"],
                      "HSIC-diff": DEP_BASELINES["HSIC-diff"](ctx, "global"),
                      "Spearman-diff": DEP_BASELINES["Spearman-diff"](ctx, "global"),
                      "CopulaCvM": DEP_BASELINES["CopulaCvM"](ctx, "global")}


def scan_test(x, y, w, K, block, seed):
    """Algorithm 1 with symmetric studentization; returns per-statistic and combined p-values and the
    studentized maxima of all K + 1 curves."""
    grid, obs = curves(x, y, w)
    rng = np.random.default_rng(seed)
    reps = {m: [] for m in STATS}
    for _ in range(K):
        idx = _perm_index(len(x), rng, block)
        _, c = curves(x[idx], y[idx], w)
        for m in STATS:
            reps[m].append(c[m])
    Tall, tau = {}, {}
    for m in STATS:
        A = np.vstack([obs[m][None, :], np.array(reps[m])])
        mu, sd = A.mean(0), A.std(0) + 1e-12
        S = (A - mu) / sd
        Tall[m] = S.max(1)
        tau[m] = int(grid[int(np.argmax(S[0]))])
    Tall["combined"] = np.maximum(Tall["DOMI-diff"], Tall["Spearman-diff"])
    tau["combined"] = tau["DOMI-diff"] if Tall["DOMI-diff"][0] >= Tall["Spearman-diff"][0] else tau["Spearman-diff"]
    p = {m: (1 + int(sum(ge(t, T[0]) for t in T[1:]))) / (K + 1) for m, T in Tall.items()}
    return p, tau, Tall


def _write_T(rows):
    path = os.path.join(OUT, "records_T.csv.gz")
    new = not os.path.exists(path)
    with gzip.open(path, "at", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["analysis", "calibration", "stat", "replica", "T"], lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows(rows)


def _rows(analysis, calib, Tall):
    return [dict(analysis=analysis, calibration=calib, stat=m, replica=i, T=repr(float(t)))
            for m, T in Tall.items() for i, t in enumerate(T)]


def part_window():
    dates, x, y = pairs()["SPX-TNX"]
    sel = [i for i, d in enumerate(dates) if "20210101" <= d <= "20221231"]
    xs, ys = x[sel], y[sel]
    out = {}
    for calib, b in [("pair", None)] + [(f"block b={b}", b) for b in CFG["window_blocks"]]:
        t0 = time.time()
        p, tau, Tall = scan_test(xs, ys, CFG["window_w"], CFG["window_K"], b, CFG["seed"])
        out[calib] = dict(p=p, tau_date={m: dates[sel[0] + t] for m, t in tau.items()}, n=len(sel),
                          seconds=round(time.time() - t0, 1))
        _write_T(_rows("SPX-TNX 2021-2022", calib, Tall))
        print(calib, {m: round(v, 3) for m, v in p.items()}, f"{time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(OUT, "window.json"), "w"), indent=1)


def _full_job(name):
    from run_exch_diag import exch_diag
    dates, x, y = pairs()[name]
    d = exch_diag(x, y, L=CFG["diag_L"], K=CFG["diag_K"], seed=0)
    b = None if d["decision"] == "pair" else int(d["b_hat"])
    w = int(CFG["full_w_frac"] * len(x))
    t0 = time.time()
    p, tau, Tall = scan_test(x, y, w, CFG["full_K"], b, CFG["seed"])
    res = dict(n=len(x), first=dates[0], last=dates[-1], w=w, diagnostic=d["decision"], block=b,
               n_blocks=(len(x) // b) if b else len(x), diag_pvalues=d["pvalues"], p=p,
               tau_date={m: dates[t] for m, t in tau.items()}, seconds=round(time.time() - t0, 1))
    return name, res, _rows(f"{name} full record", f"block b={b}" if b else "pair", Tall)


def part_full(procs):
    from multiprocessing import Pool
    out = {}
    with Pool(procs) as pool:
        for name, res, rows in pool.imap_unordered(_full_job, ["SPX-TNX", "KOSPI-USDKRW", "KOSPI-SPXlag"]):
            out[name] = res
            _write_T(rows)
            print(name, res["diagnostic"], "b", res["block"], "blocks", res["n_blocks"],
                  {m: (round(v, 3), res["tau_date"][m]) for m, v in res["p"].items()}, f"{res['seconds']}s", flush=True)
    json.dump(out, open(os.path.join(OUT, "full.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["window", "full"])
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cp = os.path.join(OUT, "config.json")
    if not os.path.exists(cp):
        json.dump(CFG, open(cp, "w"), indent=1)
    elif json.load(open(cp)) != CFG:
        sys.exit("config.json differs from CFG: use a new output folder")
    if a.part == "window":
        part_window()
    else:
        part_full(a.procs)


if __name__ == "__main__":
    main()
