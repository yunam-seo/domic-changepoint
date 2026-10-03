#!/usr/bin/env python
"""Financial application (Section 6.8): dependence regimes in financial pairs.

Data (01_ORG/FINANCE, Yahoo chart API, daily close, 2011-2026):
  Pair A (US): S&P500 log-return  vs  Δ(10Y Treasury yield)      — the stock-bond regime
  Pair B (KR): KOSPI log-return   vs  USD/KRW log-return          — equity-FX coupling

(a)  weekly segmentation (output keys e6a_*): weekly aggregates (sum of daily log-returns / yield changes), full 15y:
     Holevo-partitioning segmentation (pair-permutation bias-corrected cost, beta at <=5%
     spurious rate on permuted copies) vs rank-Gaussian PELT.
(b)  daily two-year windows (output keys e6b_*): single-break pair-permutation tests (K=99):
     A: 2021-2022 (inflation regime flip), B: 2019-2020 (COVID).
Daily returns show volatility clustering, so pair permutation is not valid at either
resolution (the diagnostic of run_exch_diag.py rejects for the daily window and for the
weekly aggregates alike): the daily tests are recomputed here under block permutation at
b = 20 and 50, and the weekly segmentation is recalibrated in run_e6a_block.py.

Writes 04_DAOU/EXPERIMENT/e6/{results.json, records_perm.csv, e6_figure.png}
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

import dots.segtests as SEG  # noqa: E402
from dots.segtests import pelt_segment, gauss_segment, single_break_test  # noqa: E402
from dots.domi import DOMIContext  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e6")
FIN = os.path.join(ROOT, "01_ORG", "FINANCE")


def load(name):
    rows = list(csv.DictReader(open(os.path.join(FIN, name))))
    return {r["date"]: float(r["close"]) for r in rows}


def align_pair(a: dict, b: dict, transform=("logret", "logret")):
    dates = sorted(set(a) & set(b))
    xa = np.array([a[d] for d in dates])
    xb = np.array([b[d] for d in dates])

    def tr(x, kind):
        return np.diff(np.log(x)) if kind == "logret" else np.diff(x)

    return dates[1:], tr(xa, transform[0]), tr(xb, transform[1])


def weekly_sum(x, k=5):
    m = len(x) // k
    return x[: m * k].reshape(m, k).sum(1)


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    spx, tnx = load("yahoo_gspc.csv"), load("yahoo_tnx.csv")
    ksp, fx = load("yahoo_kospi.csv"), load("yahoo_usdkrw.csv")
    pairs = {
        "SPX-TNX": align_pair(spx, tnx, ("logret", "diff")),
        "KOSPI-USDKRW": align_pair(ksp, fx, ("logret", "logret")),
    }
    betas = list(np.linspace(0.5, 12, 24))
    for name, (dates, x, y) in pairs.items():
        xw, yw = weekly_sum(x), weekly_sum(y)
        wk_dates = dates[::5][: len(xw)]
        t0 = time.time()
        SEG.PERM_TAG = f"e6a_{name}"
        obs, fa = pelt_segment(xw, yw, step=4, betas=betas)
        beta = next((b for b in betas if fa[b] <= 0.05), betas[-1])
        SEG.PERM_TAG = f"e6a_{name}|gauss"
        gobs, gfa = gauss_segment(xw, yw, step=4, betas=betas)
        gbeta = next((b for b in betas if gfa[b] <= 0.05), betas[-1])
        res[f"e6a_{name}"] = dict(
            n_weeks=len(xw), beta=beta, fa=fa[beta],
            cps_dates=[wk_dates[min(c, len(wk_dates) - 1)] for c in obs[beta]],
            gauss_n_cps=len(gobs[gbeta]),
            gauss_cps_dates=[wk_dates[min(c, len(wk_dates) - 1)] for c in gobs[gbeta]])
        print(f"E6a {name}: {time.time()-t0:.0f}s Holevo {res[f'e6a_{name}']['cps_dates']} "
              f"(β={beta:.1f}) | Gauss {res[f'e6a_{name}']['gauss_n_cps']} cps", flush=True)
    for name, span in [("SPX-TNX", ("20210101", "20221231")), ("KOSPI-USDKRW", ("20190101", "20201231"))]:
        dates, x, y = pairs[name]
        sel = [i for i, d in enumerate(dates) if span[0] <= d <= span[1]]
        t0 = time.time()
        SEG.PERM_TAG = f"e6b_{name}_{span[0][:4]}-{span[1][:4]}"
        r = single_break_test(x[sel], y[sel], w=60, K=99)
        # Daily returns show volatility clustering, so the pairs are not exchangeable; block
        # permutation (Proposition P5) is the valid calibration here (Section 4.3; Supplementary Section B.17).
        for b in (20, 50):
            rb = single_break_test(x[sel], y[sel], w=60, K=99, block=b)
            for mth, rv in rb.items():
                r[f"{mth}|block{b}"] = rv
        for m in r:
            r[m]["tau_date"] = dates[sel[0] + r[m]["tau_hat"]]
        r["_acf1_absret"] = dict(
            x=float(np.corrcoef(np.abs(x[sel])[:-1], np.abs(x[sel])[1:])[0, 1]),
            y=float(np.corrcoef(np.abs(y[sel])[:-1], np.abs(y[sel])[1:])[0, 1]))
        res[f"e6b_{name}_{span[0][:4]}-{span[1][:4]}"] = r
        print(f"E6b {name} {span}: {time.time()-t0:.0f}s " +
              ", ".join(f"{m}: p={v['p_value']:.2f}" for m, v in r.items() if m != "_acf1_absret"), flush=True)
    SEG.flush_perm_log(os.path.join(OUT, "records_perm.csv"))
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)

    # figure: SPX-TNX weekly, rolling 26w Spearman + rolling DOMI + segmentation breaks
    dates, x, y = pairs["SPX-TNX"]
    xw, yw = weekly_sum(x), weekly_sum(y)
    wk_dates = dates[::5][: len(xw)]
    ctx = DOMIContext(xw[:, None], yw[:, None], 26, D=8, seed=2026, n_perm=1)
    from scipy.stats import spearmanr
    roll_r = [spearmanr(xw[t - 26:t], yw[t - 26:t])[0] for t in range(26, len(xw))]
    roll_q = [ctx.domi_bc(t - 26, t) for t in range(26, len(xw))]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    tt = np.arange(26, len(xw))
    axes[0].plot(tt, roll_r, color="#1f618d", lw=1.2)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("trailing 26-week Spearman ρ")
    axes[1].plot(tt, roll_q, color="#7d3c98", lw=1.2)
    axes[1].set_ylabel("trailing 26-week DOMI (bias-corr.)")
    for c in [i for i, d in enumerate(wk_dates) if d in res["e6a_SPX-TNX"]["cps_dates"]]:
        for ax in axes:
            ax.axvline(c, color="k", ls="--", lw=1)
    ticks = np.arange(0, len(xw), 52)
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([wk_dates[t][:4] for t in ticks], rotation=0, fontsize=8)
    axes[0].set_title("S&P500 return vs Δ(10Y yield), weekly 2011-2026; dashed = Holevo-partitioning breaks")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e6_figure.png"), dpi=140)
    print("E6 done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
