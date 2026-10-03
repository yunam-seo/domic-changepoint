#!/usr/bin/env python
"""Block-exchangeability diagnostics for the financial application (Section 6.8).

Section 6.8 calibrates the weekly segmentation by pair permutation only where the exchangeability
diagnostic of Section 4.3 does not reject, and by block permutation otherwise; the daily 2021-2022
example is re-calibrated at b = 20 and b = 50. This adds the direct block-level checks that the
weather application gets in `run_block_exch_diag.py`, matched to what is actually permuted:

  weekly, whole record -- the unit permuted by the weekly segmentation. Volatility clustering does
      not show in the returns themselves but in their magnitude, so the ACF and Ljung-Box are run on
      r_t, |r_t| and r_t^2 separately. A rejection on |r| or r^2 with none on r is the signature.

  daily blocks inside the 2021-2022 window -- the unit permuted by the daily example. That window
      is about two calendar years of trading days, so b = 20 leaves roughly 25 blocks and b = 50
      roughly 10, and the chi-square approximation behind Ljung-Box is not to be trusted on that
      many. It does not have to be: the hypothesis under test IS exchangeability of the blocks, so
      permuting the block order and recomputing the statistic gives an EXACT p-value under that
      null, with no asymptotics. Both are reported -- the chi-square p-value for comparability with
      the weekly table, and the permutation p-value as the one to read.

Everything is loaded through `run_e6_finance`'s own functions, so the series are the ones the paper
analyses rather than a re-derivation of them.

    python 00_SRC/run_block_exch_diag_fin.py
Writes 04_DAOU/EXPERIMENT/block_exch_diag/{finance.csv,finance.json,run_fin.log}.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "00_SRC"))
from dots.perm import ge  # noqa: E402

from run_e6_finance import load, align_pair, weekly_sum  # noqa: E402
from run_block_exch_diag import ljung_box  # noqa: E402


def lb_perm_p(v, L, K=9999, seed=20260831):
    """Exact-under-block-exchangeability p-value for the Ljung-Box statistic on a block summary.

    The null being tested is that the block-level sequence is exchangeable, and the permutation
    distribution of any statistic is exact under it. With ten or twenty-five blocks the chi-square
    approximation is not usable, so the permutation p-value is the one to read.
    """
    rng = np.random.default_rng(seed)
    obs = ljung_box(v, L)[0]
    n_ge = sum(ge(ljung_box(v[rng.permutation(len(v))], L)[0], obs) for _ in range(K))
    return float((1 + n_ge) / (K + 1))

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "block_exch_diag")
WINDOW = ("20210101", "20221231")          # the daily example of Section 6.8


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "run_fin.log"), "w")

    def say(m):
        print(m, flush=True)
        log.write(m + "\n")

    spx, tnx = load("yahoo_gspc.csv"), load("yahoo_tnx.csv")
    ksp, fx = load("yahoo_kospi.csv"), load("yahoo_usdkrw.csv")
    pairs = {"SPX-TNX": align_pair(spx, tnx, ("logret", "diff")),
             "KOSPI-USDKRW": align_pair(ksp, fx, ("logret", "logret"))}

    rows = []
    say("=== weekly full record -- what breaks exchangeability ===")
    say(f"  {'pair / series':22s} {'n wk':>5s} {'LB p: r':>9s} {'LB p: |r|':>10s} {'LB p: r²':>9s}"
        f" {'ACF1 |r|':>9s}")
    for name, (dates, x, y) in pairs.items():
        for lab, s in ((name.split("-")[0], x), (name.split("-")[1], y)):
            w = weekly_sum(s)
            out = {}
            for k, v in (("r", w), ("absr", np.abs(w)), ("r2", w ** 2)):
                _, p, r = ljung_box(v, 10)
                out[k] = (p, r[0])
            say(f"  {name+' / '+lab:22s} {len(w):5d} {out['r'][0]:9.3f} {out['absr'][0]:10.3f}"
                f" {out['r2'][0]:9.3f} {out['absr'][1]:9.3f}")
            rows.append(dict(scope="weekly_full", pair=name, series=lab, n=len(w),
                             lb_p_r=round(out["r"][0], 4), lb_p_absr=round(out["absr"][0], 4),
                             lb_p_r2=round(out["r2"][0], 4), acf1_absr=round(out["absr"][1], 4)))

    say("\n=== daily 2021-2022 window, block summaries (SPX-TNX) ===")
    dates, x, y = pairs["SPX-TNX"]
    sel = [i for i, d in enumerate(dates) if WINDOW[0] <= d <= WINDOW[1]]
    say(f"  window length {len(sel)} trading days")
    for b in (20, 50):
        nb = len(sel) // b
        say(f"  b={b}: {nb} blocks")
        for lab, s in (("SPX", np.asarray(x)[sel]), ("TNX", np.asarray(y)[sel])):
            seg = s[: nb * b].reshape(nb, b)
            bm, bv = seg.mean(1), seg.var(1)
            L = max(1, min(3, nb // 4))
            pm, pv = ljung_box(bm, L)[1], ljung_box(np.log(bv), L)[1]
            qm, qv = lb_perm_p(bm, L), lb_perm_p(np.log(bv), L)
            say(f"    {lab:5s} block means  chi2 p {pm:.3f} / permutation p {qm:.4f}"
                f"   log block variances  chi2 p {pv:.3f} / permutation p {qv:.4f}   (L={L})")
            rows.append(dict(scope=f"daily_b{b}", pair="SPX-TNX", series=lab, n=nb,
                             lb_p_block_mean=round(pm, 4), lb_p_log_block_var=round(pv, 4),
                             perm_p_block_mean=round(qm, 4), perm_p_log_block_var=round(qv, 4),
                             lb_lags=L))
    say("\n  With only 25 (b=20) and 10 (b=50) blocks the chi2 approximation is not usable; the")
    say("  permutation p-values are exact under the block exchangeability being tested.")
    say("  Table B.19 reports the window's p-values over 20 permutation draws at b = 20 and 50"
        " (run_combined_block_seeds.py).")

    with open(os.path.join(OUT, "finance.csv"), "w", newline="") as f:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(rows)
    json.dump({"window": WINDOW, "rows": rows}, open(os.path.join(OUT, "finance.json"), "w"), indent=1)
    say(f"\nwritten: {OUT}")
    log.close()


if __name__ == "__main__":
    main()
