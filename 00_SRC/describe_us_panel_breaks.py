#!/usr/bin/env python
"""Descriptive statistics either side of each break of the US panel (not used for any decision).

For each pair, on the stage-two window of us_panel/stage2.json split at the break date: Spearman correlation,
bias-corrected DOMI (D = 8, seeds 2026/2027), Spearman correlation of absolute returns, and annualized
standard deviations of both series.

Also, per pair: the whole record's Spearman correlation, the two sides of the whole record split at the
break date, and calendar-year Spearman and bias-corrected DOMI (years with more than 50 days).

Run:    python 00_SRC/describe_us_panel_breaks.py            (break windows)
        python 00_SRC/describe_us_panel_breaks.py yearly     (whole-record split and calendar years)
Writes 04_DAOU/EXPERIMENT/us_panel/describe_breaks.json, describe_yearly.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import run_us_finance_panel as U  # noqa: E402
from dots.domi import DOMIContext  # noqa: E402


def stats(x, y):
    ctx = DOMIContext(x[:, None], y[:, None], 10, D=8, seed=2026, n_perm=1)
    return dict(n=len(x), spearman=float(spearmanr(x, y)[0]), domi_bc=float(ctx.domi_bc(0, len(x))),
                spearman_abs=float(spearmanr(np.abs(x), np.abs(y))[0]),
                sd_x_annual=float(np.std(x) * np.sqrt(252)), sd_y_annual=float(np.std(y) * np.sqrt(252)))


def yearly():
    s2 = json.load(open(os.path.join(U.OUT, "stage2.json")))["results"]
    out = {}
    for name, r in s2.items():
        dates, x, y = U.pair_data(name)
        c = dates.index(r["date"])
        yrs = {}
        for yr in sorted({d[:4] for d in dates}):
            m = np.array([d[:4] == yr for d in dates])
            if m.sum() > 50:
                yrs[yr] = dict(n=int(m.sum()), spearman=float(spearmanr(x[m], y[m])[0]),
                               domi_bc=float(DOMIContext(x[m][:, None], y[m][:, None], 10, D=8, seed=2026, n_perm=1).domi_bc(0, int(m.sum()))))
        out[name] = dict(whole_record_spearman=float(spearmanr(x, y)[0]), break_date=r["date"],
                         split_left=stats(x[:c], y[:c]), split_right=stats(x[c:], y[c:]), years=yrs)
        print(name, json.dumps(out[name]["years"]))
    json.dump(out, open(os.path.join(U.OUT, "describe_yearly.json"), "w"), indent=1)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "yearly":
        return yearly()
    s2 = json.load(open(os.path.join(U.OUT, "stage2.json")))["results"]
    out = {}
    for name, r in s2.items():
        dates, x, y = U.pair_data(name)
        a, b = r["window"].split("-")
        i0, i1, c = dates.index(a), dates.index(b) + 1, dates.index(r["date"])
        out[name] = dict(date=r["date"], window=r["window"], before=stats(x[i0:c], y[i0:c]), after=stats(x[c:i1], y[c:i1]))
        print(name, json.dumps(out[name]))
    json.dump(out, open(os.path.join(U.OUT, "describe_breaks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
