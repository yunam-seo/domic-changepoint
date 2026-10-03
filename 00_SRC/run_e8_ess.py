#!/usr/bin/env python
"""Effective sample sizes of the hourly weather application (Section 6.7).

Effective sample size n (1 - r1) / (1 + r1), with r1 the lag-one autocorrelation of the hourly
anomaly series (02_MART/WEATHER_HOURLY_ANOM.npz, built by 00_SRC/prep_weather_hourly.py),
computed
  (1) over the full record, per station and variable (temperature ta, humidity hm, wind ws,
      pressure pa), after dropping missing hours -- the same computation that
      prep_weather_hourly.py prints; and
  (2) in the stage-two re-test windows: for each of the 27 stage-one candidates in
      04_DAOU/EXPERIMENT/e8/retest_K999.json, the window that 00_SRC/run_e8_retest.py
      tests -- the station's pair series restricted to hours where both variables are
      observed, cut into 336-hour units, and taken from 26 units before to 26 units after the
      candidate unit (about +/- 1 year). The effective sample size is reported for each of the
      two variables of the pair ("pair variables"), and additionally for all four variables
      in every candidate window ("all variables"), which is the only way pressure enters.

Run:    python 00_SRC/run_e8_ess.py
Writes: 04_DAOU/EXPERIMENT/e8/effective_sample_sizes.json
        (per-station and per-window values are included in the JSON)
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANOM = os.path.join(ROOT, "02_MART", "WEATHER_HOURLY_ANOM.npz")
E8 = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8")
OUT = os.path.join(E8, "effective_sample_sizes.json")
VARS = ("ta", "hm", "ws", "pa")
UNIT, HALF_UNITS = 336, 26


def ess(x):
    x = x[np.isfinite(x)]
    r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    return len(x) * (1 - r1) / (1 + r1), r1, len(x)


def summary(vals):
    v = np.array(vals)
    return {"min": int(round(v.min())), "max": int(round(v.max())),
            "median": int(round(float(np.median(v))))}


def main():
    z = np.load(ANOM)
    stns = sorted({k.split("|")[0] for k in z.files})

    full = {v: {} for v in VARS}
    for s in stns:
        for v in VARS:
            e, r1, n = ess(z[f"{s}|{v}"])
            full[v][s] = {"n": n, "acf1": r1, "n_eff": e}

    cands = json.load(open(os.path.join(E8, "retest_K999.json")))["results"]
    windows = []
    for c in cands:
        s, pair, k = c["stn"], c["pair"], c["interval"]
        u, w = pair.split("-")
        ok = np.isfinite(z[f"{s}|{u}"]) & np.isfinite(z[f"{s}|{w}"])
        n = int(ok.sum())
        edges = np.arange(0, n + 1, UNIT)
        if edges[-1] != n:
            edges = np.append(edges, n)
        lo, hi = max(0, k - HALF_UNITS), min(len(edges) - 1, k + HALF_UNITS)
        rec = {"stn": s, "pair": pair, "date": c["date"], "interval": k,
               "hours": int(edges[hi] - edges[lo])}
        for v in VARS:
            rec[v] = ess(z[f"{s}|{v}"][ok][edges[lo]:edges[hi]])[0]
        windows.append(rec)

    out = {"definition": "n_eff = n (1 - r1) / (1 + r1), r1 = lag-one autocorrelation",
           "full_record": {v: summary([d["n_eff"] for d in full[v].values()]) for v in VARS},
           "window_pair_variables": {
               v: summary([r[v] for r in windows if v in r["pair"].split("-")])
               for v in ("ta", "hm", "ws")},
           "window_all_variables": {v: summary([r[v] for r in windows]) for v in VARS},
           "per_station_full_record": full,
           "per_window": windows}
    os.makedirs(E8, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    for key in ("full_record", "window_pair_variables", "window_all_variables"):
        print(key, out[key])
    print("saved", OUT)


if __name__ == "__main__":
    main()
