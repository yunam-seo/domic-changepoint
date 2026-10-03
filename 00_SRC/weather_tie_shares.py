#!/usr/bin/env python
"""Share of tied values in the hourly weather anomaly series (Supplementary Section B.8).

For each station and variable (temperature ta, humidity hm, wind speed ws), the record is cut into
consecutive non-overlapping two-year windows (2 x 8766 hours); in each window the share of observations
whose value equals that of at least one other observation is computed, and the shares are averaged over
the windows. Supplementary Section B.8 quotes, per variable, the range of these averages across
the twelve stations.

Input:  02_MART/WEATHER_HOURLY_ANOM.npz (built by prep_weather_hourly.py from the KMA ASOS archive)
Output: 04_DAOU/EXPERIMENT/tie_check/weather_tie_shares.json
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "02_MART", "WEATHER_HOURLY_ANOM.npz")
OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "tie_check", "weather_tie_shares.json")
WIN = 2 * 8766


def share(x):
    _, c = np.unique(x, return_counts=True)
    return float(c[c > 1].sum() / len(x))


def main():
    d = np.load(SRC, allow_pickle=True)
    stations = sorted({k.split("|")[0] for k in d.files})
    res = {}
    for v in ("ta", "hm", "ws"):
        per = {}
        for s in stations:
            a = d[f"{s}|{v}"]
            a = a[np.isfinite(a)]
            per[s] = float(np.mean([share(a[i:i + WIN]) for i in range(0, len(a) - WIN + 1, WIN)]))
        res[v] = dict(per_station=per, min=min(per.values()), median=float(np.median(list(per.values()))),
                      max=max(per.values()))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(window_hours=WIN, results=res), open(OUT, "w"), indent=1)
    for v in res:
        print(v, "min %.3f median %.3f max %.3f" % (res[v]["min"], res[v]["median"], res[v]["max"]))


if __name__ == "__main__":
    main()
