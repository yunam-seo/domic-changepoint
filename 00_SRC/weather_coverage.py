#!/usr/bin/env python
"""Coverage of the hourly weather record (Supplementary Section B.8).

For each of the twelve stations: hours present out of the 70,128 hours of 2018-2025, and for each of the
three variable pairs the number of hours with both values present, which is what every weather analysis
uses (hours missing from the record or lacking either value are dropped).

Run:  python 00_SRC/weather_coverage.py
Reads 02_MART/WEATHER_HOURLY_ANOM.npz; writes 04_DAOU/EXPERIMENT/e8/data_coverage.json.
"""
import json
import os

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(SRC)
ANOM = os.path.join(ROOT, "02_MART", "WEATHER_HOURLY_ANOM.npz")
OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8", "data_coverage.json")
HOURS = 70128                       # 2,922 days x 24 hours, 2018-01-01 00h to 2025-12-31 23h
PAIRS = (("ta", "hm"), ("ta", "ws"), ("hm", "ws"))


def main():
    z = np.load(ANOM)
    stations = sorted({k.split("|")[0] for k in z.files})
    per = {}
    for s in stations:
        rec = dict(hours_present=int(len(z[f"{s}|tm"])), first=str(z[f"{s}|tm"][0]), last=str(z[f"{s}|tm"][-1]))
        for u, v in PAIRS:
            ok = np.isfinite(z[f"{s}|{u}"]) & np.isfinite(z[f"{s}|{v}"])
            rec[f"{u}-{v}"] = dict(hours_used=int(ok.sum()), hours_dropped=HOURS - int(ok.sum()))
        per[s] = rec
    dropped = [(r[f"{u}-{v}"]["hours_dropped"], s, f"{u}-{v}") for s, r in per.items() for u, v in PAIRS]
    worst = max(dropped)
    out = dict(hours_in_period=HOURS, stations=per,
               min_hours_present=min(r["hours_present"] for r in per.values()),
               max_hours_dropped_per_pair=worst[0], max_share_dropped_per_pair=worst[0] / HOURS,
               worst_pair=dict(station=worst[1], pair=worst[2]))
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"max hours dropped per pair {worst[0]} ({100 * worst[0] / HOURS:.2f}%) at {worst[1]} {worst[2]}")


if __name__ == "__main__":
    main()
