#!/usr/bin/env python
"""Build hourly anomaly series from 02_MART/WEATHER_HOURLY.csv and report the effective sample size.

Hourly observations carry two deterministic cycles, diurnal and annual, and the diurnal amplitude
itself varies with season. The climatology is therefore indexed by (hour, day-of-year) and smoothed
over a +/- 7 day window within each hour, so that a real coupling change is not absorbed into it.

Writes 02_MART/WEATHER_HOURLY_ANOM.npz and prints acf1 / n_eff per station and variable.
"""
from __future__ import annotations
import csv, os
from datetime import date
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "02_MART", "WEATHER_HOURLY.csv")
OUT = os.path.join(ROOT, "02_MART", "WEATHER_HOURLY_ANOM.npz")
COLS = ["ta", "hm", "ws", "pa", "td"]
HALF = 7   # +/- days of smoothing within each hour


def anomaly(vals, doy, hour):
    """Subtract a (hour, day-of-year) climatology smoothed over +/- HALF days."""
    a = vals.astype(float).copy()
    out = np.full_like(a, np.nan)
    for h in range(24):
        mh = hour == h
        if mh.sum() == 0:
            continue
        vh, dh = a[mh], doy[mh]
        clim = np.zeros(367)
        for d in range(1, 367):
            off = np.abs(((dh - d + 183) % 366) - 183)      # circular day distance
            sel = off <= HALF
            clim[d] = np.nanmean(vh[sel]) if np.isfinite(vh[sel]).any() else np.nan
        out[mh] = vh - clim[dh]
    return out


if __name__ == "__main__":
    rows = list(csv.DictReader(open(SRC)))
    stns = sorted({r["stn"] for r in rows})
    data = {}
    print(f"{'stn':>5} {'var':>4} {'n':>8} {'missing':>8} {'acf1':>7} {'n_eff':>9}")
    for s in stns:
        rs = [r for r in rows if r["stn"] == s]
        tm = np.array([r["tm"] for r in rs])
        doy = np.array([date(int(t[:4]), int(t[4:6]), int(t[6:8])).timetuple().tm_yday for t in tm])
        hour = np.array([int(t[8:10]) for t in tm])
        d = {"tm": tm}
        for c in COLS:
            v = np.array([float(r[c]) if r[c] != "" else np.nan for r in rs])
            an = anomaly(v, doy, hour)
            d[c] = an
            ok = np.isfinite(an)
            if ok.sum() > 100:
                z = an[ok]
                p = float(np.corrcoef(z[:-1], z[1:])[0, 1])
                neff = len(z) * (1 - p) / (1 + p)
                print(f"{s:>5} {c:>4} {len(v):>8} {int((~ok).sum()):>8} {p:>7.3f} {neff:>9.0f}")
        data[s] = d
    np.savez_compressed(OUT, **{f"{s}|{k}": v for s, d in data.items() for k, v in d.items()})
    print(f"\nwrote {OUT}")
