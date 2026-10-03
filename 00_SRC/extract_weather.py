#!/usr/bin/env python
"""Daily station aggregates from the KMA API Hub ASOS daily files, for run_design_checks.py (Supplementary Section B.5).

Input:  $ASOS_ARCHIVE/<YYYY>/<MM>/asos_YYYYMMDD.csv, columns tm,stn,ta,wd,ws,td,hm,pa,ps,rn
        (hourly, all stations); the archive is only read.
Output: 02_MART/WEATHER_DAILY.csv, per (date, stn): daily mean of ta, hm, ws, pa, td and the count.
Stations: 108 (Seoul), 184 (Jeju).

Run:  ASOS_ARCHIVE=<archive directory> python 00_SRC/extract_weather.py
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

SRC_DIR = os.environ.get("ASOS_ARCHIVE", "")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "02_MART", "WEATHER_DAILY.csv")
STATIONS = {"108", "184"}
VARS = ["ta", "hm", "ws", "pa", "td"]


def one_file(path):
    acc = {}
    with open(path) as f:
        rd = csv.DictReader(f)
        for row in rd:
            stn = row["stn"]
            if stn not in STATIONS:
                continue
            date = row["tm"][:8]
            key = (date, stn)
            if key not in acc:
                acc[key] = {v: [] for v in VARS}
            for v in VARS:
                x = row.get(v, "NA")
                if x not in ("NA", "", None):
                    try:
                        xv = float(x)
                        if xv > -90:  # KMA missing codes are large negatives
                            acc[key][v].append(xv)
                    except ValueError:
                        pass
    rows = []
    for (date, stn), d in acc.items():
        r = dict(date=date, stn=stn)
        for v in VARS:
            r[v] = round(float(np.mean(d[v])), 3) if d[v] else ""
            r[f"n_{v}"] = len(d[v])
        rows.append(r)
    return rows


def main():
    if not SRC_DIR:
        sys.exit("set ASOS_ARCHIVE to the directory of the KMA API Hub ASOS daily files (<YYYY>/<MM>/asos_YYYYMMDD.csv)")
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*", "*", "asos_*.csv")))
    print(f"{len(files)} daily files", flush=True)
    with Pool(16) as pool:
        all_rows = []
        for i, rows in enumerate(pool.imap_unordered(one_file, files, chunksize=8)):
            all_rows += rows
            if i % 200 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    all_rows.sort(key=lambda r: (r["stn"], r["date"]))
    keys = ["date", "stn"] + VARS + [f"n_{v}" for v in VARS]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
