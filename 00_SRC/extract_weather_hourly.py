#!/usr/bin/env python
"""Extract hourly ASOS observations for twelve major stations into 02_MART/WEATHER_HOURLY.csv.

Input:  $ASOS_ARCHIVE/<YYYY>/<MM>/asos_YYYYMMDD.csv (daily files of the KMA API Hub ASOS service, only read).
Run:    ASOS_ARCHIVE=<archive directory> python 00_SRC/extract_weather_hourly.py
Next:   python 00_SRC/prep_weather_hourly.py builds the anomaly series.
"""
from __future__ import annotations
import csv, glob, os, sys, time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.environ.get("ASOS_ARCHIVE", "")
OUT = os.path.join(ROOT, "02_MART", "WEATHER_HOURLY.csv")

# twelve major ASOS stations, geographically spread
STATIONS = ["105", "108", "112", "119", "129", "133", "143", "146", "156", "159", "165", "184"]
NAMES = {"105": "Gangneung", "108": "Seoul", "112": "Incheon", "119": "Suwon", "129": "Seosan",
         "133": "Daejeon", "143": "Daegu", "146": "Jeonju", "156": "Gwangju", "159": "Busan",
         "165": "Mokpo", "184": "Jeju"}
COLS = ["ta", "hm", "ws", "pa", "td"]


def _one(path):
    rows = []
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("stn") not in STATIONS:
                    continue
                out = {"tm": r["tm"], "stn": r["stn"]}
                for c in COLS:
                    v = r.get(c, "")
                    out[c] = "" if v in ("NA", "", "None", None) else v
                rows.append(out)
    except Exception as e:                       # report and skip an unreadable daily file
        print(f"unreadable {path}: {e}", file=sys.stderr, flush=True)
        return []
    return rows


if __name__ == "__main__":
    if not ARCH:
        sys.exit("set ASOS_ARCHIVE to the directory of the KMA API Hub ASOS daily files (<YYYY>/<MM>/asos_YYYYMMDD.csv)")
    files = sorted(glob.glob(os.path.join(ARCH, "*", "*", "asos_*.csv")))
    print(f"archive files: {len(files)}", flush=True)
    t0 = time.time()
    with Pool(12) as pool:
        chunks = pool.map(_one, files, chunksize=32)
    rows = [r for c in chunks for r in c]
    rows.sort(key=lambda r: (r["stn"], r["tm"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tm", "stn"] + COLS)
        w.writeheader(); w.writerows(rows)
    per = {}
    for r in rows:
        per[r["stn"]] = per.get(r["stn"], 0) + 1
    print(f"wrote {len(rows)} rows to {OUT} in {time.time()-t0:.0f}s", flush=True)
    for s in STATIONS:
        print(f"  {s} {NAMES[s]:<10} {per.get(s,0):>7} hours", flush=True)
    print(f"  date range: {min(r['tm'] for r in rows)} .. {max(r['tm'] for r in rows)}", flush=True)
