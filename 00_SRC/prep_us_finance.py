#!/usr/bin/env python
"""Convert the stored Yahoo chart responses of the US panel to date,close files (US panel of Section 6.8).

Reads 01_ORG/FINANCE/raw_{GLD,DXY}.json, keeps the days with a non-null close, dates in the
exchange time zone (timestamp + gmtoffset) as YYYYMMDD, and writes yahoo_{gld,dxy}.csv beside them, in the
format of 01_ORG/FINANCE/yahoo_*.csv. Prints a completeness report against the S&P 500 trading days.

Run:    python 00_SRC/prep_us_finance.py
"""
import csv
import datetime as dt
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "01_ORG", "FINANCE")
SPX = os.path.join(ROOT, "01_ORG", "FINANCE", "yahoo_gspc.csv")


def convert(code):
    r = json.load(open(os.path.join(RAW, f"raw_{code}.json"), encoding="utf-8-sig"))["chart"]["result"][0]
    off = r["meta"].get("gmtoffset", 0)
    rows = [(dt.datetime.fromtimestamp(t + off, dt.timezone.utc).strftime("%Y%m%d"), c)
            for t, c in zip(r["timestamp"], r["indicators"]["quote"][0]["close"]) if c is not None and c > 0]
    assert len({d for d, _ in rows}) == len(rows), "duplicate dates"
    with open(os.path.join(RAW, f"yahoo_{code.lower()}.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["date", "close"])
        w.writerows((d, repr(float(c))) for d, c in rows)
    return [d for d, _ in rows]


def main():
    spx = [r["date"] for r in csv.DictReader(open(SPX))]
    for code in ("GLD", "DXY"):
        days = convert(code)
        s, o = set(spx), set(days)
        print(f"{code}: {len(days)} days {days[0]}..{days[-1]}; S&P days without {code}: {len(s - o)}; "
              f"{code} days without S&P: {len(o - s)}; common: {len(s & o)}")
        miss = sorted(s - o)
        if miss:
            print("   first missing:", miss[:12])


if __name__ == "__main__":
    main()
