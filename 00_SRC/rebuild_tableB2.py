#!/usr/bin/env python
"""Regenerate Table B.2 from the bandwidth-sweep aggregates: level-0 null and matched null side by side.

The companion of `rebuild_table1.py`: Table B.2 is assembled from the level-0 null sweeps under an
explicit rule, with the D3 and D4 columns taken from the matched-null runs under the same rule.

Rule. DOMI is scored studentized only, as `run_domi_bandwidth.py` computes it. The full-Gram HSIC is
scored in both forms; each power cell reports the larger of the two powers, and the M1 cell the
larger of the two detection (false-alarm) rates (the `best` flag of `run_hsic_fullgram.py` is not
read). Only D3 and D4 have matched-null runs; D1, D2 and M1 have a null that does
not depend on `level`, so their columns are unchanged.

Run:  python 00_SRC/rebuild_tableB2.py
Writes 04_DAOU/EXPERIMENT/bandwidth_matched_null/tableB2.csv.
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "04_DAOU", "EXPERIMENT")

ROWS = [("DOMI, ×1 (deployed)", "DOMI", 1.0), ("DOMI, ×1/2", "DOMI", 0.5), ("DOMI, ×1/4", "DOMI", 0.25),
        ("HSIC, full Gram, ×1", "HSIC-fullgram", 1.0),
        ("HSIC, full Gram, ×1/4", "HSIC-fullgram", 0.25)]
COLS = [("D1", 0.35, "S1 r=.35", "power"), ("D1", 0.5, "S1 r=.5", "power"),
        ("D2", 0.7, "S2 a=.7", "power"), ("D2", 0.9, "S2 a=.9", "power"),
        ("D3", 0.5, "S3 τ=.5", "power"), ("D4", 0.7, "S4 r=.7", "power"),
        ("D4", 0.85, "S4 r=.85", "power"), ("M1", 2.0, "M1", "detect_rate")]
AFFECTED = {("D3", 0.5), ("D4", 0.7), ("D4", 0.85)}


def load_level0():
    d = defaultdict(lambda: defaultdict(float))
    for f in glob.glob(os.path.join(EXP, "domi_bandwidth", "results_*.csv")):
        for r in csv.DictReader(open(f)):
            k = ("DOMI", r["scen"], float(r["level"]), float(r["sigma_mult"]))
            for rate in ("power", "detect_rate"):
                d[k][rate] = max(d[k][rate], float(r[rate]))
    for r in csv.DictReader(open(os.path.join(EXP, "hsic_fullgram", "results.csv"))):
        k = ("HSIC-fullgram", r["scen"], float(r["level"]), float(r["mult"]))
        for rate in ("power", "detect_rate"):
            d[k][rate] = max(d[k][rate], float(r[rate]))
    return d


def load_matched():
    d = defaultdict(lambda: defaultdict(float))
    for f in glob.glob(os.path.join(EXP, "bandwidth_matched_null", "results_*.csv")):
        for r in csv.DictReader(open(f)):
            k = (r["method"], r["scen"], float(r["level"]), float(r["bandwidth_key"]))
            for rate in ("power", "detect_rate"):
                d[k][rate] = max(d[k][rate], float(r[rate]))
    return d


def main():
    pub, mat = load_level0(), load_matched()
    print("== 1. Table B.2 (S3/S4 columns under the matched null) ==")
    print(f"  {'':22s} " + " ".join(f"{h:>9s}" for _, _, h, _ in COLS))
    out_rows = []
    for label, stat, mlt in ROWS:
        vals = [(mat if (s, l) in AFFECTED else pub)[(stat, s, l, mlt)][rate]
                for s, l, _, rate in COLS]
        print(f"  {label:22s} " + " ".join(
            f"{('*' if (s, l) in AFFECTED else '') + f'{v:.2f}':>9s}"
            for v, (s, l, _, _) in zip(vals, COLS)))
        out_rows.append(dict(row=label, **{h: round(v, 3) for (_, _, h, _), v in zip(COLS, vals)}))
    print("  (* = recomputed under the matched null)")

    print("\n== 2. Best over bandwidths, DOMI against full-Gram HSIC ==")
    ALL = [4.0, 2.0, 1.0, 0.5, 0.25]
    for s, l, h, rate in COLS[:-1]:
        src = mat if (s, l) in AFFECTED else pub
        q = max((src[("DOMI", s, l, m)][rate], m) for m in ALL)
        x = max((src[("HSIC-fullgram", s, l, m)][rate], m) for m in ALL)
        print(f"  {h:10s} DOMI {q[0]:.3f} @×{q[1]:<5g} HSIC {x[0]:.3f} @×{x[1]:<5g} "
              f"{100*(q[0]-x[0]):+6.1f} pt" + ("" if (s, l) in AFFECTED else "   (unaffected)"))

    p = os.path.join(EXP, "bandwidth_matched_null", "tableB2.csv")
    with open(p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        wr.writeheader(); wr.writerows(out_rows)
    print("\nwritten:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
