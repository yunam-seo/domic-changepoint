#!/usr/bin/env python
"""Assemble the single-break aggregates behind Table 1 (DOMI, HSIC, dCor and Spearman columns) and
Supplementary Table B.6: S1, S2 and M1 under the level-0 null, S3 and S4 under the matched null.

Table 1 is assembled under an explicit selection rule: the S1, S2 and M1 cells from
`e1/results.csv` (level-0 null, which for these scenarios does not depend on the level) and the
S3/S4 cells from `e1_matched_null/results.csv` (matched null).

Selection rule (Section 5): each method is reported at whichever of its two forms -- the raw
weighted global difference or the per-t studentized one -- is the more powerful over the block.
The choice is therefore one mode per method ("g" or "gs"), held across every column, not a per-cell
maximum.

Each method keeps the same mode under both nulls, so that only the null changes.

Run:  python 00_SRC/rebuild_table1.py
Writes 04_DAOU/EXPERIMENT/e1_matched_null/table1.csv.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "04_DAOU", "EXPERIMENT")

# One row per statistic (Table 1 prints one row per setting; Joint-state Holevo, MMD, Gaussian LR and
# the global-rank copula row are Table B.6; Table 1's Gram and CvM columns come from
# matmi_baseline_full_* and cvm_subsample).
ROWS = [("DOMI (studentised)", "DOMI-diff"),
        ("Joint-state Holevo", "Holevo-joint"),
        ("HSIC difference", "HSIC-diff"),
        ("Distance correlation", "dCor-diff"),
        ("Spearman difference", "Spearman-diff"),
        ("Empirical-copula CvM", "CopulaCvM"),
        ("MMD (median bandwidth)", "MMDx1.0"),
        ("Gaussian likelihood ratio", "GaussLR")]
# Table 1's columns: (scenario, level, printed heading, which rate).
COLS = [("D1", 0.35, "S1 r=.35", "power"), ("D2", 0.7, "S2 a=.7", "power"),
        ("D2", 0.9, "S2 a=.9", "power"), ("D3", 0.5, "S3 τ=.5", "power"),
        ("D4", 0.7, "S4 r=.7", "power"), ("D4", 0.85, "S4 r=.85", "power"),
        ("M1", 2.0, "M1 (FA)", "detect_rate")]
AFFECTED = {("D3", 0.5), ("D4", 0.7), ("D4", 0.85)}


# The mode each method is reported at (the selection rule above).
MODE = {"DOMI-diff": "gs", "Holevo-joint": "g", "HSIC-diff": "gs", "dCor-diff": "gs",
        "Spearman-diff": "g", "CopulaCvM": "g", "MMDx1.0": "g", "GaussLR": "g"}


def load(path):
    """(method, scen, level, mode) -> {rate: value}."""
    d = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r["method"] == "method" or r["mode"] not in ("g", "gs"):
            continue
        k = (r["method"], r["scen"], float(r["level"]), r["mode"])
        for rate in ("power", "detect_rate"):
            d[k][rate] = float(r[rate])
    return d


def at(d, key, s, l, rate, mode=None):
    return d[(key, s, l, mode or MODE[key])][rate]


def main():
    pub = load(os.path.join(EXP, "e1", "results.csv"))
    mat = load(os.path.join(EXP, "e1_matched_null", "results.csv"))
    print("== Table 1 (S3/S4 columns under the matched null) ==")
    hdr = [h for _, _, h, _ in COLS]
    print(f"  {'':26s} " + " ".join(f"{h:>9s}" for h in hdr))
    out_rows = []
    for label, key in ROWS:
        vals, flags = [], []
        for s, l, _, rate in COLS:
            src = mat if (s, l) in AFFECTED else pub
            v = at(src, key, s, l, rate)
            vals.append(v)
            flags.append((s, l) in AFFECTED)
        print(f"  {label:26s} " + " ".join(
            f"{('*' if f else '') + f'{v:.2f}':>9s}" for v, f in zip(vals, flags)))
        out_rows.append(dict(row=label, **{h: round(v, 3) for h, v in zip(hdr, vals)}))
    print("  (* = recomputed under the matched null)")


    p = os.path.join(EXP, "e1_matched_null", "table1.csv")
    with open(p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        wr.writeheader(); wr.writerows(out_rows)
    print("\nwritten:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
