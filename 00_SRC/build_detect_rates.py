#!/usr/bin/env python
"""Unlocalized detection rates next to localized power for every cell of Table 1 (no new runs).

Purpose
-------
Table 1 reports localized power: a replicate counts only if the maximum exceeds the threshold AND
|tau_hat - tau| <= 30. That mixes detection with localization. This script reads the stored
aggregates and writes, for each Table 1 cell and each of its six statistics, the detection rate
(maximum above threshold, wherever it falls) beside the localized power, so that the two can be
separated (Supplementary Table B.10).

Sources and variants (as in Table 1, except the copula column)
---------------------------------------------------------------
The CvM column here is the global-rank copula statistic (CopulaCvM); Table 1 and Supplementary
Table B.10 take the copula column from cvm_subsample (build_tables.build_detect).
Upper block (S1-S4, M1): the variant per method is the one of rebuild_table1.MODE -- DOMI, HSIC and
dCor studentized ("gs"), Spearman and the copula CvM raw ("g") -- with S1, S2 and M1 from
e1/results.csv (level-0 null) and S3, S4 from e1_matched_null/results.csv (matched null); the Gram
form is studentized, from matmi_baseline_full_T1a (S1, S2, M1) and matmi_baseline_full_T1b (S3, S4),
as in build_tables.build_t1_combined.
Lower block (G1-G6): ng_suite_A/B with the Gram form overlaid from ng_suite_gfull_G1..G6, each
method at the variant build_tables.pick_mode selects (larger mean power over the block); on the
stored aggregates this is the studentized variant for DOMI and the Gram form, the one Algorithm 1 fixes.

Run:   python 00_SRC/build_detect_rates.py
Outputs: 04_DAOU/EXPERIMENT/detect_rates/
    table1_detect.csv   block, setting, method, variant, power, detect_rate, binomial SEs, source
    table1_detect.tex   LaTeX rows "power / detection rate" per method, Table 1 row order
"""
from __future__ import annotations

import csv
import math
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import build_tables as B  # noqa: E402
from rebuild_table1 import MODE as MODE_UPPER  # noqa: E402

EXP = B.EXP
OUT = os.path.join(EXP, "detect_rates")
N = 500
METHODS = [("DOMI", "DOMI-diff"), ("Gram", "matmi-a1"), ("HSIC", "HSIC-diff"), ("dCor", "dCor-diff"),
           ("Spearman", "Spearman-diff"), ("CvM", "CopulaCvM")]
UPPER = [("S1, r=0.35", "D1", 0.35), ("S2, a=0.7", "D2", 0.7),
         ("S2, a=0.9", "D2", 0.9), ("S3, tau=0.5", "D3", 0.5),
         ("S4, r=0.7", "D4", 0.7), ("S4, r=0.85", "D4", 0.85),
         ("M1, s=2 (FA)", "M1", 2.0)]


def raw(sub):
    d = {}
    p = os.path.join(EXP, sub, "results.csv")
    for r in csv.DictReader(open(p)):
        if r["method"] == "method":
            continue
        d[(r["scen"], round(float(r["level"]), 4), r["method"], r["mode"])] = dict(r, _file=sub)
    return d


def se(p):
    return round(math.sqrt(p * (1 - p) / N), 4)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    # ------------------------------------------------------------------ upper block
    e1, mat = raw("e1"), raw("e1_matched_null")
    ga = B.overlay_gram(B.load("matmi_baseline"), ["matmi_baseline_full_T1a"])
    gb = B.overlay_gram(B.load("matmi_baseline_matched_D8"), ["matmi_baseline_full_T1b"])
    for lab, s, l in UPPER:
        matched = s in ("D3", "D4")
        for name, key in METHODS:
            if key == "matmi-a1":
                r = (gb if matched else ga)[(s, round(l, 4), key, "gs")]
                mode, src = "gs", r["_file"]
            else:
                mode = MODE_UPPER[key]
                r = (mat if matched else e1)[(s, round(l, 4), key, mode)]
                src = "04_DAOU/EXPERIMENT/" + r["_file"] + "/results.csv"
            pw, dt = float(r["power"]), float(r["detect_rate"])
            rows.append(dict(block="upper", setting=lab, scen=s, level=l, method=name, variant=mode,
                             power=pw, power_se=se(pw), detect_rate=dt, detect_se=se(dt),
                             detect_minus_power=round(dt - pw, 4), source=src))
    # ------------------------------------------------------------------ lower block
    ng = B.load("ng_suite_A")
    ng.update(B.load("ng_suite_B"))
    ng = B.overlay_gram(ng, [f"ng_suite_gfull_G{i}" for i in range(1, 7)])
    LV = {"G1": [8.0, 4.0, 2.0], "G2": [0.2, 0.35, 0.5], "G3": [0.2, 0.35, 0.5],
          "G4": [0.5, 0.7, 0.9], "G5": [0.5, 0.7, 0.9], "G6": [0.3, 0.5, 0.8]}
    PAR = {"G1": "nu", "G2": "tau", "G3": "tau", "G4": "a", "G5": "a", "G6": "b"}
    gcols = [(s, l, "power") for s in LV for l in LV[s]]
    lower_key = {"DOMI-diff": "DOMI"}
    modes = {key: B.pick_mode(ng, lower_key.get(key, key), gcols) for _, key in METHODS}
    modes["DOMI-diff"] = modes["matmi-a1"] = "gs"  # fixed in advance by Algorithm 1 (Section 5)
    for s in LV:
        for l in LV[s]:
            for name, key in METHODS:
                k = lower_key.get(key, key)
                r = ng[(s, round(l, 4), k, modes[key])]
                pw, dt = float(r["power"]), float(r["detect_rate"])
                rows.append(dict(block="lower", setting=f"{s}, {PAR[s]}={l:g}", scen=s, level=l,
                                 method=name, variant=modes[key], power=pw, power_se=se(pw),
                                 detect_rate=dt, detect_se=se(dt), detect_minus_power=round(dt - pw, 4),
                                 source=r["_file"]))
    with open(os.path.join(OUT, "table1_detect.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # ------------------------------------------------------------------ LaTeX fragment
    lines = [r"% power / unlocalised detection rate; M1 row: both entries are false-alarm rates",
             r"% columns: " + " & ".join(n for n, _ in METHODS)]
    for blk in ("upper", "lower"):
        settings = []
        for r in rows:
            if r["block"] == blk and r["setting"] not in settings:
                settings.append(r["setting"])
        for st in settings:
            cells = [f"{r['power']:.2f}/{r['detect_rate']:.2f}" for r in rows
                     if r["block"] == blk and r["setting"] == st]
            lines.append(st + " & " + " & ".join(cells) + r" \\")
        if blk == "upper":
            lines.append(r"\midrule")
    open(os.path.join(OUT, "table1_detect.tex"), "w").write("\n".join(lines) + "\n")
    print(f"{len(rows)} cells written; variants lower block: {modes}")


if __name__ == "__main__":
    main()
