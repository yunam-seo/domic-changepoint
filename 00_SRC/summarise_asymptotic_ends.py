#!/usr/bin/env python
"""Share of the unstudentized scan's null maxima near the ends of the candidate grid (Supplementary Section B.16).

Reads the per-replicate records of run_asymptotic_threshold.py and counts, for each n, the null replicates whose
maximizer tau_hat_raw lies within 0.015 n of either end of the grid [0.1 n, 0.9 n].

Run:    python 00_SRC/summarise_asymptotic_ends.py
Writes 04_DAOU/EXPERIMENT/asymptotic_threshold/end_share.json
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "asymptotic_threshold")


def main():
    by = {}
    for r in csv.DictReader(open(os.path.join(D, "level_records.csv"))):
        if r["kind"] != "H0":
            continue
        t = int(r["tau_hat_raw"]) / int(r["n"])
        by.setdefault(r["n"], []).append(t <= 0.115 or t >= 0.885)
    out = {n: dict(replicates=len(v), near_ends=sum(v), share=sum(v) / len(v)) for n, v in sorted(by.items(), key=lambda x: int(x[0]))}
    allv = [x for v in by.values() for x in v]
    out["pooled"] = dict(replicates=len(allv), near_ends=sum(allv), share=sum(allv) / len(allv))
    json.dump(out, open(os.path.join(D, "end_share.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
