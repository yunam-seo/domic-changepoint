#!/usr/bin/env python
"""Location recovery on scenario MB from the stored per-replicate segmentations (no new runs).

P(K_hat = 3) counts replicates with the correct number of breaks. This script adds the rate at which
all three breaks (450, 900, 1350) are recovered within a tolerance delta, matching the sorted estimates
one to one, and the recall and precision of the estimated breaks at that tolerance.

Inputs (per-replicate change-point lists already on disk):
  e2/records_alt.csv.gz          Holevo partitioning, rank-Gaussian PELT, BinSeg-MMD (at the e2 penalties)
  seg_baselines/records_reps.csv.gz   KCP, e.divisive and the re-tested variants
  bmctc/records_mb.csv.gz        BMCTC, calibrated split rule
Run:     python 00_SRC/mb_location_recovery.py
Output:  04_DAOU/EXPERIMENT/mb_location/{results.csv, records_location.csv.gz}
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dots.persist import save_records  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "04_DAOU", "EXPERIMENT")
OUT = os.path.join(EXP, "mb_location")
TRUE = [450, 900, 1350]
DELTAS = (45, 90)


def parse(c):
    if not isinstance(c, str):
        return []
    return sorted(int(float(x)) for x in re.findall(r"-?\d+(?:\.\d+)?", c))


def score(cps, d):
    exact = len(cps) == 3
    loc = exact and all(abs(a - t) <= d for a, t in zip(cps, TRUE))
    rec = sum(any(abs(c - t) <= d for c in cps) for t in TRUE) / 3
    prec = sum(any(abs(c - t) <= d for t in TRUE) for c in cps) / len(cps) if cps else np.nan
    return exact, loc, rec, prec


def e2_rows():
    rec = pd.read_csv(os.path.join(EXP, "e2", "records_alt.csv.gz"))
    res = pd.read_csv(os.path.join(EXP, "e2", "results.csv"))
    rows = []
    for _, r in res.iterrows():
        col = [c for c in rec.columns[3:] if abs(float(c) - r.beta) < 1e-9][0]
        g = rec[(rec.level == r.level) & (rec.stat == r.method)]
        for rep, c in zip(g.rep, g[col]):
            rows.append(dict(source="e2", method=r.method, level=r.level, rep=int(rep), cps=json.loads(c)))
    return rows


def seg_rows():
    s = pd.read_csv(os.path.join(EXP, "seg_baselines", "records_reps.csv.gz"))
    s = s[s.design == "MB"]
    return [dict(source="seg_baselines", method=m, level=lv, rep=int(rep), cps=parse(c))
            for m, lv, rep, c in zip(s.method, s.level, s.rep, s.cps)
            if m != "Holevo-partition"]          # Holevo partitioning alone is read from e2 above


def bmctc_rows():
    b = pd.read_csv(os.path.join(EXP, "bmctc", "records_mb.csv.gz"))
    b = b[(b.rule == "calibrated") & b.level.notna()]
    return [dict(source="bmctc", method=st, level=lv, rep=int(rep), cps=parse(c))
            for st, lv, rep, c in zip(b.stat, b.level, b.rep, b.cps)]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = e2_rows() + seg_rows() + bmctc_rows()
    recs = []
    for r in rows:
        x = dict(source=r["source"], method=r["method"], level=r["level"], rep=r["rep"],
                 k=len(r["cps"]), cps=" ".join(map(str, r["cps"])))
        for d in DELTAS:
            ex, loc, rc, pr = score(r["cps"], d)
            x.update({"exact3": int(ex), f"all3_within_{d}": int(loc), f"recall_{d}": rc, f"precision_{d}": pr})
        recs.append(x)
    p = os.path.join(OUT, "records_location.csv.gz")
    if os.path.exists(p):
        os.remove(p)
    save_records(OUT, "records_location.csv", recs)
    d = pd.DataFrame(recs)
    agg = d.groupby(["source", "method", "level"]).agg(
        reps=("rep", "size"), exact3=("exact3", "mean"),
        all3_within_45=("all3_within_45", "mean"), all3_within_90=("all3_within_90", "mean"),
        recall_90=("recall_90", "mean"), precision_90=("precision_90", "mean")).reset_index()
    agg.round(4).to_csv(os.path.join(OUT, "results.csv"), index=False)
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
