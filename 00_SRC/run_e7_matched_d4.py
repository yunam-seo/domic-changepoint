#!/usr/bin/env python
"""Model-based baselines of Supplementary Section B.4 on S4 (code key D4) with each level calibrated against its own
null. In S4 the no-change regime is the sign-mixed dependence at level r, so the null depends on r; run_e7_modelbased.py
drew every scenario's null at the first level, which is correct for S1, S2 and M1 (null = independence or unchanged
scale) but not for S4 at r = 0.7 and 0.85. This runner draws the S4 null at every level (same generator, seeds and
statistics as run_e7_modelbased._job) and re-scores the stored alternative replicates of e7/records_reps.csv against the
matched thresholds. The first level reproduces the original threshold, which is checked.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_e7_matched_d4.py --procs 20
Writes 04_DAOU/EXPERIMENT/e7_matched_d4/{config.json, records_null.csv, results.csv}"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, csv, json, sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from multiprocessing import Pool  # noqa: E402
import numpy as np  # noqa: E402
SRC = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(SRC); sys.path.insert(0, SRC)
import run_e7_modelbased as E  # noqa: E402
from dots.synth import SCENARIOS  # noqa: E402
EXP = os.path.join(ROOT, "04_DAOU", "EXPERIMENT")
OUT = os.path.join(EXP, "e7_matched_d4")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=20); a = ap.parse_args()
    if os.path.exists(os.path.join(OUT, "results.csv")):
        raise SystemExit("results exist: use a new folder")
    cfg = dict(E.CFG); levels = SCENARIOS["D4"]["levels"]
    json.dump(dict(cfg, scen="D4", levels=levels, null="matched: drawn at each level (null=True, level_idx=li)",
                   alternatives="stored in e7/records_reps.csv"), open(os.path.join(OUT, "config.json"), "w"), indent=1)
    alt = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(EXP, "e7", "records_reps.csv"))):
        if r["scen"] == "D4":
            alt[(float(r["level"]), r["method"])].append((float(r["stat"]), int(r["tau_hat"]), float(r["threshold"])))
    rows, rec = [], []
    with Pool(a.procs, maxtasksperchild=40) as pool:
        for li, lv in enumerate(levels):
            nulls = pool.map(E._job, [("D4", li, r, True, cfg) for r in range(cfg["reps"])], chunksize=2)
            for r_, d in enumerate(nulls):
                for m, (s, t) in d.items():
                    rec.append(dict(level=lv, method=m, rep=r_, stat=s, tau_hat=t))
            for m in nulls[0]:
                thr = float(np.quantile([d[m][0] for d in nulls], 1 - cfg["fpr"]))
                al = alt[(float(lv), m)]
                assert len(al) == cfg["reps"], (lv, m, len(al))
                old_thr = al[0][2]
                if li == 0:
                    assert abs(thr - old_thr) < 1e-9, ("first-level threshold not reproduced", m, thr, old_thr)
                pw = float(np.mean([s > thr and abs(t - cfg["tau"]) <= cfg["tol"] for s, t, _ in al]))
                det = float(np.mean([s > thr for s, _, _ in al]))
                rows.append(dict(scen="D4", level=lv, method=m, threshold=thr, threshold_first_level=old_thr, power=pw, detect_rate=det))
                print(lv, m, f"thr {thr:.4f} (first-level {old_thr:.4f})  power {pw:.3f}  detect {det:.3f}", flush=True)
    with open(os.path.join(OUT, "records_null.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec[0])); w.writeheader(); w.writerows(rec)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
