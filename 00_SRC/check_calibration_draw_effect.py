#!/usr/bin/env python
"""How much of a reported power is the calibration draw rather than the method?

A reported power depends on which 500 null replicates set the threshold and the pointwise
studentizing moments. This script scores ONE fixed set of alternative curves under several
independent null draws, the deployed calibration among them, so any difference between the
resulting powers is the calibration draw alone (Supplementary Section B.5).

Run:  python 00_SRC/check_calibration_draw_effect.py --scen D2 --li 1 --procs 6
Writes 04_DAOU/EXPERIMENT/diagnostics/calibration_draw_effect_<scen>_<level>.json.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

csv.field_size_limit(10 ** 9)
SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from dots.synth import SCENARIOS, generate  # noqa: E402
from dots.extras import studentize  # noqa: E402
from dots.evaluate import summarize_alt, aggregate  # noqa: E402
import run_experiment as E  # noqa: E402

EXP = os.path.join(ROOT, "04_DAOU", "EXPERIMENT")
OUT = os.path.join(EXP, "diagnostics")
REC = os.path.join(EXP, "e1_matched_null")
CFG = E.CFG
REPS = 500
# e1_matched_null saved one null record file per (scenario, level). Because the null of D1, D2 and M1
# does not depend on the level at all, and that of D3/D4 does, the files usable as INDEPENDENT
# DRAWS FROM THE SAME NULL are exactly the three levels of a level-independent scenario; for D3 and
# D4 only the matching level's file is a valid null, so those cells get one draw and no spread.
LEVEL_INDEPENDENT = {"D1", "D2", "M1"}
# For D3 and D4 the other levels' null files are NOT draws from this cell's null, so a spread has to
# come from fresh null draws at the same level under a different base seed. These are the seeds; the
# alternative always stays on the deployed seed, so the curves being scored never change.
EXTRA_SEEDS = [20260901, 20260902]


def _alt(a):
    s, li, r, cfg = a
    smp = generate(s, SCENARIOS[s]["levels"][li], r, null=False,
                   base_seed=cfg["base_seed"], level_idx=li)
    return E.e1_stats(smp, cfg["w"][s], cfg)


def _null(a):
    s, li, r, cfg = a
    smp = generate(s, SCENARIOS[s]["levels"][li], r, null=True,
                   base_seed=cfg["base_seed"], level_idx=li)
    return E.e1_stats(smp, cfg["w"][s], cfg)[0]


def read_nulls(path):
    """{key: array over replicates of the null curve} for the '|g' statistics."""
    rows = []
    with gzip.open(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    keys = [k for k in rows[0] if k.endswith("|g")]
    return {k: np.array([np.fromstring(r[k].strip("[]").replace("\n", " "), sep=" ")
                         for r in rows]) for k in keys}


def calibrate(nulls, fpr):
    half = len(next(iter(nulls.values()))) // 2
    mu0, sd0, thr = {}, {}, {}
    for k, N in nulls.items():
        mu0[k], sd0[k] = N[:half].mean(0), N[:half].std(0)
        thr[k] = float(np.quantile([np.nanmax(c) for c in N], 1 - fpr))
        thr[k[:-1] + "gs"] = float(np.quantile(
            [np.nanmax(studentize(c, mu0[k], sd0[k])) for c in N[half:]], 1 - fpr))
    return thr, mu0, sd0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", default="D1")
    ap.add_argument("--li", type=int, default=1)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    SCEN, LI = a.scen, a.li
    level = SCENARIOS[SCEN]["levels"][LI]
    fresh = SCEN not in LEVEL_INDEPENDENT
    if fresh:
        # The null depends on the level here, so other levels' files are a different null. Draw
        # fresh nulls at THIS level under other base seeds instead.
        draws = {f"base_seed={CFG['base_seed']} (the matched-null re-run)": None}
        draws.update({f"base_seed={s}": s for s in EXTRA_SEEDS})
    else:
        draws = {f"draw from level_idx={i}"
                 + (" (Table 1 calibration)" if i == 0 else ""):
                 "records_null_%s_%s.csv.gz" % (SCEN, SCENARIOS[SCEN]["levels"][i])
                 for i in range(len(SCENARIOS[SCEN]["levels"]))}

    with Pool(a.procs) as pool:
        alts = pool.map(_alt, [(SCEN, LI, r, CFG) for r in range(REPS)], chunksize=4)
        fresh_nulls = {}
        if fresh:
            for tag, seed in draws.items():
                cfg = dict(CFG) if seed is None else {**CFG, "base_seed": seed}
                got = pool.map(_null, [(SCEN, LI, r, cfg) for r in range(REPS)], chunksize=4)
                fresh_nulls[tag] = {k: np.array([g[k] for g in got])
                                    for k in got[0] if k.endswith("|g")}
                print(f"  null draw ready: {tag}", flush=True)
    tau = int(CFG["tau"])
    print(f"\nalternative curves: {len(alts)} replicates of {SCEN} level={level:g} "
          f"(the same curves under every calibration)\n")

    out = {"scen": SCEN, "level": level, "reps": REPS,
           "null_draws": "fresh base seeds" if fresh else "other levels of a level-independent null",
           "calibrations": []}
    show = ["DOMI-diff|gs", "dCor-diff|gs", "HSIC-diff|gs", "Spearman-diff|g", "GaussLR|g"]
    print(f"  {'calibration (null draw)':36s} " + " ".join(f"{s:>17s}" for s in show))
    rec_rows = []
    for tag, fn in draws.items():
        nulls = fresh_nulls[tag] if fresh else read_nulls(os.path.join(REC, fn))
        thr, mu0, sd0 = calibrate(nulls, CFG["fpr"])
        recs = []
        for st, grid in alts:
            s2 = dict(st)
            for k in list(s2):
                if k.endswith("|g"):
                    s2[k[:-1] + "gs"] = studentize(s2[k], mu0[k], sd0[k])
            recs.append(summarize_alt(s2, grid, tau, CFG["w"][SCEN], CFG["n"], thr, CFG["tol"]))
        agg = aggregate(recs, thr)
        # one row per (calibration, replicate, statistic): the aggregate above is a mean of the
        # power_hit column, and the threshold a quantile of the stored null maxima below
        for ri, r in enumerate(recs):
            for k, v in r.items():
                rec_rows.append(dict(calibration=tag, rep=ri, key=k, max=v["max"],
                                     tau_hat=v["tau_hat"], detected=int(v["detected"]),
                                     power_hit=int(v["power_hit"]), threshold=thr.get(k, "")))
        # the null side, enough to regenerate every threshold: per-replicate raw maxima and,
        # for the studentized form, the second-half studentized maxima under the stored moments
        half = len(next(iter(nulls.values()))) // 2
        for k, N in nulls.items():
            for ri, c in enumerate(N):
                row = dict(calibration=tag, rep=ri, key="null:" + k,
                           max=float(np.nanmax(c)), tau_hat="", detected="", power_hit="",
                           threshold="")
                rec_rows.append(row)
                if ri >= half:
                    rec_rows.append(dict(calibration=tag, rep=ri, key="null:" + k[:-1] + "gs",
                                         max=float(np.nanmax(studentize(c, mu0[k], sd0[k]))),
                                         tau_hat="", detected="", power_hit="", threshold=""))
        print(f"  {tag:36s} " + " ".join(f"{agg[s]['power']:17.3f}" for s in show))
        out["calibrations"].append(dict(calibration=tag,
                                        **{s: round(agg[s]["power"], 3) for s in show}))
    sp = {s: round(max(c[s] for c in out["calibrations"])
                   - min(c[s] for c in out["calibrations"]), 3) for s in show}
    out["spread"] = sp
    print(f"  {'spread across draws':36s} " + " ".join(f"{sp[s]:17.3f}" for s in show))
    fn = f"calibration_draw_effect_{SCEN}_{level:g}.json"
    json.dump(out, open(os.path.join(OUT, fn), "w"), indent=1)
    rp = os.path.join(OUT, f"records_calib_draw_{SCEN}_{level:g}.csv.gz")
    with gzip.open(rp, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["calibration", "rep", "key", "max", "tau_hat",
                                           "detected", "power_hit", "threshold"])
        w.writeheader()
        w.writerows(rec_rows)
    print("records:", rp, f"({len(rec_rows)} rows)")
    print("\nwritten:", os.path.join(OUT, fn))


if __name__ == "__main__":
    main()
