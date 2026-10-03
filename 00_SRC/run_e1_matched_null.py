#!/usr/bin/env python
"""The E1 comparison with each alternative calibrated against the null at its own level (all five
scenarios; for D1, D2 and M1 this null does not depend on the level).

`run_experiment.run_e1` draws its 500 null replicates once per scenario, at
`SCENARIOS[s]["levels"][0]`, and applies the resulting threshold and studentizing moments to every
alternative level. For D1, D2 and M1 the null is a pair of independent standard normals and does
not depend on `level`, so that is exact. For D3 and D4 it is not:

    D3  null = Gaussian copula at Kendall's tau = level        (level-dependent)
    D4  null = mixed-sign dependence at r = level, throughout  (level-dependent)

Both nulls carry more dependence at the higher levels, and the sampling variability of the segment
statistics grows with it, so a threshold taken from the level-0 null would be too low. The S3 and
S4 cells of Table 1 come from this runner (`rebuild_table1.py` states the selection rule).

This runner runs E1 with the null drawn at the SAME level as the
alternative, the reference of Section 5: the alternative's pre-change regime held throughout.
Everything else -- generators, seeds, candidate grid, the split of the 500 null replicates into
pointwise moments and threshold quantile, the |tau_hat - tau| <= 30 tolerance, and
the full method set -- is `run_e1` unchanged.

Writes to its own directory, separate from the level-0 null results of e1/.

ONE CELL PER INVOCATION. `dots.persist.save_records` reads the whole existing file back and
rewrites it on every append so that no column can be silently dropped; the E1 null records hold one
full curve per method per replicate (about 72 MB per cell), so each cell gets its own record files,
written exactly once.

Run:  python 00_SRC/run_e1_matched_null.py --scen D3 --li 1
Writes 04_DAOU/EXPERIMENT/e1_matched_null/{results_<scen>_<level>.csv,
records_{null,alt}_<scen>_<level>.csv.gz,run.log}; combine with `--combine`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from dots.synth import SCENARIOS, generate  # noqa: E402
from dots.extras import studentize  # noqa: E402
from dots.evaluate import summarize_alt, aggregate  # noqa: E402
from dots.persist import save_records  # noqa: E402
import run_experiment as E  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e1_matched_null")
CFG = E.CFG


def _null(a):
    """As run_experiment._e1_null, but at the alternative's own level."""
    s, li, r, cfg = a
    smp = generate(s, SCENARIOS[s]["levels"][li], r, null=True,
                   base_seed=cfg["base_seed"], level_idx=li)
    return E.e1_stats(smp, cfg["w"][s], cfg)[0]


def _alt(a):
    s, li, r, cfg, thr, mu0, sd0 = a
    smp = generate(s, SCENARIOS[s]["levels"][li], r, null=False,
                   base_seed=cfg["base_seed"], level_idx=li)
    st, grid = E.e1_stats(smp, cfg["w"][s], cfg)
    for k in list(st):
        if k.endswith("|g"):
            st[k[:-1] + "gs"] = studentize(st[k], mu0[k], sd0[k])
    return summarize_alt(st, grid, smp["tau"], cfg["w"][s], cfg["n"], thr, cfg["tol"])


def calibrate(nulls, reps, fpr):
    """Section 5 protocol: for the studentized variant, first half for the pointwise moments and
    second half for the threshold quantile; the raw variant needs no moments and takes its
    threshold from all replicates, as in run_e1."""
    keys = list(nulls[0].keys())
    half = reps // 2
    mu0 = {k: np.mean([d[k] for d in nulls[:half]], axis=0) for k in keys if k.endswith("|g")}
    sd0 = {k: np.std([d[k] for d in nulls[:half]], axis=0) for k in keys if k.endswith("|g")}
    thr = {k: float(np.quantile([np.nanmax(d[k]) for d in nulls], 1 - fpr)) for k in keys}
    for k in keys:
        if k.endswith("|g"):
            thr[k[:-1] + "gs"] = float(np.quantile(
                [np.nanmax(studentize(d[k], mu0[k], sd0[k])) for d in nulls[half:]], 1 - fpr))
    return thr, mu0, sd0


def combine():
    """Merge the per-cell aggregates into one results.csv, in scenario/level order."""
    import csv as _csv
    rows = []
    for s in ("D1", "D2", "D3", "D4", "M1"):
        for level in SCENARIOS[s]["levels"]:
            p = os.path.join(OUT, f"results_{s}_{level}.csv")
            if os.path.exists(p):
                rows += list(_csv.DictReader(open(p)))
    if not rows:
        print("nothing to combine")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = _csv.DictWriter(f, fieldnames=keys)
        wr.writeheader(); wr.writerows(rows)
    print(f"combined {len(rows)} rows -> {os.path.join(OUT, 'results.csv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen")
    ap.add_argument("--li", type=int)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.combine:
        return combine()

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        with open(os.path.join(OUT, "run.log"), "a") as f:
            f.write(line + "\n")

    s, li = a.scen, a.li
    level = SCENARIOS[s]["levels"][li]
    tag = f"{s}_{level}"
    rows = []
    with Pool(a.procs) as pool:
        t0 = time.time()
        nulls = pool.map(_null, [(s, li, r, CFG) for r in range(a.reps)], chunksize=4)
        save_records(OUT, f"records_null_{tag}.csv", nulls, {"scen": s, "level": level})
        thr, mu0, sd0 = calibrate(nulls, a.reps, CFG["fpr"])
        say(f"{s} level={level}: matched null {a.reps} reps {time.time()-t0:.0f}s")

        t1 = time.time()
        recs = pool.map(_alt, [(s, li, r, CFG, thr, mu0, sd0) for r in range(a.reps)], chunksize=4)
        save_records(OUT, f"records_alt_{tag}.csv", recs, {"scen": s, "level": level})
        agg = aggregate(recs, thr)
        for k, v in agg.items():
            name, mode = k.split("|")
            rows.append(dict(scen=s, level=level, method=name, mode=mode, key=k,
                             null_level=level, **v))
        top = sorted(agg.items(), key=lambda kv: -kv[1]["power"])[:6]
        say(f"{s} level={level}: alt {time.time()-t1:.0f}s | "
            + ", ".join(f"{k}={v['power']:.2f}" for k, v in top))

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    import csv as _csv
    with open(os.path.join(OUT, f"results_{tag}.csv"), "w", newline="") as f:
        wr = _csv.DictWriter(f, fieldnames=keys)
        wr.writeheader(); wr.writerows(rows)
    say(f"written: results_{tag}.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
