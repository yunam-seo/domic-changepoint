#!/usr/bin/env python
"""Monte Carlo evaluation of the subsample-rank copula change-point test under the Table 1 protocol.

Purpose
-------
The empirical-copula Cramer-von Mises statistic computed from global pseudo-observations
(dots.domi.copula_cvm) rejects under the marginal-only control M1. The test of Buecher, Kojadinovic,
Rohmer and Segers (2014) ranks WITHIN each subsample (cvm_subsample.py). This runner measures the
false-alarm rate of that test under M1 and its localized power on S1-S4 and G1-G6 (the CvM column
of Table 1; Section 6.2; Supplementary Section B.15).

Protocol (identical to Table 1; the calibration machinery is imported from run_matmi_baseline)
-----------------------------------------------------------------------------------------------
n = 600, tau = 300, candidate grid k = 60..540, 500 null and 500 alternative replicates per cell,
same generators and seeds; threshold = 0.95 quantile of the null max-over-grid; the studentized
variant uses per-candidate null moments from the first 250 null replicates and its threshold from
the last 250; a detection counts toward localized power only if |tau_hat - tau| <= 30.
S1 (D1), S2 (D2) and M1 at every level are calibrated against the level-0 null (the null does not
depend on the level there); S3 (D3) and S4 (D4) at every level against the matched null (pre-change regime held throughout), as in run_e1_matched_null /
run_matmi_baseline --matched; G1-G6 at every level against their independence null.

Statistic computed on every replicate (global split, raw "g" and studentized "gs"):
  CvM-sub      subsample-rank copula CvM (the statistic of Buecher et al., cvm_subsample.py)

Run:   python 00_SRC/run_cvm_subsample.py --procs 12
Outputs: 04_DAOU/EXPERIMENT/cvm_subsample/
    results.csv          one row per (scenario, level, statistic, variant): power, detect_rate,
                         binomial standard errors, threshold, n_reps
    summary.json         configuration, rows
    records_null.csv.gz  per-replicate null curves' maxima (one row per replicate and statistic)
    records_alt.csv.gz   per-replicate alternative records (max, tau_hat, detected, power_hit)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import multiprocessing  # noqa: E402

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

import run_matmi_baseline as R  # noqa: E402
from dots import synth  # noqa: E402
from dots.synth_ng import SCENARIOS_NG, DEFAULT_W_NG, generate_ng  # noqa: E402
from dots.domi import DOMIContext  # noqa: E402
from cvm_subsample import copula_cvm_subsample  # noqa: E402

OUT = os.path.join(R.ROOT, "04_DAOU", "EXPERIMENT", "cvm_subsample")

# (family, scenarios, matched null?, level filter or None for all levels) -- the cells of Table 1 plus
# every M1 and G level
BATCHES = [
    ("D", ["D1"], False, None),
    ("D", ["D2"], False, None),
    ("D", ["M1"], False, None),
    ("D", ["D3"], True, None),
    ("D", ["D4"], True, None),
    ("G", ["G1", "G2", "G3", "G4", "G5", "G6"], False, None),
]


def stats_cvm(smp, w, cfg, base):
    """Replacement for run_matmi_baseline.stats_one: the global CvM-sub curve."""
    Z = smp["Z"]
    X, Y = Z[:, smp["blocks"][0]], Z[:, smp["blocks"][1]]
    ctx = DOMIContext(X, Y, w, D=cfg["D"], seed=2026, n_perm=0)
    st = {"CvM-sub|g": copula_cvm_subsample(ctx, "global")}
    return st, ctx.grid, (np.nan, np.nan)


def se(p, n):
    return float(np.sqrt(p * (1 - p) / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--families", default="D,G", help="families to (re)run; rows of the others are kept")
    a = ap.parse_args()
    run_fams = set(a.families.split(","))
    os.makedirs(OUT, exist_ok=True)
    R.BASE = OUT
    # Long-lived workers in place of the imported runner's recycled ones (maxtasksperchild);
    # results do not depend on worker identity.
    R.Pool = lambda procs, maxtasksperchild=None: multiprocessing.Pool(procs)
    R.stats_one = stats_cvm                    # workers fork after this, so they inherit it
    t0 = time.time()
    rows = []
    old = os.path.join(OUT, "results.csv")
    if run_fams != {"D", "G"} and os.path.exists(old):
        keep = [r for r in csv.DictReader(open(old)) if r["scen"][0] not in "".join(
            "DM" if f == "D" else f for f in run_fams)]
        rows += keep
    for family, scens, matched, levels in BATCHES:
        if family not in run_fams:
            continue
        if family == "G":
            R.SCENARIOS, R.DEFAULT_W, R.generate = SCENARIOS_NG, DEFAULT_W_NG, generate_ng
            R.SCEN_ID.update({k: 10 + int(k[1:]) for k in SCENARIOS_NG})
        else:
            R.SCENARIOS, R.DEFAULT_W, R.generate = synth.SCENARIOS, synth.DEFAULT_W, synth.generate
        cfg = dict(R.CFG, matched=matched, modes=("global",))
        if levels:
            cfg["levels"] = levels
        got, _ = R.run(cfg, a.procs, a.reps, scens)
        for r in got:
            r["null"] = "matched" if matched else "level-0"
            r["power_se"] = round(se(r["power"], r["n_reps"]), 4)
            r["detect_se"] = round(se(r["detect_rate"], r["n_reps"]), 4)
        rows += got
    fields = ["scen", "level", "method", "mode", "key", "null", "power", "power_se", "detect_rate",
              "detect_se", "threshold", "n_reps"]
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(config=dict(R.CFG, reps=a.reps, batches=[[b[0], b[1], b[2], b[3]] for b in BATCHES],
                               grid="k = w..n-w, w = 60", statistic_module="cvm_subsample.py"),
                   runtime_sec=round(time.time() - t0, 1), rows=rows),
              open(os.path.join(OUT, "summary.json"), "w"), indent=1, ensure_ascii=False)
    R.say(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
