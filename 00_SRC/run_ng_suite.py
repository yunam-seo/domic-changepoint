#!/usr/bin/env python
"""Non-Gaussian dependence-change suite G1-G6 (Table 1, lower block; Supplementary Section B.11).

Same protocol as run_matmi_baseline (identical rank inputs, grid, raw and studentized forms,
max-over-grid null calibration) with the G-series generator of dots/synth_ng.py. Compares the
random-feature DOMI difference (key DOMI-diff) with the Gram form (matmi-a1, order-2 Renyi matmi-r2)
and the classical dependence baselines HSIC / dCor / Spearman / empirical-copula CvM.

Runs behind the article:
    python 00_SRC/run_ng_suite.py --scenarios G1,G2,G3 --tag A     # random-feature DOMI + baselines
    python 00_SRC/run_ng_suite.py --scenarios G4,G5,G6 --tag B
    python 00_SRC/run_ng_suite.py --scenarios Gk --no-deps --msub 0 --tag gfull_Gk
                                                                   # Gram form, k = 1..6
(the Gram-form commands with all flags are in run_gram_full.sh).
Outputs: 04_DAOU/EXPERIMENT/ng_suite[_<tag>]/{results.csv, summary.json, records_*.csv.gz, run.log}
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, csv, json, sys, time  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_matmi_baseline as R  # noqa: E402
from dots.synth_ng import SCENARIOS_NG, DEFAULT_W_NG, generate_ng  # noqa: E402

LABEL = {"DOMI-diff": "DOMI-diff"}   # paper name for the statistic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="G1,G2,G3,G4,G5,G6")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--no-matmi", action="store_true")
    ap.add_argument("--tag", default="", help="output subfolder suffix")
    ap.add_argument("--subsample", choices=["perseg", "bottomk"], default="perseg")
    ap.add_argument("--no-deps", action="store_true", help="skip the classical baselines (Gram-form run only)")
    ap.add_argument("--msub", type=int, default=256, help="rows per segment for the Gram form; 0 = every row")
    a = ap.parse_args()
    # point the shared runner at the G-series generator (module globals read at call time;
    # Pool forks after this, so workers inherit the patch)
    R.SCENARIOS, R.DEFAULT_W, R.generate = SCENARIOS_NG, DEFAULT_W_NG, generate_ng
    R.SCEN_ID.update({k: 10 + int(k[1:]) for k in SCENARIOS_NG})
    R.BASE = os.path.join(R.ROOT, "04_DAOU", "EXPERIMENT", "ng_suite" + (("_" + a.tag) if a.tag else ""))
    os.makedirs(R.BASE, exist_ok=True)
    R.SUBMODE = a.subsample
    R.MSUB = a.msub if a.msub > 0 else 10 ** 9
    cfg = dict(R.CFG, matmi=not a.no_matmi, deps=not a.no_deps, subsample=a.subsample, msub=a.msub)
    t0 = time.time()
    rows, bw = R.run(cfg, a.procs, a.reps, a.scenarios.split(","))
    for r in rows:
        r["method"] = LABEL.get(r["method"], r["method"])
        r["key"] = LABEL.get(r["key"].split("|")[0], r["key"].split("|")[0]) + "|" + r["key"].split("|")[1]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(R.BASE, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(config=dict(cfg, reps=a.reps, scenarios=a.scenarios, MSUB=R.MSUB),
                   scenarios={k: v["name"] for k, v in SCENARIOS_NG.items()},
                   note="DOMI-diff is the DOMI difference statistic",
                   bandwidths=bw, runtime_sec=round(time.time() - t0, 1), rows=rows),
              open(os.path.join(R.BASE, "summary.json"), "w"), indent=1, ensure_ascii=False)
    R.say(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
