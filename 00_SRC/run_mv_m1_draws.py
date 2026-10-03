#!/usr/bin/env python
"""Diagnostic: rejection rate of the DOMI block test on the margins-only control MV-M1 at d = 1 over four further
independent draws of replicates and null calibration (the run of run_mv_blocks.py gave 0.106 at d = 1, against about
0.05 for the M1 design of Table 1). Same generator, statistic and calibration as run_mv_blocks.py (pointwise
studentization from null half 1, threshold = 0.95 quantile of the studentized null maxima of half 2), DOMI only;
draw j uses seeds offset by 10_000_000 * (j + 1). Reports the unlocalized rejection rate and the localized rate.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_mv_m1_draws.py --procs 24
Writes 04_DAOU/EXPERIMENT/mv_m1_draws/{config.json, records.csv, results.json}"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, csv, json, sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
import numpy as np  # noqa: E402
SRC = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SRC)
import run_mv_blocks as M  # noqa: E402
from dots.domi import DOMIContext, domi_stats  # noqa: E402
OUT = os.path.join(os.path.dirname(SRC), "04_DAOU", "EXPERIMENT", "mv_m1_draws")
DRAWS, REPS, D_BLOCK = 4, 500, 1

def job(a):
    j, r, null = a
    seed = 20260830 + 1000 * D_BLOCK + r + (500000 if null else 0) + 10_000_000 * (j + 1)
    X, Y = M.sample_m1(D_BLOCK, M.SCALE, seed, null)
    ctx = DOMIContext(X, Y, M.W, D=M.D, seed=2026, n_perm=0)
    return np.asarray(domi_stats(ctx, "global")["DOMI-diff"]), ctx.grid

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=24); a = ap.parse_args()
    if os.path.exists(os.path.join(OUT, "results.json")):
        raise SystemExit("results exist: use a new folder")
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(design="MV-M1 d=1 (run_mv_blocks.sample_m1, scale 2)", n=M.N, tau=M.TAU, tol=M.TOL, W=M.W, D=M.D,
                   draws=DRAWS, reps=REPS, seed_offset="10_000_000*(draw+1)", feature_seed=2026), open(os.path.join(OUT, "config.json"), "w"), indent=1)
    res, rec = {}, []
    with Pool(a.procs) as pool:
        for j in range(DRAWS):
            nulls = pool.map(job, [(j, r, True) for r in range(REPS)], chunksize=4)
            alts = pool.map(job, [(j, r, False) for r in range(REPS)], chunksize=4)
            grid = nulls[0][1]; half = REPS // 2
            st = np.array([c for c, _ in nulls[:half]]); mu, sd = st.mean(0), st.std(0)
            def smax(c):
                v = (c - mu) / np.where(sd > 0, sd, 1.0); i = int(np.nanargmax(v)); return float(v[i]), int(grid[i])
            thr = float(np.quantile([smax(c)[0] for c, _ in nulls[half:]], 0.95))
            out = [smax(c) for c, _ in alts]
            rej = np.mean([m > thr for m, _ in out]); loc = np.mean([m > thr and abs(t - M.TAU) <= M.TOL for m, t in out])
            res[str(j)] = dict(reject_rate=float(rej), localised_rate=float(loc), threshold=thr)
            rec += [dict(draw=j, rep=i, stat=m, tau_hat=t, reject=int(m > thr)) for i, (m, t) in enumerate(out)]
            print(j, res[str(j)], flush=True)
    rr = [v["reject_rate"] for v in res.values()]
    res["summary"] = dict(min=min(rr), max=max(rr), mean=float(np.mean(rr)))
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    with open(os.path.join(OUT, "records.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["draw", "rep", "stat", "tau_hat", "reject"]); w.writeheader(); w.writerows(rec)
    print(res["summary"])

if __name__ == "__main__":
    main()
