#!/usr/bin/env python
"""Gram form of DOMI on every row versus a 256-row per-segment subsample (Supplementary Section B.10).

Recomputes the Gram form (run_matmi_baseline.py) on the same replicates twice: on every row of each
segment, and on a fresh subsample of at most 256 rows per segment, so the two differ only in the
subsample. Reports power and localization for both.

Cells: G2 tau=0.5, G4 a=0.5, G1 nu=2 (non-Gaussian suite) and D2 (S2) a=0.7. Global form only (the
studentized form is derived from it); 200 null and 200 alternative replicates per cell; calibration
exactly as in run_matmi_baseline (first half of the null -> studentizing moments, second half ->
threshold; max over the grid; |tau_hat - tau| <= 30).

Run:  python 00_SRC/run_gram_subsample_check.py --procs 24
Output: 04_DAOU/EXPERIMENT/gram_subsample_check/{results.csv, records_null.csv.gz, records_alt.csv.gz, run.log}
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, csv, sys, time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_matmi_baseline as R  # noqa: E402
from dots.domi import DOMIContext, domi_stats  # noqa: E402
from dots.synth import SCENARIOS, generate  # noqa: E402
from dots.synth_ng import SCENARIOS_NG, generate_ng  # noqa: E402
from dots.extras import studentize  # noqa: E402
from dots.evaluate import summarize_alt, aggregate  # noqa: E402
from dots.persist import save_records  # noqa: E402

OUT = os.path.join(R.ROOT, "04_DAOU", "EXPERIMENT", "gram_subsample_check")
CELLS = [("G2", 0.5), ("G4", 0.5), ("G1", 2.0), ("D2", 0.7)]
W = 60


def _gen(scen, level, rep, null):
    if scen.startswith("G"):
        li = SCENARIOS_NG[scen]["levels"].index(level)
        return generate_ng(scen, level, rep, null=null, base_seed=R.CFG["base_seed"], level_idx=li if not null else 0)
    li = SCENARIOS[scen]["levels"].index(level)
    return generate(scen, level if not null else SCENARIOS[scen]["levels"][0], rep, null=null,
                    base_seed=R.CFG["base_seed"], level_idx=li if not null else 0)


def curves(a):
    """Global-form curves on one replicate: DOMI, Gram (alpha=1, 2) subsampled and full."""
    scen, level, rep, null = a
    smp = _gen(scen, level, rep, null)
    Z = smp["Z"]
    X, Y = Z[:, smp["blocks"][0]], Z[:, smp["blocks"][1]]
    ctx = DOMIContext(X, Y, W, D=8, seed=2026, n_perm=0)
    gx, gy = R.rff_gamma(ctx.UX, 2026), R.rff_gamma(ctx.UY, 2027)
    base = [R.SUBSEED, 90 + CELLS.index((scen, level)), 0, rep, int(null)]
    st = {"DOMI|g": domi_stats(ctx, "global")["DOMI-diff"]}
    for tag, msub in (("sub256", 256), ("full", 10 ** 9)):
        R.MSUB = msub
        a1, r2 = R.matmi_curves(ctx, "global", gx, gy, base)
        st[f"gram-a1-{tag}|g"], st[f"gram-r2-{tag}|g"] = a1, r2
    return st, ctx.grid, smp["tau"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--procs", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "run.log"), "a")
    rows = []
    with Pool(a.procs) as pool:
        for scen, level in CELLS:
            t0 = time.time()
            nulls = [x[0] for x in pool.map(curves, [(scen, level, r, True) for r in range(a.reps)], chunksize=2)]
            keys = list(nulls[0])
            half = a.reps // 2
            mu0 = {k: np.mean([d[k] for d in nulls[:half]], axis=0) for k in keys}
            sd0 = {k: np.std([d[k] for d in nulls[:half]], axis=0) for k in keys}
            thr = {k: float(np.quantile([np.nanmax(d[k]) for d in nulls], 0.95)) for k in keys}
            for k in keys:
                thr[k[:-1] + "gs"] = float(np.quantile(
                    [np.nanmax(studentize(d[k], mu0[k], sd0[k])) for d in nulls[half:]], 0.95))
            alts = pool.map(curves, [(scen, level, r, False) for r in range(a.reps)], chunksize=2)
            recs = []
            for st, grid, tau in alts:
                for k in list(st):
                    st[k[:-1] + "gs"] = studentize(st[k], mu0[k], sd0[k])
                recs.append(summarize_alt(st, grid, tau, W, R.CFG["n"], thr, R.CFG["tol"]))
            save_records(OUT, "records_null.csv",
                         [{f"max|{k}": float(np.nanmax(d[k])) for k in keys} for d in nulls],
                         {"scen": scen, "level": level})
            save_records(OUT, "records_alt.csv", recs, {"scen": scen, "level": level})
            agg = aggregate(recs, thr)
            for k, v in sorted(agg.items()):
                rows.append(dict(scen=scen, level=level, key=k, power=round(v["power"], 4),
                                 detect_rate=round(v["detect_rate"], 4), n_reps=v["n_reps"]))
            msg = f"[{time.strftime('%H:%M:%S')}] {scen} {level}: {time.time()-t0:.0f}s | " + ", ".join(
                f"{k}={v['power']:.2f}/{v['detect_rate']:.2f}" for k, v in sorted(agg.items()) if k.endswith("gs"))
            print(msg, flush=True)
            log.write(msg + "\n"); log.flush()
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log.write(f"[{time.strftime('%H:%M:%S')}] done\n")


if __name__ == "__main__":
    main()
