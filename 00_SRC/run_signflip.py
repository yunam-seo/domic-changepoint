#!/usr/bin/env python
"""A sign reversal of the correlation, +r -> -r, under the Monte Carlo calibration of Table 1
(Supplementary Section B.21, Table B.17).

Why: the Gaussian copula with correlation r and with -r have the same mutual information, and the Gram
form of DOMI is invariant under the reflection of one margin's ranks (u -> 1 - u), so its population
contrast for this change is zero; the random-feature form is invariant only approximately. This run
measures whether the deployed scan detects and localizes such a change.

Design
    Gaussian copula, standard normal margins, n = 600, break at tau = 300; correlation +r before and -r
    after, r in {0.35, 0.7}. Null: +r held throughout. 500 null and 500 alternative replicates per r.
    Calibration of Table 1 (run_offcentre.py): raw ("g") threshold = 95% quantile of the null maxima;
    studentized ("gs") per-candidate moments from nulls 0..249 and threshold from nulls 250..499.
    detect = max > threshold; localized = detect and |tau_hat - tau| <= 30. Candidates 60..540 (w = 60).
Statistics: the six of Table 1 (DOMI random-feature form D = 8, Gram form, HSIC, distance correlation,
    Spearman, the copula test with ranks within each segment), the correlation CUSUM of Supplementary
    Section B.4, the copula statistic from global pseudo-observations, and the combined statistic of
    Section 4.5, max of the studentized DOMI and Spearman curves' maxima, calibrated on the same nulls.

Run:    python 00_SRC/run_signflip.py --procs 20
        python 00_SRC/run_signflip.py --evaluate
Writes 04_DAOU/EXPERIMENT/signflip/{config.json, curves/, records_alt.csv.gz, results.csv}
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "signflip")
CURVES = os.path.join(OUT, "curves")
CFG = dict(n=600, tau=300, w=60, tol=30, D=8, fpr=0.05, reps=500, chunk=50, levels=[0.35, 0.7],
           base_seed=20260928)
STATS = ["DOMI-diff", "matmi-a1", "HSIC-diff", "dCor-diff", "Spearman-diff", "CvM-sub", "CorrCUSUM", "CopulaCvM"]


def sample(r_level, setname, rep):
    rng = np.random.default_rng([CFG["base_seed"], int(round(100 * r_level)), int(setname == "alt"), rep])
    n, tau = CFG["n"], CFG["tau"]
    x, e = rng.standard_normal(n), rng.standard_normal(n)
    rho = np.full(n, r_level)
    if setname == "alt":
        rho[tau:] = -r_level
    y = rho * x + np.sqrt(1 - rho * rho) * e
    return x[:, None], y[:, None]


def job(a):
    r_level, setname, k = a
    p = os.path.join(CURVES, f"r{r_level}_{setname}_{k:02d}.npz")
    if os.path.exists(p):
        return
    from dots.domi import DOMIContext, domi_stats, DEP_BASELINES
    from run_perm_compare import dcor_diff_fast
    from run_matmi_baseline import rff_gamma
    from gram_gpu import gram_curves_lowrank
    from cvm_subsample import copula_cvm_subsample
    from run_e7_modelbased import corr_cusum
    out = {m: [] for m in STATS}
    ux, uy, gx, gy = [], [], [], []
    for rep in range(k * CFG["chunk"], (k + 1) * CFG["chunk"]):
        X, Y = sample(r_level, setname, rep)
        ctx = DOMIContext(X, Y, CFG["w"], D=CFG["D"], seed=2026, n_perm=0)
        out["DOMI-diff"].append(domi_stats(ctx, "global")["DOMI-diff"])
        out["HSIC-diff"].append(DEP_BASELINES["HSIC-diff"](ctx, "global"))
        out["dCor-diff"].append(dcor_diff_fast(ctx))
        out["Spearman-diff"].append(DEP_BASELINES["Spearman-diff"](ctx, "global"))
        out["CopulaCvM"].append(DEP_BASELINES["CopulaCvM"](ctx, "global"))
        out["CvM-sub"].append(np.asarray(copula_cvm_subsample(ctx, "global"), float))
        ts, cc = corr_cusum(X[:, 0], Y[:, 0], CFG["w"])
        assert np.array_equal(ts, ctx.grid)
        out["CorrCUSUM"].append(cc)
        ux.append(ctx.UX[:, 0]); uy.append(ctx.UY[:, 0])
        gx.append(rff_gamma(ctx.UX, 2026)); gy.append(rff_gamma(ctx.UY, 2027))
    a1, _ = gram_curves_lowrank(np.array(ux), np.array(uy), np.array(gx), np.array(gy), CFG["w"])
    out["matmi-a1"] = list(np.asarray(a1, float))
    tmp = p[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **{m: np.array(v, float) for m, v in out.items()})
    os.replace(tmp, p)
    print(f"{r_level} {setname} chunk {k} done", flush=True)


def _load(r_level, setname):
    nch = CFG["reps"] // CFG["chunk"]
    parts = [np.load(os.path.join(CURVES, f"r{r_level}_{setname}_{k:02d}.npz")) for k in range(nch)]
    return {m: np.concatenate([p[m] for p in parts]) for m in STATS}


def evaluate():
    grid = np.arange(CFG["w"], CFG["n"] - CFG["w"] + 1)
    half, tau, tol, fpr = CFG["reps"] // 2, CFG["tau"], CFG["tol"], CFG["fpr"]
    rows, recs = [], []
    for r_level in CFG["levels"]:
        N, A = _load(r_level, "null"), _load(r_level, "alt")
        stud_null, stud_alt, thr = {}, {}, {}
        for m in STATS:
            thr[f"{m}|g"] = float(np.quantile(N[m].max(1), 1 - fpr))
            mu, sd = N[m][:half].mean(0), N[m][:half].std(0)
            sd = np.where(sd > 0, sd, 1.0)
            stud_null[m], stud_alt[m] = (N[m] - mu) / sd, (A[m] - mu) / sd
            thr[f"{m}|gs"] = float(np.quantile(stud_null[m][half:].max(1), 1 - fpr))
        comb_null = np.maximum(stud_null["DOMI-diff"].max(1), stud_null["Spearman-diff"].max(1))[half:]
        thr["combined|gs"] = float(np.quantile(comb_null, 1 - fpr))
        curves = {}
        for m in STATS:
            curves[f"{m}|g"] = A[m]
            curves[f"{m}|gs"] = stud_alt[m]
        for key, C in curves.items():
            mx, th = C.max(1), grid[C.argmax(1)]
            det = mx > thr[key]
            hit = det & (np.abs(th - tau) <= tol)
            m, form = key.split("|")
            for i in range(len(C)):
                recs.append(dict(r=r_level, stat=m, form=form, rep=i, max=repr(float(mx[i])), tau_hat=int(th[i]),
                                 detected=int(det[i]), hit=int(hit[i])))
            q_all = np.percentile(th, [25, 50, 75])
            q_det = np.percentile(th[det], [25, 50, 75]) if det.any() else [np.nan] * 3
            rows.append(dict(r=r_level, stat=m, form=form, detect_rate=round(float(det.mean()), 4),
                             power_localised=round(float(hit.mean()), 4),
                             tau_hat_q1_all=float(q_all[0]), tau_hat_median_all=float(q_all[1]), tau_hat_q3_all=float(q_all[2]),
                             tau_hat_q1_detected=float(q_det[0]), tau_hat_median_detected=float(q_det[1]),
                             tau_hat_q3_detected=float(q_det[2]), threshold=thr[key], reps=len(C)))
        # combined statistic: studentized maxima of DOMI and Spearman, located by the larger of the two
        dm, sm = stud_alt["DOMI-diff"].max(1), stud_alt["Spearman-diff"].max(1)
        cm = np.maximum(dm, sm)
        th = np.where(dm >= sm, grid[stud_alt["DOMI-diff"].argmax(1)], grid[stud_alt["Spearman-diff"].argmax(1)])
        det = cm > thr["combined|gs"]; hit = det & (np.abs(th - tau) <= tol)
        q_all = np.percentile(th, [25, 50, 75]); q_det = np.percentile(th[det], [25, 50, 75]) if det.any() else [np.nan] * 3
        rows.append(dict(r=r_level, stat="combined", form="gs", detect_rate=round(float(det.mean()), 4),
                         power_localised=round(float(hit.mean()), 4), tau_hat_q1_all=float(q_all[0]),
                         tau_hat_median_all=float(q_all[1]), tau_hat_q3_all=float(q_all[2]),
                         tau_hat_q1_detected=float(q_det[0]), tau_hat_median_detected=float(q_det[1]),
                         tau_hat_q3_detected=float(q_det[2]), threshold=thr["combined|gs"], reps=len(cm)))
        for i in range(len(cm)):
            recs.append(dict(r=r_level, stat="combined", form="gs", rep=i, max=repr(float(cm[i])), tau_hat=int(th[i]),
                             detected=int(det[i]), hit=int(hit[i])))
    with gzip.open(os.path.join(OUT, "records_alt.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]), lineterminator="\n"); w.writeheader(); w.writerows(recs)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    for x in rows:
        print(f"r={x['r']:<5} {x['stat']:14s} {x['form']:3s} detect {x['detect_rate']:.3f} localized {x['power_localised']:.3f} "
              f"tau_hat median(all) {x['tau_hat_median_all']:.0f} IQR [{x['tau_hat_q1_all']:.0f},{x['tau_hat_q3_all']:.0f}] "
              f"median(detected) {x['tau_hat_median_detected']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    os.makedirs(CURVES, exist_ok=True)
    cp = os.path.join(OUT, "config.json")
    if not os.path.exists(cp):
        json.dump(dict(CFG, statistics=STATS), open(cp, "w"), indent=1)
    elif json.load(open(cp)) != json.loads(json.dumps(dict(CFG, statistics=STATS))):
        sys.exit("config.json differs from CFG: use a new output folder")
    if not a.evaluate:
        nch = CFG["reps"] // CFG["chunk"]
        jobs = [(r, s, k) for k in range(nch) for r in CFG["levels"] for s in ("null", "alt")]
        with Pool(a.procs) as pool:
            for _ in pool.imap_unordered(job, jobs, chunksize=1):
                pass
    evaluate()


if __name__ == "__main__":
    main()
