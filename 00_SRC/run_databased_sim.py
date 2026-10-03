#!/usr/bin/env python
"""Data-based simulation with known changes: empirical weather margins, latent sign-mixed dependence.

Design (conditions C0-C3b fixed before any result was seen; C4-C5b are exploratory follow-ups)
    Margins: hourly observations of the first station of the analyzed list (105, Gangneung) in the
    first complete year (2018): temperature for X and relative humidity for Y. The winter (DJF) and
    summer (JJA) empirical quantile functions Q_winter, Q_summer of each variable are the two
    marginal regimes.
    Latent dependence: (U, V) with uniform margins from the S4 construction, Y = s r X + sqrt(1-r^2) E
    with an independent random sign s and r = 0.85 (correlation zero, dependence non-monotone), or
    independence; U = Phi(X), V = Phi(Y).
    Observations: X_t = Q_X(U_t), Y_t = Q_Y(V_t) with the regime's quantile functions. The recorded
    values are quantized (0.1 degC, 1 %), so the reference samples are first spread uniformly within
    their resolution (a fixed draw, seed [20260929, len(variable), len(season)], so samples whose
    names have equal length share one uniform stream) and the quantile function interpolates linearly
    between the order statistics of that continuous sample ("interp", no ties); the "rounded" variant
    then rounds the observations back to the sensor resolution (ties as in the record).
    n = 1200, change at 600; candidate grid w..n-w with w = 120; localization tolerance 60.

Conditions
    C0     sign-mixed throughout, winter margins throughout                       no change
    C1     sign-mixed throughout, winter -> summer margins at the change          margins only
    C2a    sign-mixed -> independent, winter margins throughout                   dependence only
    C2b    independent -> sign-mixed, winter margins throughout                   dependence only
    C3a    sign-mixed -> independent, winter -> summer margins                    both
    C3b    independent -> sign-mixed, winter -> summer margins                    both
Exploratory follow-up conditions, added after C0-C3b were seen, for a marginal change within one season:
    C4     sign-mixed throughout, December -> February margins                    margins only
    C5a    sign-mixed -> independent, December -> February margins                both
    C5b    independent -> sign-mixed, December -> February margins                both

Test (per replicate, identical for every statistic): K = 99 joint pair permutations shared by all
statistics; each statistic's curve recomputed on each permuted series; studentized variant (centered
and scaled over all K + 1 curves) for DOMI, HSIC-diff and dCor-diff, raw variant for Spearman-diff and
the copula CvM statistics, as in Supplementary Table B.11; p = (1 + #{T_k >= T_0}) / (K + 1) with the
tie rule of dots/perm.py; reject at p <= 0.05; localized hit = reject and |tau_hat - 600| <= 60.

Run:   OMP_NUM_THREADS=1 python 00_SRC/run_databased_sim.py --variant interp --procs 24
       python 00_SRC/run_databased_sim.py --variant rounded --procs 24
       python 00_SRC/run_databased_sim.py --aggregate
Needs 02_MART/WEATHER_HOURLY.csv (see README, Data).
Writes 04_DAOU/EXPERIMENT/databased_sim/{config.json, margins.json, records.csv.gz, results.csv}
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
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.domi import DOMIContext, domi_stats, DEP_BASELINES  # noqa: E402
from dots.perm import ge  # noqa: E402
import run_perm_compare as PC  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "databased_sim")
MART = os.path.join(ROOT, "02_MART", "WEATHER_HOURLY.csv")
CFG = dict(station="105", year="2018", x="ta", y="hm", resolution={"ta": 0.1, "hm": 1.0},
           n=1200, tau=600, w=120, tol=60, K=99, alpha=0.05, D=8, r=0.85, reps=200,
           seed=20260927, perm_seed=20260928)
CONDS = {   # name -> (dependence before, after, margins before, after)
    "C0": ("mix", "mix", "winter", "winter"),
    "C1": ("mix", "mix", "winter", "summer"),
    "C2a": ("mix", "ind", "winter", "winter"),
    "C2b": ("ind", "mix", "winter", "winter"),
    "C3a": ("mix", "ind", "winter", "summer"),
    "C3b": ("ind", "mix", "winter", "summer"),
    "C4": ("mix", "mix", "dec", "feb"),
    "C5a": ("mix", "ind", "dec", "feb"),
    "C5b": ("ind", "mix", "dec", "feb"),
}
COND_ID = {c: i + 1 for i, c in enumerate(CONDS)}
STATS = ["DOMI", "HSIC-diff", "dCor-diff", "Spearman-diff", "CopulaCvM", "CvM-sub"]
VARIANT = {"DOMI": "stud", "HSIC-diff": "stud", "dCor-diff": "stud", "Spearman-diff": "raw",
           "CopulaCvM": "raw", "CvM-sub": "raw"}
_Q = {}


def margins():
    """Sorted winter and summer samples of each variable (the empirical quantile functions)."""
    if _Q:
        return _Q
    rows = [r for r in csv.DictReader(open(MART)) if r["stn"] == CFG["station"] and r["tm"][:4] == CFG["year"]]
    for v in (CFG["x"], CFG["y"]):
        for season, months in (("winter", ("01", "02", "12")), ("summer", ("06", "07", "08")),
                               ("dec", ("12",)), ("feb", ("02",))):
            vals = np.array([float(r[v]) for r in rows if r["tm"][4:6] in months and r[v] not in ("", "nan")])
            vals = vals[np.isfinite(vals)]
            res = CFG["resolution"][v]
            jit = np.random.default_rng([20260929, len(v), len(season)]).uniform(-res / 2, res / 2, len(vals))
            _Q[(v, season)] = np.sort(vals + jit)
    return _Q


def qmap(u, sample):
    """Empirical quantile function, linear interpolation between order statistics."""
    return np.quantile(sample, np.clip(u, 0.0, 1.0), method="linear")


def sample(cond, r, variant):
    dep0, dep1, m0, m1 = CONDS[cond]
    n, tau, rr = CFG["n"], CFG["tau"], CFG["r"]
    rng = np.random.default_rng([CFG["seed"], COND_ID[cond], r])
    E = rng.standard_normal((n, 2))
    sgn = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    lx = E[:, 0]
    ly = E[:, 1].copy()
    for seg, dep in ((slice(0, tau), dep0), (slice(tau, n), dep1)):
        if dep == "mix":
            ly[seg] = sgn[seg] * rr * E[seg, 0] + np.sqrt(1 - rr * rr) * E[seg, 1]
    U, V = norm.cdf(lx), norm.cdf(ly)
    Q = margins()
    x, y = np.empty(n), np.empty(n)
    for seg, m in ((slice(0, tau), m0), (slice(tau, n), m1)):
        x[seg] = qmap(U[seg], Q[(CFG["x"], m)])
        y[seg] = qmap(V[seg], Q[(CFG["y"], m)])
    if variant == "rounded":
        x = np.round(x / CFG["resolution"][CFG["x"]]) * CFG["resolution"][CFG["x"]]
        y = np.round(y / CFG["resolution"][CFG["y"]]) * CFG["resolution"][CFG["y"]]
    return x[:, None], y[:, None]


def curves(X, Y, sub):
    ctx = DOMIContext(X, Y, CFG["w"], D=CFG["D"], seed=2026, n_perm=0)
    out = {"DOMI": domi_stats(ctx, "global")["DOMI-diff"]}
    for k in ("HSIC-diff", "Spearman-diff", "CopulaCvM"):
        out[k] = DEP_BASELINES[k](ctx, "global")
    out["dCor-diff"] = PC.dcor_diff_fast(ctx)
    out["CvM-sub"] = np.asarray(sub(ctx, "global"), float)
    return out


def job(a):
    cond, r, variant = a
    X, Y = sample(cond, r, variant)
    rng = np.random.default_rng([CFG["perm_seed"], COND_ID[cond], r])
    perms = [rng.permutation(CFG["n"]) for _ in range(CFG["K"])]
    sub = PC.cvm_sub_fn()
    per = [curves(X, Y, sub)] + [curves(X[i], Y[i], sub) for i in perms]
    grid = np.arange(CFG["w"], CFG["n"] - CFG["w"] + 1)
    rec = []
    for k in STATS:
        A = np.array([p[k] for p in per], float)
        if VARIANT[k] == "stud":
            A = (A - A.mean(0)) / (A.std(0) + 1e-12)
        T = A.max(1)
        p = (1 + int(np.sum(ge(T[1:], T[0])))) / (CFG["K"] + 1)
        tau_hat = int(grid[int(np.argmax(A[0]))])
        rej = p <= CFG["alpha"]
        rec.append(dict(variant=variant, cond=cond, rep=r, stat=k, form=VARIANT[k], p=p, T_obs=repr(float(T[0])),
                        tau_hat=tau_hat, reject=int(rej),
                        hit=int(rej and abs(tau_hat - CFG["tau"]) <= CFG["tol"])))
    return rec


def shard_path(variant, cond, r):
    return os.path.join(OUT, "shards", f"{variant}_{cond}_{r:03d}.json")


def work(a):
    variant, cond, r = a
    p = shard_path(variant, cond, r)
    if os.path.exists(p):
        return
    t0 = time.time()
    rec = job((cond, r, variant))
    tmp = p + ".tmp"
    json.dump(rec, open(tmp, "w"))
    os.replace(tmp, p)
    print(f"[{time.strftime('%H:%M:%S')}] {variant} {cond} rep {r}: {time.time()-t0:.0f}s", flush=True)


def aggregate():
    rows = []
    for variant in ("interp", "rounded"):
        for cond in CONDS:
            for r in range(CFG["reps"]):
                p = shard_path(variant, cond, r)
                if os.path.exists(p):
                    rows += json.load(open(p))
    with gzip.open(os.path.join(OUT, "records.csv.gz"), "wt", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    res = []
    for variant in ("interp", "rounded"):
        for cond in CONDS:
            for k in STATS:
                rr = [x for x in rows if x["variant"] == variant and x["cond"] == cond and x["stat"] == k]
                if not rr:
                    continue
                m = len(rr)
                rej = np.mean([x["reject"] for x in rr])
                hit = np.mean([x["hit"] for x in rr])
                err = [abs(x["tau_hat"] - CFG["tau"]) for x in rr]
                res.append(dict(variant=variant, cond=cond, stat=k, form=VARIANT[k], reps=m,
                                reject_rate=round(float(rej), 4), reject_se=round(float(np.sqrt(rej * (1 - rej) / m)), 4),
                                localised_power=round(float(hit), 4),
                                median_abs_loc_err=float(np.median(err))))
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(res[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(res)
    print(f"aggregated {len(rows)} records, {len(res)} cells")
    for x in res:
        print(f"{x['variant']:7s} {x['cond']:4s} {x['stat']:14s} reject {x['reject_rate']:.3f}  localized {x['localised_power']:.3f}  n={x['reps']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["interp", "rounded"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--reps", type=int, default=CFG["reps"])
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if not a.aggregate and a.variant is None:
        ap.error("--variant is required unless --aggregate is given")
    os.makedirs(os.path.join(OUT, "shards"), exist_ok=True)
    if a.aggregate:
        aggregate()
        return
    Q = margins()
    json.dump(CFG, open(os.path.join(OUT, "config.json"), "w"), indent=1)
    json.dump({f"{v}|{s}": dict(n=len(x), quantiles=[float(q) for q in np.quantile(x, [0, .01, .1, .25, .5, .75, .9, .99, 1])])
               for (v, s), x in Q.items()}, open(os.path.join(OUT, "margins.json"), "w"), indent=1)
    jobs = [(a.variant, c, r) for r in range(a.reps) for c in CONDS]
    with Pool(a.procs) as pool:
        pool.map(work, jobs, chunksize=1)
    aggregate()


if __name__ == "__main__":
    main()
