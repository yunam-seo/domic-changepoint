#!/usr/bin/env python
"""Pair-permutation calibration of every dependence-specific statistic, like for like.

Purpose
    Table 1 calibrates each statistic by oracle Monte Carlo (thresholds from replicates of the
    known pre-change regime). The deployed method is calibrated instead by joint
    pair permutation (Section 4.3). This runner puts every dependence-specific statistic through
    that same data-driven calibration, on the same permutations, so that level and power are
    compared like for like (Supplementary Section B.15, Table B.11).

Protocol (per replicate, identical for every statistic)
    * K = 99 joint pair permutations of the time order of the pairs (X_t, Y_t), drawn once per
      replicate from a seed fixed by (cell, replicate) and shared by ALL statistics.
    * Each statistic's candidate curve over the grid w..n-w (w = 60) is recomputed on the observed
      series and on each permuted series (rank inputs, random features and bandwidths recomputed
      on the permuted series, as run_experiment._e3_job does).
    * studentized variant ("stud"): every candidate is centered and scaled by the mean and standard
      deviation over all K+1 curves (the observed one included); T = max over the grid;
      p = (1 + #{T_k >= T_obs}) / (K + 1); tau_hat = argmax of the observed studentized curve.
    * raw variant ("raw"): T = max over the grid of the unstudentized curve, same p-value rule;
      tau_hat = argmax of the observed raw curve.
    * reject at p <= 0.05; localized hit = reject and |tau_hat - tau| <= 30.

Statistics
    DOMI-diff  random-feature DOMI difference, D = 8 (dots.domi.domi_stats, global)
    Gram-a1   Gram form, von Neumann (alpha = 1), every row of each segment
    Gram-r2   Gram form, order-2 Renyi, every row of each segment
    HSIC-diff, dCor-diff, Spearman-diff, CopulaCvM   (dots.domi.DEP_BASELINES, global)
    CvM-sub   subsample-rank copula CvM of cvm_subsample.py (job kind "cvmsub"; run with
              --worker cvmsub)
    dCor-diff is evaluated by an O(n^2) update formula for the same statistic as dots.domi.dcor_diff

Cells (n = 600, tau = 300; scenario codes D2 = S2, D4 = S4 of the article)
    null_S2    S2 generator with null=True (independence throughout)  -> level
    S2_a0.7    S2, a = 0.7                                             -> power
    S4_r0.85   S4, r = 0.85 (its pre-change regime is dependent)       -> power
    G6_b0.5    G6, b = 0.5 (non-Gaussian suite)                        -> power
    null_S3gauss  Gaussian copula with Kendall tau 0.5 throughout          -> level, dependent null
    null_S4mix    sign-mixed dependence r = 0.85 throughout                -> level, dependent null
    M1_s2.0, M2_a0.9   marginal-only changes (scale; shape)                -> specificity
    S1_r0.35, S3_tau0.5   linear correlation change; copula-shape change   -> power
    200 replicates each.

Work is split into jobs (cell, replicate, kind) with kind "light" (all statistics except the Gram
form) or "gram" (the Gram form). Each job writes one shard; workers on several hosts may run at
once (a job is claimed by creating its lock file exclusively), and --aggregate combines the shards.

Run
    python 00_SRC/run_perm_compare.py --worker light --procs 16
    python 00_SRC/run_perm_compare.py --worker gram --device cuda --helpers 3     # Gram form, GPU
    python 00_SRC/run_perm_compare.py --worker gram-cpu --procs 16                # Gram form, CPU
      (gram-cpu uses gram_gpu.gram_curves_lowrank, equal to the reference matmi_curves to ~1e-12)
    python 00_SRC/run_perm_compare.py --worker cvmsub --procs 16                  # CvM-sub
    python 00_SRC/run_perm_compare.py --aggregate
    python 00_SRC/run_perm_compare.py --status
Outputs
    04_DAOU/EXPERIMENT/perm_compare/
        shards/<cell>_<rep>_<kind>.npz   per-replicate maxima of all K+1 curves, observed curves
        records_perm.csv.gz              one row per (cell, replicate, statistic, variant):
                                         p-value, tau_hat, reject, localized hit, T_obs
        results.csv                      level / power per (cell, statistic, variant) with
                                         binomial standard errors
        summary.json                     configuration and results
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from dots.synth import SCENARIOS, generate  # noqa: E402
from dots.synth_ng import SCENARIOS_NG, generate_ng  # noqa: E402
from dots.domi import DOMIContext, domi_stats, DEP_BASELINES, ranks01  # noqa: E402
from dots.persist import save_records  # noqa: E402
from dots.perm import ge  # noqa: E402
from run_matmi_baseline import rff_gamma  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "perm_compare")
SHARDS = os.path.join(OUT, "shards")
CFG = dict(n=600, tau=300, w=60, tol=30, K=99, alpha=0.05, D=8, base_seed=20260825,
           perm_seed=20260925, reps=200)
# cell -> (scenario code, level, null)
CELLS = {
    "null_S2": ("D2", 0.7, True),
    "S2_a0.7": ("D2", 0.7, False),
    "S4_r0.85": ("D4", 0.85, False),
    "G6_b0.5": ("G6", 0.5, False),
    # dependent nulls, marginal-only controls, linear and copula-shape changes
    "null_S3gauss": ("D3", 0.5, True),     # Gaussian copula, Kendall tau 0.5, throughout (no change)
    "null_S4mix": ("D4", 0.85, True),      # sign-mixed dependence r = 0.85 throughout (no change)
    "M1_s2.0": ("M1", 2.0, False),         # marginal scale change only
    "M2_a0.9": ("M2", 0.9, False),         # marginal shape change only
    "S1_r0.35": ("D1", 0.35, False),       # Gaussian correlation 0 -> 0.35
    "S3_tau0.5": ("D3", 0.5, False),       # Gaussian -> Clayton copula at Kendall tau 0.5
}
CELL_ID = {c: i + 1 for i, c in enumerate(CELLS)}
LIGHT = ["DOMI-diff", "HSIC-diff", "dCor-diff", "Spearman-diff", "CopulaCvM"]
GRAM = ["Gram-a1", "Gram-r2"]


# ---------------------------------------------------------------- data and permutations
def sample(cell, r):
    scen, level, null = CELLS[cell]
    if scen.startswith("G"):
        li = SCENARIOS_NG[scen]["levels"].index(level)
        smp = generate_ng(scen, level, r, n=CFG["n"], tau=CFG["tau"], null=null,
                          base_seed=CFG["base_seed"], level_idx=li)
    else:
        li = SCENARIOS[scen]["levels"].index(level)
        smp = generate(scen, level, r, n=CFG["n"], tau=CFG["tau"], null=null,
                       base_seed=CFG["base_seed"], level_idx=li)
    Z = smp["Z"]
    return Z[:, smp["blocks"][0]], Z[:, smp["blocks"][1]], smp["tau"]


def permutations(cell, r):
    """K joint pair permutations for (cell, replicate); the same for every statistic."""
    rng = np.random.default_rng([CFG["perm_seed"], CELL_ID[cell], r, 11])
    return [rng.permutation(CFG["n"]) for _ in range(CFG["K"])]


def cvm_sub_fn():
    """The subsample-rank copula CvM statistic of cvm_subsample.py (copula_cvm_subsample)."""
    if not os.path.exists(os.path.join(SRC, "cvm_subsample.py")):
        return None
    mod = importlib.import_module("cvm_subsample")
    for name in ("cvm_subsample_curve", "copula_cvm_subsample", "cvm_sub"):
        if hasattr(mod, name):
            return getattr(mod, name)
    return None


# ---------------------------------------------------------------- curves
def dcor_diff_fast(ctx, mode="global"):
    """dots.domi.dcor_diff (global mode) in O(n^2) instead of O(n^3).

    Same statistic: sqrt(t(n-t)/n) |dCor[0,t) - dCor[t,n)| with distance correlation on the rank
    inputs. For a segment of size m with within-segment distance matrices a, b, row sums r^a, r^b
    and totals S^a, S^b, the double-centered inner product is
        sum_ij a~_ij b~_ij = sum_ij a_ij b_ij - (2/m) sum_i r^a_i r^b_i + S^a S^b / m^2,
    and every term is updated in O(m) when the segment grows by one index (left segments grow
    to the right, right segments to the left).
    """
    assert mode == "global"
    UX, UY = ctx.UX, ctx.UY
    n = ctx.n

    def pd(A):
        sq = (A * A).sum(1)
        return np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * A @ A.T, 0, None))

    DX, DY = pd(UX), pd(UY)

    def sweep(order):
        """dCor of the growing segment {order[0..m-1]} for m = 1..n."""
        out = np.zeros(n + 1)
        ra = np.zeros(n)
        rb = np.zeros(n)
        Sab = Saa = Sbb = Sa = Sb = 0.0
        seen = []
        for m, k in enumerate(order, start=1):
            if seen:
                idx = np.array(seen)
                a = DX[k, idx]
                b = DY[k, idx]
                Sab += 2.0 * float(a @ b) + DX[k, k] * DY[k, k]
                Saa += 2.0 * float(a @ a) + DX[k, k] ** 2
                Sbb += 2.0 * float(b @ b) + DY[k, k] ** 2
                ra[idx] += a
                rb[idx] += b
                ra[k] = a.sum() + DX[k, k]
                rb[k] = b.sum() + DY[k, k]
                Sa += 2.0 * a.sum() + DX[k, k]
                Sb += 2.0 * b.sum() + DY[k, k]
            else:
                Sab, Saa, Sbb = DX[k, k] * DY[k, k], DX[k, k] ** 2, DY[k, k] ** 2
                ra[k], rb[k], Sa, Sb = DX[k, k], DY[k, k], DX[k, k], DY[k, k]
            seen.append(k)
            idx = np.array(seen)
            m2 = float(m * m)
            dxy = (Sab - 2.0 / m * float(ra[idx] @ rb[idx]) + Sa * Sb / m2) / m2
            dxx = (Saa - 2.0 / m * float(ra[idx] @ ra[idx]) + Sa * Sa / m2) / m2
            dyy = (Sbb - 2.0 / m * float(rb[idx] @ rb[idx]) + Sb * Sb / m2) / m2
            out[m] = np.sqrt(max(dxy, 0) / np.sqrt(dxx * dyy + 1e-18))
        return out

    L = sweep(range(n))                    # L[t]   = dCor of [0, t)
    R = sweep(range(n - 1, -1, -1))        # R[n-t] = dCor of [t, n)
    grid = ctx.grid
    return np.array([np.sqrt(t * (n - t) / n) * abs(L[t] - R[n - t]) for t in grid])


def light_curves(X, Y):
    ctx = DOMIContext(X, Y, CFG["w"], D=CFG["D"], seed=2026, n_perm=0)
    out = {"DOMI-diff": domi_stats(ctx, "global")["DOMI-diff"]}
    for k in LIGHT[1:]:
        out[k] = dcor_diff_fast(ctx) if k == "dCor-diff" else DEP_BASELINES[k](ctx, "global")
    return out


def cvmsub_curves(X, Y, sub):
    ctx = DOMIContext(X, Y, CFG["w"], D=CFG["D"], seed=2026, n_perm=0)
    return {"CvM-sub": np.asarray(sub(ctx, "global"), float)}


def summarise(curves, grid):
    """curves: (K+1, G), row 0 observed -> per-variant maxima of all curves and observed argmax."""
    A = np.asarray(curves, float)
    mu, sd = A.mean(0), A.std(0) + 1e-12
    Z = (A - mu) / sd
    return dict(max_raw=A.max(1), max_stud=Z.max(1),
                tau_raw=int(grid[int(np.argmax(A[0]))]), tau_stud=int(grid[int(np.argmax(Z[0]))]),
                obs_raw=A[0], obs_stud=Z[0])


def _save_shard(cell, r, kind, summ):
    flat = {}
    for stat, d in summ.items():
        for k, v in d.items():
            flat[f"{stat}::{k}"] = np.asarray(v)
    tmp = os.path.join(SHARDS, f".{cell}_{r:03d}_{kind}.tmp.npz")
    np.savez_compressed(tmp, **flat)
    os.replace(tmp, os.path.join(SHARDS, f"{cell}_{r:03d}_{kind}.npz"))


def light_job(a, kind="light"):
    cell, r = a
    X, Y, _ = sample(cell, r)
    if kind == "light":
        f = light_curves
    else:
        sub = cvm_sub_fn()
        f = lambda x, y: cvmsub_curves(x, y, sub)  # noqa: E731
    per = [f(X, Y)]
    for idx in permutations(cell, r):
        per.append(f(X[idx], Y[idx]))
    grid = np.arange(CFG["w"], CFG["n"] - CFG["w"] + 1)
    summ = {k: summarise(np.array([p[k] for p in per]), grid) for k in per[0]}
    _save_shard(cell, r, kind, summ)
    return cell, r


def cvmsub_job(a):
    return light_job(a, "cvmsub")


def gram_inputs(cell, r):
    X, Y, _ = sample(cell, r)
    UX, UY = ranks01(X), ranks01(Y)
    ux, uy, gx, gy = [UX[:, 0]], [UY[:, 0]], [rff_gamma(UX, 2026)], [rff_gamma(UY, 2027)]
    for idx in permutations(cell, r):
        ux.append(UX[idx, 0])
        uy.append(UY[idx, 0])
        gx.append(rff_gamma(UX[idx], 2026))   # bandwidth recomputed on the permuted series
        gy.append(rff_gamma(UY[idx], 2027))
    return np.array(ux), np.array(uy), np.array(gx), np.array(gy)


def _gram_finish(cell, r, a1, r2):
    grid = np.arange(CFG["w"], CFG["n"] - CFG["w"] + 1)
    _save_shard(cell, r, "gram", {"Gram-a1": summarise(a1, grid), "Gram-r2": summarise(r2, grid)})


def gram_cpu_job(a):
    from gram_gpu import gram_curves_lowrank
    cell, r = a
    ux, uy, gx, gy = gram_inputs(cell, r)
    a1, r2 = gram_curves_lowrank(ux, uy, gx, gy, CFG["w"])
    _gram_finish(cell, r, a1, r2)
    return cell, r


# ---------------------------------------------------------------- job bookkeeping
def all_jobs(kind):
    # the job list is the same for every kind
    return [(c, r) for r in range(CFG["reps"]) for c in CELLS]   # replicate-major: cells fill evenly


def _done(cell, r, kind):
    return os.path.exists(os.path.join(SHARDS, f"{cell}_{r:03d}_{kind}.npz"))


def _claim(cell, r, kind):
    """Claim a job by exclusive creation of its lock file; False if done or claimed elsewhere."""
    if _done(cell, r, kind):
        return False
    try:
        fd = os.open(os.path.join(SHARDS, f"{cell}_{r:03d}_{kind}.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return False
    if _done(cell, r, kind):              # finished between the first check and the lock
        _release(cell, r, kind)
        return False
    return True


def _release(cell, r, kind):
    try:
        os.remove(os.path.join(SHARDS, f"{cell}_{r:03d}_{kind}.lock"))
    except FileNotFoundError:
        pass


def _claimed_iter(kind):
    for c, r in all_jobs(kind):
        if _claim(c, r, kind):
            yield c, r


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(os.path.join(OUT, "run.log"), "a") as f:
        f.write(line + "\n")


def _loop(a):
    """One pool process: claim the next free job of `kind`, run it, repeat until none is left.
    (Claiming inside the process keeps claims lazy, so several hosts can share the job list.)"""
    kind, fn_name = a
    fn = globals()[fn_name]
    n = 0
    for c, r in _claimed_iter(kind):
        t1 = time.time()
        try:
            fn((c, r))
        finally:
            _release(c, r, kind)
        n += 1
        say(f"{kind} {c} rep {r}: {time.time()-t1:.0f}s")
    return n


def _pool_worker(kind, fn_name, procs):
    t0 = time.time()
    with Pool(procs) as pool:
        done = sum(pool.map(_loop, [(kind, fn_name)] * procs, chunksize=1))
    say(f"{kind}: worker finished, {done} jobs, {time.time()-t0:.0f}s")


def _gpu_worker(device, helpers):
    import torch
    from gram_gpu import gram_curves_gpu
    t0 = time.time()
    done = 0
    with Pool(helpers) as pool:
        for c, r in _claimed_iter("gram"):
            t1 = time.time()
            ux, uy, gx, gy = gram_inputs(c, r)
            a1, r2 = gram_curves_gpu(ux, uy, gx, gy, CFG["w"], device=device, pool=pool)
            torch.cuda.empty_cache()
            _gram_finish(c, r, a1, r2)
            _release(c, r, "gram")
            done += 1
            say(f"gram[{device}] {c} rep {r}: {time.time()-t1:.0f}s ({done} jobs, {time.time()-t0:.0f}s)")
    say(f"gram[{device}]: worker finished, {done} jobs, {time.time()-t0:.0f}s")


# ---------------------------------------------------------------- aggregation
def _load(cell, r, kind):
    z = np.load(os.path.join(SHARDS, f"{cell}_{r:03d}_{kind}.npz"))
    out = {}
    for key in z.files:
        stat, f = key.split("::")
        out.setdefault(stat, {})[f] = z[key]
    return out


def aggregate():
    K, alpha, tol = CFG["K"], CFG["alpha"], CFG["tol"]
    recs, rows = [], []
    missing = []
    for cell, (scen, level, null) in CELLS.items():
        per = {}
        for r in range(CFG["reps"]):
            d = {}
            for kind in ("light", "gram", "cvmsub"):
                if kind == "cvmsub" and not HAS_CVMSUB_SHARDS():
                    continue
                if _done(cell, r, kind):
                    d.update(_load(cell, r, kind))
                else:
                    missing.append((cell, r, kind))
            for stat, s in d.items():
                for var in ("raw", "stud"):
                    M = s[f"max_{var}"]
                    p = (1.0 + float(np.sum(ge(M[1:], M[0])))) / (K + 1.0)
                    th = int(s[f"tau_{var}"])
                    rej = bool(p <= alpha)
                    hit = bool(rej and abs(th - CFG["tau"]) <= tol) if not null else None
                    rec = dict(cell=cell, scen=scen, level=level, null=int(null), rep=r, stat=stat,
                               variant=var, T_obs=float(M[0]), p=p, tau_hat=th, reject=int(rej),
                               hit="" if hit is None else int(hit))
                    recs.append(rec)
                    per.setdefault((stat, var), []).append(rec)
        for (stat, var), rs in per.items():
            nrep = len(rs)
            rej = np.mean([x["reject"] for x in rs])
            row = dict(cell=cell, scen=scen, level=level, null=int(null), stat=stat, variant=var,
                       n_reps=nrep, reject_rate=round(float(rej), 4),
                       reject_se=round(float(np.sqrt(rej * (1 - rej) / nrep)), 4))
            if not null:
                hit = np.mean([x["hit"] for x in rs])
                err = [abs(x["tau_hat"] - CFG["tau"]) for x in rs if x["reject"]]
                row.update(power_localised=round(float(hit), 4),
                           power_localised_se=round(float(np.sqrt(hit * (1 - hit) / nrep)), 4),
                           median_abs_err_rejected=float(np.median(err)) if err else "")
            rows.append(row)
    path = os.path.join(OUT, "records_perm.csv.gz")
    if os.path.exists(path):
        os.remove(path)                       # rebuilt from the shards on every aggregation
    save_records(OUT, "records_perm.csv", recs)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(config=CFG, cells={c: dict(scen=s, level=l, null=n) for c, (s, l, n) in CELLS.items()},
                   statistics=LIGHT + GRAM + (["CvM-sub"] if any(r["stat"] == "CvM-sub" for r in rows) else []),
                   note=("studentised variant: per-candidate mean/sd over all K+1 curves incl. the observed; "
                         "p = (1+#{T_k>=T_obs})/(K+1); reject at p<=0.05; localised hit: reject and "
                         "|tau_hat-tau|<=30; Gram form uses every row of each segment; scenario codes "
                         "D2 = S2, D4 = S4"),
                   missing_shards=len(missing), rows=rows),
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(f"aggregated: {len(rows)} rows, {len(missing)} missing shards")


def HAS_CVMSUB_SHARDS():
    return any(f.endswith("_cvmsub.npz") for f in os.listdir(SHARDS))


def status():
    for kind in ("light", "gram", "cvmsub"):
        n = sum(_done(c, r, kind) for c, r in all_jobs(kind))
        locks = len([f for f in os.listdir(SHARDS) if f.endswith(f"_{kind}.lock")])
        print(f"{kind}: {n}/{len(all_jobs(kind))} done, {locks} running")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["light", "gram", "gram-cpu", "cvmsub"])
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--helpers", type=int, default=3)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--clear-locks", action="store_true", help="remove stale locks (no worker running)")
    ap.add_argument("--reps", type=int, default=CFG["reps"])
    a = ap.parse_args()
    CFG["reps"] = a.reps
    os.makedirs(SHARDS, exist_ok=True)
    if a.clear_locks:
        for f in os.listdir(SHARDS):
            if f.endswith(".lock"):
                os.remove(os.path.join(SHARDS, f))
    if a.status:
        return status()
    if a.aggregate:
        return aggregate()
    if a.worker == "light":
        _pool_worker("light", "light_job", a.procs)
    elif a.worker == "gram-cpu":
        _pool_worker("gram", "gram_cpu_job", a.procs)
    elif a.worker == "cvmsub":
        if cvm_sub_fn() is None:
            raise SystemExit("00_SRC/cvm_subsample.py (copula_cvm_subsample) not found")
        _pool_worker("cvmsub", "cvmsub_job", a.procs)
    elif a.worker == "gram":
        _gpu_worker(a.device, a.helpers)


if __name__ == "__main__":
    main()
