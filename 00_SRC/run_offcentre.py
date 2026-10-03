#!/usr/bin/env python
"""Off-center breaks under the Monte Carlo calibration of Table 1.

Purpose
    Every synthetic power in Table 1 has its break at the midpoint (tau = n/2). This runner
    measures how the dependence-specific statistics detect and localize a break at tau/n = 0.25
    or 0.75 under the same calibration (Supplementary Section B.15, Table B.12). tau = 300 is run in
    the same way as the center reference (with the seeds of Table 1, so its alternatives are the
    Table 1 replicates).

Protocol (Table 1, unchanged except for the break location)
    n = 600, candidate window w = 60 (grid 60..540), breaks tau in {150, 300, 450}.
    500 null replicates of the pre-change regime held throughout: for S2, G4, G6 the independence
    null (drawn at the scenario's first level index, as in Table 1); for S4 the matched null (the
    dependent pre-change regime at the alternative's own r, as run_e1_matched_null.py does).
    Raw statistic ("|g"): threshold = 95% quantile of the max over the grid, all 500 nulls.
    Studentized statistic ("|gs"): per-candidate mean/sd from null replicates 0..249, threshold
    = 95% quantile of the max of the studentized curve over null replicates 250..499.
    500 alternative replicates per tau; the same null calibration serves every tau of a cell.
    detect_rate = P(max > threshold); power (localized) = P(detect and |tau_hat - tau| <= 30);
    tau_hat = argmax of the (raw or studentized) observed curve.

Statistics (global mode, rank inputs, as run_matmi_baseline / run_ng_suite)
    DOMI-diff   random-feature DOMI difference, D = 8
    matmi-a1   Gram form, von Neumann, every row of each segment (gram_gpu.gram_curves_gpu on a
               CUDA device, or gram_gpu.gram_curves_lowrank on the CPU with --worker gram-cpu)
    HSIC-diff, dCor-diff (O(n^2) form, equal to dots.domi.dcor_diff), Spearman-diff, CopulaCvM

Cells (scenario codes D2 = S2, D4 = S4 of the article)
    S2_a0.9 (D2, a = 0.9), S4_r0.85 (D4, r = 0.85, matched null), G4_a0.5, G6_b0.5

Run
    python 00_SRC/run_offcentre.py --worker light --procs 4        # all statistics but the Gram form
    python 00_SRC/run_offcentre.py --worker gram --device cuda --helpers 3
    python 00_SRC/run_offcentre.py --worker gram-cpu --procs 4     # Gram form on the CPU instead
      (gram-cpu uses gram_gpu.gram_curves_lowrank, equal to the reference matmi_curves to ~1e-12)
    python 00_SRC/run_offcentre.py --evaluate
    python 00_SRC/run_offcentre.py --status
Outputs
    04_DAOU/EXPERIMENT/offcenter/
        curves/<cell>_<set>_<chunk>_<kind>.npz   candidate curves (set = null, tau150, tau300, tau450)
        records_null.csv.gz                      per null replicate: max of each curve
        records_alt.csv.gz                       per alternative replicate and statistic: max,
                                                 tau_hat, detected, localized hit, |tau_hat - tau|
        results.csv                              per (cell, tau, statistic): detect_rate, power,
                                                 median |tau_hat - tau|, standard errors, threshold
        summary.json                             configuration, thresholds and results
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
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from dots.synth import SCENARIOS, generate  # noqa: E402
from dots.synth_ng import SCENARIOS_NG, generate_ng  # noqa: E402
from dots.domi import DOMIContext, domi_stats, DEP_BASELINES  # noqa: E402
from dots.extras import studentize  # noqa: E402
from dots.evaluate import summarize_alt, aggregate  # noqa: E402
from dots.persist import save_records  # noqa: E402
from run_matmi_baseline import rff_gamma  # noqa: E402
from run_perm_compare import dcor_diff_fast  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "offcentre")
CURVES = os.path.join(OUT, "curves")
CFG = dict(n=600, w=60, tol=30, D=8, fpr=0.05, base_seed=20260825, reps=500, chunk=50,
           taus=[150, 300, 450])
# cell -> (scenario code, level, matched null)
CELLS = {
    "S2_a0.9": ("D2", 0.9, False),
    "S4_r0.85": ("D4", 0.85, True),
    "G4_a0.5": ("G4", 0.5, False),
    "G6_b0.5": ("G6", 0.5, False),
}
LIGHT = ["DOMI-diff", "HSIC-diff", "dCor-diff", "Spearman-diff", "CopulaCvM"]
GRAM = ["matmi-a1"]


def _levels(scen):
    return (SCENARIOS_NG if scen.startswith("G") else SCENARIOS)[scen]["levels"]


def sample(cell, setname, r):
    """setname 'null' (pre-change regime throughout) or 'tau<k>' (break at k)."""
    scen, level, matched = CELLS[cell]
    gen = generate_ng if scen.startswith("G") else generate
    li = _levels(scen).index(level)
    if setname == "null":
        lin = li if matched else 0
        smp = gen(scen, _levels(scen)[lin], r, n=CFG["n"], null=True, base_seed=CFG["base_seed"],
                  level_idx=lin)
    else:
        tau = int(setname[3:])
        smp = gen(scen, level, r, n=CFG["n"], tau=tau, null=False, base_seed=CFG["base_seed"],
                  level_idx=li)
    Z = smp["Z"]
    return Z[:, smp["blocks"][0]], Z[:, smp["blocks"][1]]


def sets():
    return ["null"] + [f"tau{t}" for t in CFG["taus"]]


def jobs():
    nch = CFG["reps"] // CFG["chunk"]
    return [(c, s, k) for k in range(nch) for c in CELLS for s in sets()]


def _path(c, s, k, kind, ext="npz"):
    return os.path.join(CURVES, f"{c}_{s}_{k:02d}_{kind}.{ext}")


# ---------------------------------------------------------------- curve jobs
def light_job(job):
    c, s, k = job
    out = {m: [] for m in LIGHT}
    for r in range(k * CFG["chunk"], (k + 1) * CFG["chunk"]):
        X, Y = sample(c, s, r)
        ctx = DOMIContext(X, Y, CFG["w"], D=CFG["D"], seed=2026, n_perm=0)
        out["DOMI-diff"].append(domi_stats(ctx, "global")["DOMI-diff"])
        for m in LIGHT[1:]:
            out[m].append(dcor_diff_fast(ctx) if m == "dCor-diff" else DEP_BASELINES[m](ctx, "global"))
    _save(c, s, k, "light", {m: np.array(v) for m, v in out.items()})


def gram_inputs(c, s, k):
    ux, uy, gx, gy = [], [], [], []
    for r in range(k * CFG["chunk"], (k + 1) * CFG["chunk"]):
        X, Y = sample(c, s, r)
        ctx = DOMIContext(X, Y, CFG["w"], D=CFG["D"], seed=2026, n_perm=0)
        ux.append(ctx.UX[:, 0]); uy.append(ctx.UY[:, 0])
        gx.append(rff_gamma(ctx.UX, 2026)); gy.append(rff_gamma(ctx.UY, 2027))
    return np.array(ux), np.array(uy), np.array(gx), np.array(gy)


def gram_cpu_job(job):
    from gram_gpu import gram_curves_lowrank
    c, s, k = job
    a1, _ = gram_curves_lowrank(*gram_inputs(c, s, k), CFG["w"])
    _save(c, s, k, "gram", {"matmi-a1": a1})


def _save(c, s, k, kind, arrs):
    tmp = os.path.join(CURVES, f".{c}_{s}_{k:02d}_{kind}.tmp.npz")
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, _path(c, s, k, kind))


def _claim(job, kind):
    if os.path.exists(_path(*job, kind)):
        return False
    try:
        os.close(os.open(_path(*job, kind, "lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False
    if os.path.exists(_path(*job, kind)):   # finished between the first check and the lock
        _release(job, kind)
        return False
    return True


def _release(job, kind):
    try:
        os.remove(_path(*job, kind, "lock"))
    except FileNotFoundError:
        pass


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(os.path.join(OUT, "run.log"), "a") as f:
        f.write(line + "\n")


def _loop(a):
    kind, fn_name = a
    fn = globals()[fn_name]
    n = 0
    for job in jobs():
        if not _claim(job, kind):
            continue
        t1 = time.time()
        try:
            fn(job)
        finally:
            _release(job, kind)
        n += 1
        say(f"{kind} {job}: {time.time()-t1:.0f}s")
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
    n = 0
    with Pool(helpers) as pool:
        for job in jobs():
            if not _claim(job, "gram"):
                continue
            t1 = time.time()
            try:
                a1, _ = gram_curves_gpu(*gram_inputs(*job), CFG["w"], device=device, pool=pool)
                torch.cuda.empty_cache()
                _save(*job, "gram", {"matmi-a1": a1})
            finally:
                _release(job, "gram")
            n += 1
            say(f"gram[{device}] {job}: {time.time()-t1:.0f}s")
    say(f"gram[{device}]: worker finished, {n} jobs, {time.time()-t0:.0f}s")


# ---------------------------------------------------------------- evaluation
def _load_set(c, s):
    """All curves of (cell, set): stat -> (reps, G); a stat is present only if every chunk has it."""
    nch = CFG["reps"] // CFG["chunk"]
    out = {}
    for kind in ("light", "gram"):
        parts = [_path(c, s, k, kind) for k in range(nch)]
        if not all(os.path.exists(p) for p in parts):
            continue
        zs = [np.load(p) for p in parts]
        for m in zs[0].files:
            out[m] = np.concatenate([z[m] for z in zs], 0)
    return out


def _median_se(x, B=1000, seed=20260925):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    return float(np.std([np.median(rng.choice(x, len(x))) for _ in range(B)]))


def evaluate():
    n, w, tol, fpr = CFG["n"], CFG["w"], CFG["tol"], CFG["fpr"]
    grid = np.arange(w, n - w + 1)
    half = CFG["reps"] // 2
    rows, thresholds = [], {}
    for p in ("records_null.csv.gz", "records_alt.csv.gz"):
        if os.path.exists(os.path.join(OUT, p)):
            os.remove(os.path.join(OUT, p))           # rebuilt from the curves on every evaluation
    for c, (scen, level, matched) in CELLS.items():
        nul = _load_set(c, "null")
        if not nul:
            continue
        stats = list(nul)
        mu0 = {m: nul[m][:half].mean(0) for m in stats}
        sd0 = {m: nul[m][:half].std(0) for m in stats}
        thr = {}
        for m in stats:
            thr[f"{m}|g"] = float(np.quantile(nul[m].max(1), 1 - fpr))
            thr[f"{m}|gs"] = float(np.quantile([np.max(studentize(v, mu0[m], sd0[m])) for v in nul[m][half:]],
                                               1 - fpr))
        thresholds[c] = thr
        save_records(OUT, "records_null.csv",
                     [{f"max|{m}": float(nul[m][r].max()) for m in stats} for r in range(CFG["reps"])],
                     {"cell": c, "scen": scen, "level": level, "matched_null": int(matched)})
        for tau in CFG["taus"]:
            alt = _load_set(c, f"tau{tau}")
            ms = [m for m in stats if m in alt]
            if not ms:
                continue
            recs = []
            for r in range(CFG["reps"]):
                st = {}
                for m in ms:
                    st[f"{m}|g"] = alt[m][r]
                    st[f"{m}|gs"] = studentize(alt[m][r], mu0[m], sd0[m])
                rec = summarize_alt(st, grid, tau, w, n, thr, tol)
                for k in rec:
                    rec[k]["abs_err"] = abs(rec[k]["tau_hat"] - tau)
                recs.append(rec)
            save_records(OUT, "records_alt.csv", recs, {"cell": c, "scen": scen, "level": level, "tau": tau})
            agg = aggregate(recs, thr)
            R = len(recs)
            for k, v in agg.items():
                m, mode = k.split("|")
                err_all = [x[k]["abs_err"] for x in recs]
                err_det = [x[k]["abs_err"] for x in recs if x[k]["detected"]]
                rows.append(dict(
                    cell=c, scen=scen, level=level, tau=tau, tau_over_n=tau / n, stat=m,
                    variant="studentised" if mode == "gs" else "raw", key=k,
                    detect_rate=round(v["detect_rate"], 4),
                    detect_se=round(float(np.sqrt(v["detect_rate"] * (1 - v["detect_rate"]) / R)), 4),
                    power_localised=round(v["power"], 4),
                    power_se=round(float(np.sqrt(v["power"] * (1 - v["power"]) / R)), 4),
                    median_abs_err_all=float(np.median(err_all)),
                    median_abs_err_all_se=round(_median_se(err_all), 2),
                    median_abs_err_detected=float(np.median(err_det)) if err_det else "",
                    median_abs_err_detected_se=round(_median_se(err_det), 2) if err_det else "",
                    threshold=v["threshold"], n_reps=R))
        say(f"evaluated {c}")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys, restval="")
        wr.writeheader()
        wr.writerows(rows)
    json.dump(dict(config=CFG, cells={c: dict(scen=s, level=l, matched_null=m) for c, (s, l, m) in CELLS.items()},
                   note=("Table 1 protocol with the break moved; raw threshold from all 500 nulls, studentised "
                         "moments from nulls 0..249 and threshold from nulls 250..499; power = detect and "
                         "|tau_hat - tau| <= 30; median |tau_hat - tau| over all replicates and over detected "
                         "ones, bootstrap SE (1000 resamples); Gram form (matmi-a1) with every row of each "
                         "segment; scenario codes D2 = S2, D4 = S4"),
                   thresholds=thresholds, rows=rows),
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(f"evaluated: {len(rows)} rows")


def status():
    for kind in ("light", "gram"):
        done = sum(os.path.exists(_path(*j, kind)) for j in jobs())
        locks = sum(os.path.exists(_path(*j, kind, "lock")) for j in jobs())
        print(f"{kind}: {done}/{len(jobs())} chunks done, {locks} running")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["light", "gram", "gram-cpu"])
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--helpers", type=int, default=3)
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--clear-locks", action="store_true", help="remove stale locks (no worker running)")
    a = ap.parse_args()
    os.makedirs(CURVES, exist_ok=True)
    if a.clear_locks:
        for f in os.listdir(CURVES):
            if f.endswith(".lock"):
                os.remove(os.path.join(CURVES, f))
    if a.status:
        return status()
    if a.evaluate:
        return evaluate()
    if a.worker == "light":
        _pool_worker("light", "light_job", a.procs)
    elif a.worker == "gram-cpu":
        _pool_worker("gram", "gram_cpu_job", a.procs)
    elif a.worker == "gram":
        _gpu_worker(a.device, a.helpers)


if __name__ == "__main__":
    main()
