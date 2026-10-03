#!/usr/bin/env python
"""Test-computation time of every dependence-specific statistic on the same task.

Timed: the pair-permutation procedure of Section 4.3 applied to one series, from ranks, bandwidths and
features through the observed candidate curve, the curves of K = 99 joint pair permutations (features and
bandwidths recomputed on each permuted series, as in run_perm_compare.py), per-candidate studentization
over all K + 1 curves, the maximum and the permutation p-value. Not timed: data generation, generation of
the permutation indices, and process start-up and imports.

Each (statistic, n, dataset) runs in a fresh process, one thread (BLAS pinned), so that the peak
resident memory of the process (ru_maxrss) belongs to that statistic alone. A test that does not finish
within the time limit is recorded as a timeout, not extrapolated.

Task: S2 (code D2) at a = 0.7 with the break at n/2, K = 99, datasets r = 0..9 (the generator seeds of
Table 1). Two candidate grids:
  dense  every split w..n-w with w = n/10 (the synthetic experiments), n in {600, 2400, 9600};
  fixed  161 splits at the fractions 0.100, 0.105, ..., 0.900 (the setting of long records, where
         candidates sit on a grid, as in Section 6.7), n in {600, 2400, 9600, 38400, 76800}.
         DOMI is then computed as in the hourly analysis: features once, moment sums per interval
         between consecutive candidates, prefix sums over intervals (dots/hourly.py).
Each test process is limited to 16 GB of address space; exceeding it is recorded as "memory". The
sizes run in increasing order; once a statistic times out or exceeds the limit at some n, its replicates
at that n that have not started, and all its runs at larger n, are recorded as "not run".

Statistics: DOMI (random-feature form, D = 8), Gram (Gram form at alpha = 1, every row of each
segment; the timed routine also evaluates the order-2 curve from the same eigenvalues; CPU low-rank
evaluation, and on a GPU when --device cuda is given), HSIC-diff, dCor-diff,
Spearman-diff, CopulaCvM (global pseudo-observations), CvM-sub (ranks within each segment).

Run:   OMP_NUM_THREADS=1 python 00_SRC/run_cost_compare.py --grid dense --procs 4
       OMP_NUM_THREADS=1 python 00_SRC/run_cost_compare.py --grid fixed --procs 4
       python 00_SRC/run_cost_compare.py --device cuda --only Gram --procs 1     # GPU timings
       python 00_SRC/run_cost_compare.py --summarize
Writes 04_DAOU/EXPERIMENT/cost_compare/{config.json, records.csv, summary.json}
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "cost_compare")
CFG = dict(scenario="D2", level=0.7, K=99, D=8, base_seed=20260825, perm_seed=20260926, datasets=10,
           ns={"dense": [600, 2400, 9600], "fixed": [600, 2400, 9600, 38400, 76800]},
           fixed_fractions=[round(0.1 + 0.005 * i, 3) for i in range(161)], timeout_s=1800,
           memory_limit_gb=16)
METHODS = ["DOMI", "HSIC-diff", "dCor-diff", "Spearman-diff", "CopulaCvM", "CvM-sub", "Gram"]
FIELDS = ["grid", "method", "device", "n", "dataset", "status", "seconds", "peak_rss_mb", "p_value", "tau_hat",
          "candidates", "K"]


def one(method, n, r, device, gridkind="dense"):
    """Run one test in this process and return its record (called in a fresh subprocess)."""
    import resource
    lim = CFG["memory_limit_gb"] * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    from dots.synth import SCENARIOS, generate
    from dots.domi import DOMIContext, domi_stats, DEP_BASELINES, ranks01
    from dots.perm import ge
    import run_perm_compare as PC
    li = SCENARIOS[CFG["scenario"]]["levels"].index(CFG["level"])
    smp = generate(CFG["scenario"], CFG["level"], r, n=n, tau=n // 2, null=False,
                   base_seed=CFG["base_seed"], level_idx=li)
    X, Y = smp["Z"][:, smp["blocks"][0]], smp["Z"][:, smp["blocks"][1]]
    w = n // 10
    rng = np.random.default_rng([CFG["perm_seed"], n, r])
    perms = [rng.permutation(n) for _ in range(CFG["K"])]
    if gridkind == "fixed":
        grid = np.unique(np.round(np.array(CFG["fixed_fractions"]) * n).astype(int))
    else:
        grid = np.arange(w, n - w + 1)
    t0 = time.perf_counter()
    if method == "DOMI" and gridkind == "fixed":
        from dots.hourly import features, block_moments, prefix, seg_domi
        edges = np.concatenate([[0], grid, [n]])
        B = len(edges) - 1

        def curve_fixed(x, y):
            P = prefix(*block_moments(*features(x, y, D=CFG["D"]), edges))
            out = np.empty(len(grid))
            for k in range(len(grid)):
                t = grid[k]
                out[k] = np.sqrt(t * (n - t) / n) * abs(seg_domi(P, 0, k + 1) - seg_domi(P, k + 1, B))
            return out
        A = np.array([curve_fixed(X[:, 0], Y[:, 0])] + [curve_fixed(X[i, 0], Y[i, 0]) for i in perms], float)
    elif method == "Gram":
        from run_matmi_baseline import rff_gamma
        UX, UY = ranks01(X), ranks01(Y)
        ux, uy = [UX[:, 0]] + [UX[i, 0] for i in perms], [UY[:, 0]] + [UY[i, 0] for i in perms]
        gx = [rff_gamma(UX, 2026)] + [rff_gamma(UX[i], 2026) for i in perms]
        gy = [rff_gamma(UY, 2027)] + [rff_gamma(UY[i], 2027) for i in perms]
        if device == "cuda":
            from gram_gpu import gram_curves_gpu
            A, _ = gram_curves_gpu(np.array(ux), np.array(uy), np.array(gx), np.array(gy), w, device="cuda")
        else:
            from gram_gpu import gram_curves_lowrank
            A, _ = gram_curves_lowrank(np.array(ux), np.array(uy), np.array(gx), np.array(gy), w)
        A = np.asarray(A, float)
        if gridkind == "fixed":          # the Gram routines return every split w..n-w; keep the candidates
            A = A[:, grid - w]
    else:
        from types import SimpleNamespace
        from dots.domi import unit_rff

        class _First:
            """Prefix sums of first moments only (all HSIC-diff reads)."""
            def __init__(self, F):
                self.S1 = np.vstack([np.zeros((1, F.shape[1])), np.cumsum(F, 0)])

            def mean(self, a, b):
                return (self.S1[b] - self.S1[a]) / (b - a)

        def context(x, y):
            # each statistic builds only the inputs it reads: DOMI the full feature moments, HSIC the
            # first moments of the same features, the rank statistics the ranks alone
            if method == "DOMI":
                return DOMIContext(x, y, w, D=CFG["D"], seed=2026, n_perm=0)
            UX, UY = ranks01(x), ranks01(y)
            c = SimpleNamespace(UX=UX, UY=UY, n=len(x), w=w, grid=grid, D=CFG["D"])
            if method == "HSIC-diff":
                FX, FY = unit_rff(UX, CFG["D"], 2026), unit_rff(UY, CFG["D"], 2027)
                c.cx, c.cy = _First(FX), _First(FY)
                c.cj = _First(np.einsum("ti,tj->tij", FX, FY).reshape(len(x), -1))
            return c

        def curve(x, y):
            ctx = context(x, y)
            ctx.grid = grid
            if method == "DOMI":
                return domi_stats(ctx, "global")["DOMI-diff"]
            if method == "dCor-diff":
                return PC.dcor_diff_fast(ctx)
            if method == "CvM-sub":
                return np.asarray(PC.cvm_sub_fn()(ctx, "global"), float)
            return DEP_BASELINES[method](ctx, "global")
        A = np.array([curve(X, Y)] + [curve(X[i], Y[i]) for i in perms], float)
    mu, sd = A.mean(0), A.std(0) + 1e-12
    Z = (A - mu) / sd
    T = Z.max(1)
    p = (1 + int(np.sum(ge(T[1:], T[0])))) / (CFG["K"] + 1)
    secs = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return dict(grid=gridkind, method=method, device=device, n=n, dataset=r, status="ok", seconds=round(secs, 3),
                peak_rss_mb=round(rss, 1), p_value=p, tau_hat=int(grid[int(np.argmax(Z[0]))]),
                candidates=len(grid), K=CFG["K"])


def run_job(job):
    gridkind, method, n, r, device = job
    base = dict(grid=gridkind, method=method, device=device, n=n, dataset=r)
    cmd = [sys.executable, os.path.abspath(__file__), "--one", method, str(n), str(r), device, gridkind]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=CFG["timeout_s"])
        if out.returncode != 0:
            err = out.stderr.strip()
            return dict(base, status="memory" if "MemoryError" in err or "Unable to allocate" in err
                        else "error: " + err[-200:])
        return json.loads(out.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return dict(base, status="timeout", seconds=CFG["timeout_s"])


def load():
    p = os.path.join(OUT, "records.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def summarise():
    rows = load()
    summ = {}
    for r in rows:
        k = f"{r.get('grid', 'dense')}|{r['method']}|{r['device']}|{r['n']}"
        summ.setdefault(k, {"ok": [], "rss": [], "timeouts": 0, "memory": 0, "errors": 0})
        if r["status"] == "ok":
            summ[k]["ok"].append(float(r["seconds"]))
            summ[k]["rss"].append(float(r["peak_rss_mb"]))
        elif r["status"] == "timeout":
            summ[k]["timeouts"] += 1
        elif r["status"] == "memory":
            summ[k]["memory"] += 1
        elif r["status"] == "not run":
            summ[k].setdefault("not_run", 0)
            summ[k]["not_run"] += 1
        else:
            summ[k]["errors"] += 1
    out = {k: dict(completed=len(v["ok"]), timeouts=v["timeouts"], memory=v["memory"], errors=v["errors"],
                   median_seconds=float(np.median(v["ok"])) if v["ok"] else None,
                   max_seconds=float(np.max(v["ok"])) if v["ok"] else None,
                   median_peak_rss_mb=float(np.median(v["rss"])) if v["rss"] else None)
           for k, v in sorted(summ.items())}
    json.dump(dict(config=CFG, results=out), open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    for k, v in out.items():
        print(f"{k:32s} done {v['completed']:2d} timeout {v['timeouts']:2d} err {v['errors']}  "
              f"median {v['median_seconds']} s  rss {v['median_peak_rss_mb']} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=5, metavar=("METHOD", "N", "R", "DEVICE", "GRID"))
    ap.add_argument("--grid", default="dense", choices=["dense", "fixed"])
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--only", default="", help="comma-separated methods")
    ap.add_argument("--ns", default="", help="comma-separated n (default: all)")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.one:
        m, n, r, dev, g = a.one
        print(json.dumps(one(m, int(n), int(r), dev, g)))
        return
    os.makedirs(OUT, exist_ok=True)
    if a.summarise:
        summarise()
        return
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip() if a.device == "cuda" else ""
    host = dict(machine=platform.node(), cpu=cpu, gpu=gpu, python=platform.python_version(),
                numpy=np.__version__, procs=a.procs, threads_per_test=1, memory_limit_gb=CFG["memory_limit_gb"],
                timeout_s=CFG["timeout_s"])
    cp = os.path.join(OUT, "config.json")
    prev = json.load(open(cp)) if os.path.exists(cp) else {}
    hosts = prev.get("runs", {})
    hosts[f"{a.grid}|{a.device}"] = host            # one entry per (grid, device) invocation
    json.dump(dict(CFG, runs=hosts), open(cp, "w"), indent=1)
    methods = [m for m in METHODS if not a.only or m in a.only.split(",")]
    ns = [int(x) for x in a.ns.split(",")] if a.ns else CFG["ns"][a.grid]
    done = {(r.get("grid", "dense"), r["method"], r["device"], int(r["n"]), int(r["dataset"])) for r in load()}
    path = os.path.join(OUT, "records.csv")
    new = not os.path.exists(path)
    # n in increasing order; a statistic that times out or exceeds the memory limit at some n is not
    # run at larger n (recorded as "not run"), never extrapolated
    first_fail = {}                                    # (method, device) -> smallest n at which it failed
    for r in load():
        if r.get("grid", "dense") == a.grid and r["status"] in ("timeout", "memory"):
            k = (r["method"], r["device"])
            first_fail[k] = min(first_fail.get(k, 10 ** 9), int(r["n"]))
    with open(path, "a", newline="") as f, ThreadPoolExecutor(a.procs) as ex:
        wr = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        if new:
            wr.writeheader()
        for n in sorted(ns):
            failed = {k for k, n0 in first_fail.items() if n0 <= n}
            jobs = [(a.grid, m, n, r, a.device) for m in methods for r in range(CFG["datasets"])
                    if (a.grid, m, a.device, n, r) not in done and not (a.device == "cuda" and m != "Gram")]
            def emit(rec):
                wr.writerow(rec)
                f.flush()
                print(f"[{time.strftime('%H:%M:%S')}] {rec['grid']} {rec['method']} {rec['device']} n={rec['n']} "
                      f"r={rec['dataset']}: {rec['status']} {rec.get('seconds')} s", flush=True)

            def not_run(j):
                g, m, nn, r, d = j
                return dict(grid=g, method=m, device=d, n=nn, dataset=r, status="not run")

            for j in jobs:
                if (j[1], j[4]) in failed:
                    emit(not_run(j))
            futs = {ex.submit(run_job, j): j for j in jobs if (j[1], j[4]) not in failed}
            for fu in as_completed(futs):
                if fu.cancelled():
                    continue
                rec = fu.result()
                emit(rec)
                if rec["status"] in ("timeout", "memory"):
                    key = (rec["method"], rec["device"])
                    failed.add(key)
                    first_fail[key] = min(first_fail.get(key, 10 ** 9), int(rec["n"]))
                    # the other replicates of this statistic at this n that have not started are not run
                    for g2, j2 in futs.items():
                        if (j2[1], j2[4]) == key and not g2.cancelled() and g2.cancel():
                            emit(not_run(j2))
    summarise()


if __name__ == "__main__":
    main()
