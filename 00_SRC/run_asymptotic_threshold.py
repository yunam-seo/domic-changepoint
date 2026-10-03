#!/usr/bin/env python
"""Level, power and cost of the permutation-free asymptotic calibration (``asymptotic_threshold.py``).

Permutation-free calibration by the boundary limit law (P6 / P6-R and their process-level
Corollaries, Supplementary Section A.12); reproduces Supplementary Section B.16, Table B.13.  Phases:

  bias_constant : bias constant c = sum of limit weights (n = 600, replicate 0 of the H0 design)
              at support cuts tau = 1e-12, 1e-9, 1e-6, 1e-4 (sensitivity of c to the cut).
  exactmc   : exact Monte Carlo null of T_raw (independent uniform ranks, deployed features),
              2000 draws per n; also oracle-margin draws at n=9600 (rank limit = oracle limit).
  level     : level of the asymptotic test (T_raw and T_std) under
              H0 (independent Gaussian pairs, code D2 null) and M1 (X scale 1 -> 1.6 at n/2,
              independence throughout), n = 600, 2400, 9600, 38400; permutation tests (K=99)
              under both nulls at n = 600, 2400.
  power     : S2-type emergence of dependence (code D2: independent -> y = sqrt(1-a^2) e1 + a|x| e2
              at n/2), a = 0.7, 0.9; asymptotic at all four n, permutation (K=99) at 600, 2400.
  timing    : single-core wall-clock per test: asymptotic, permutation (K=99), exact-MC table.

Run (4 processes, BLAS pinned to one thread):
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  python 00_SRC/run_asymptotic_threshold.py --phase all --procs 4

Writes under 04_DAOU/EXPERIMENT/asymptotic_threshold/:
  bias_constant.json, exactmc_records.csv, level_records.csv, power_records.csv, timing_records.csv,
  results.json (aggregates with binomial standard errors), results_tables.txt.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
import asymptotic_threshold as AT  # noqa: E402
from dots.synth import generate  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "asymptotic_threshold")
NS = [600, 2400, 9600, 38400]
PERM_NS = [600, 2400]
BASE = 20260925
ALPHA = 0.05
N_DRAW = 2000
K = 99


def write_csv(name, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(name):
    with open(os.path.join(OUT, name)) as f:
        return list(csv.DictReader(f))


def sample(kind, n, rep, a=None):
    """kind in {H0, M1, S2}; returns X, Y (n,1) and the true break (None under H0)."""
    if kind == "H0":
        s = generate("D2", 0.7, rep, n=n, tau=n // 2, null=True, base_seed=BASE, level_idx=9)
    elif kind == "M1":
        s = generate("M1", 1.6, rep, n=n, tau=n // 2, null=False, base_seed=BASE, level_idx=1)
    else:
        s = generate("D2", a, rep, n=n, tau=n // 2, null=False, base_seed=BASE,
                     level_idx={0.7: 1, 0.9: 2}[a])
    Z = s["Z"]
    return Z[:, [0]], Z[:, [1]]


# ---------------------------------------------------------------- bias constant
def phase_bias_constant():
    X, Y = sample("H0", 600, 0)
    FX, FY = AT.features(X, Y)
    sens = {}
    for tau in (1e-12, 1e-9, 1e-6, 1e-4):
        w, info = AT.limit_weights(FX, FY, tau)
        sens[str(tau)] = info
    out = dict(c_support_sensitivity_n600_rep0=sens)
    json.dump(out, open(os.path.join(OUT, "bias_constant.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- exact MC
def _mc_job(a):
    n, i, oracle = a
    rng = np.random.default_rng([BASE, 77, n, i, int(oracle)])
    R, nI = AT.exact_mc_draw(n, rng, oracle=oracle)
    return dict(n=n, draw=i, oracle=int(oracle), T_raw=float(R.max()), nI_whole=float(nI),
                argmax_frac=float(AT.fraction_grid(n)[int(np.argmax(R))] / n))


def phase_exactmc(procs, draws):
    jobs = [(n, i, False) for n in NS for i in range(draws)] + [(9600, i, True) for i in range(draws)]
    with Pool(procs) as p:
        rows = p.map(_mc_job, jobs, chunksize=8)
    write_csv("exactmc_records.csv", rows)


# ---------------------------------------------------------------- level / power
def _test_job(a):
    kind, n, rep, aa, do_perm = a
    X, Y = sample(kind, n, rep, aa)
    t0 = time.process_time()
    r = AT.asymptotic_test(X, Y, alpha=ALPHA, n_draw=N_DRAW, seed=rep)
    t_as = time.process_time() - t0
    row = dict(kind=kind, a=aa if aa is not None else "", n=n, rep=rep, c=r["c"],
               T_raw=r["T_raw"], crit_raw=r["crit_raw"], p_raw=r["p_raw"],
               T_std=r["T_std"], crit_std=r["crit_std"], p_std=r["p_std"],
               tau_hat_raw=r["tau_hat_raw"], tau_hat_std=r["tau_hat_std"], cpu_asym=t_as,
               p_perm_std="", p_perm_raw="", tau_hat_perm_std="", cpu_perm="")
    if do_perm:
        t0 = time.process_time()
        pr = AT.permutation_test(X, Y, K=K, alpha=ALPHA, seed=int(1e6) + rep)
        row.update(pr, cpu_perm=time.process_time() - t0)
        row.pop("tau_hat_perm_raw", None)
    return row


def phase_level(procs, reps, perm_reps):
    jobs = []
    for kind in ("H0", "M1"):
        for n in NS:
            for rep in range(reps):
                jobs.append((kind, n, rep, None, n in PERM_NS and rep < perm_reps))
    with Pool(procs) as p:
        rows = p.map(_test_job, jobs, chunksize=4)
    write_csv("level_records.csv", rows)


def phase_power(procs, reps):
    jobs = [("S2", n, rep, a, n in PERM_NS) for a in (0.7, 0.9) for n in NS for rep in range(reps)]
    with Pool(procs) as p:
        rows = p.map(_test_job, jobs, chunksize=2)
    write_csv("power_records.csv", rows)


# ---------------------------------------------------------------- timing (single core, sequential)
def phase_timing(n_rep):
    rows = []
    for n in NS:
        for rep in range(n_rep):
            X, Y = sample("H0", n, 10_000 + rep)
            t0 = time.perf_counter(); AT.asymptotic_test(X, Y, n_draw=N_DRAW, seed=rep); t1 = time.perf_counter()
            AT.permutation_test(X, Y, K=K, seed=rep); t2 = time.perf_counter()
            rng = np.random.default_rng([BASE, 99, n, rep])
            t3 = time.perf_counter()
            for _ in range(20):
                AT.exact_mc_draw(n, rng)
            t4 = time.perf_counter()
            # one curve on the every-t grid [n/10, 9n/10] (the deployed grid spacing), for scale
            FX, FY = AT.features(X, Y)
            AT.segment_curves(FX, FY, np.arange(n // 10, n - n // 10 + 1))
            t5 = time.perf_counter()
            rows.append(dict(n=n, rep=rep, wall_asym=t1 - t0, wall_perm_K99=t2 - t1,
                             wall_one_curve_every_t=t5 - t4,
                             wall_exactmc_per_draw=(t4 - t3) / 20,
                             wall_exactmc_2000_table=(t4 - t3) / 20 * N_DRAW))
            print(rows[-1], flush=True)
    write_csv("timing_records.csv", rows)


# ---------------------------------------------------------------- aggregate
def _rate(v):
    v = np.asarray(v, float)
    p = float(v.mean())
    return dict(rate=p, se=float(np.sqrt(p * (1 - p) / len(v))), reps=int(len(v)))


def aggregate():
    res = {}
    lines = []
    mc = read_csv("exactmc_records.csv")
    lev = read_csv("level_records.csv")
    pw = read_csv("power_records.csv")
    tm = read_csv("timing_records.csv") if os.path.exists(os.path.join(OUT, "timing_records.csv")) else []
    f = lambda rows, k: np.array([float(r[k]) for r in rows])

    # exact MC vs asymptotic critical values of T_raw
    res["exactmc"] = {}
    lines.append("Block A. Critical value (95%) of T_raw: exact Monte Carlo (ranks) vs asymptotic")
    lines.append("n       exactMC q95 [95% CI]      asym crit median [IQR over H0 reps]   c_MC=mean nI   c_limit median")
    for n in NS:
        rows = [r for r in mc if int(r["n"]) == n and r["oracle"] == "0"]
        T = f(rows, "T_raw"); m = len(T)
        q = float(np.quantile(T, 0.95))
        ci = [int(np.floor(m * 0.95 - 1.96 * np.sqrt(m * .0475))), int(np.ceil(m * 0.95 + 1.96 * np.sqrt(m * .0475)))]
        lo, hi = np.sort(T)[np.clip(ci, 0, m - 1)]  # distribution-free order-statistic CI for q95
        hl = [r for r in lev if r["kind"] == "H0" and int(r["n"]) == n]
        cr = f(hl, "crit_raw"); cs = f(hl, "crit_std"); cc = f(hl, "c")
        # exact-MC level of the asymptotic rule, using each H0 rep's crit against MC draws is not
        # meaningful (crit depends on the bandwidth); report fraction of MC draws above median crit
        frac = float((T > np.median(cr)).mean())
        res["exactmc"][n] = dict(draws=m, q95=q, q95_ci=[float(lo), float(hi)],
                                 asym_crit_median=float(np.median(cr)),
                                 asym_crit_iqr=[float(np.quantile(cr, .25)), float(np.quantile(cr, .75))],
                                 mc_exceed_median_asym_crit=frac,
                                 c_mc_mean_nI=float(f(rows, "nI_whole").mean()),
                                 c_mc_se=float(f(rows, "nI_whole").std() / np.sqrt(m)),
                                 c_limit_median=float(np.median(cc)),
                                 asym_crit_std_median=float(np.median(cs)))
        e = res["exactmc"][n]
        lines.append(f"{n:<7d} {q:7.2f} [{lo:6.2f},{hi:6.2f}]       {e['asym_crit_median']:6.2f} "
                     f"[{e['asym_crit_iqr'][0]:.2f},{e['asym_crit_iqr'][1]:.2f}]"
                     f"                  {e['c_mc_mean_nI']:.3f}±{e['c_mc_se']:.3f}    {e['c_limit_median']:.3f}")
    orc = f([r for r in mc if r["oracle"] == "1"], "T_raw")
    rk = f([r for r in mc if r["oracle"] == "0" and int(r["n"]) == 9600], "T_raw")
    from scipy import stats
    ks = stats.ks_2samp(orc, rk)
    res["rank_vs_oracle_n9600"] = dict(ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
                                       q95_oracle=float(np.quantile(orc, .95)), q95_rank=float(np.quantile(rk, .95)))
    lines.append(f"Rank vs oracle margins, n=9600: q95 {res['rank_vs_oracle_n9600']['q95_rank']:.2f} vs "
                 f"{res['rank_vs_oracle_n9600']['q95_oracle']:.2f}; KS p={ks.pvalue:.3f}")

    lines.append("")
    lines.append("Block B. Level at nominal 0.05 (rate ± SE, reps)")
    lines.append("null  n       asym T_raw          asym T_std          perm studentised    perm T_raw")
    res["level"] = {}
    for kind in ("H0", "M1"):
        for n in NS:
            rows = [r for r in lev if r["kind"] == kind and int(r["n"]) == n]
            e = dict(asym_raw=_rate(f(rows, "p_raw") <= ALPHA), asym_std=_rate(f(rows, "p_std") <= ALPHA))
            pr = [r for r in rows if r["p_perm_std"] != ""]
            if pr:
                e["perm_std"] = _rate(f(pr, "p_perm_std") <= ALPHA)
                e["perm_raw"] = _rate(f(pr, "p_perm_raw") <= ALPHA)
            res["level"][f"{kind}_{n}"] = e
            fmt = lambda d: f"{d['rate']:.3f}±{d['se']:.3f} ({d['reps']})" if d else "-"
            lines.append(f"{kind:<5} {n:<7d} {fmt(e['asym_raw']):<19} {fmt(e['asym_std']):<19} "
                         f"{fmt(e.get('perm_std')):<19} {fmt(e.get('perm_raw'))}")

    lines.append("")
    lines.append("Block C. Power, S2-type emergence of dependence at n/2 (reject rate; in brackets: reject and |tau_hat - n/2| <= n/20)")
    lines.append("a    n       asym T_raw            asym T_std            perm studentised      perm T_raw")
    res["power"] = {}
    for a in ("0.7", "0.9"):
        for n in NS:
            rows = [r for r in pw if r["a"] == a and int(r["n"]) == n]
            if not rows:
                continue
            tol = n / 20
            def both(pk, tk):
                p = f(rows, pk) <= ALPHA
                loc = p & (np.abs(f(rows, tk) - n // 2) <= tol)
                return dict(reject=_rate(p), reject_and_localised=_rate(loc))
            e = dict(asym_raw=both("p_raw", "tau_hat_raw"), asym_std=both("p_std", "tau_hat_std"))
            if rows[0]["p_perm_std"] != "":
                e["perm_std"] = both("p_perm_std", "tau_hat_perm_std")
                p = f(rows, "p_perm_raw") <= ALPHA
                e["perm_raw"] = dict(reject=_rate(p))
            res["power"][f"{a}_{n}"] = e
            fm = lambda d: (f"{d['reject']['rate']:.3f}±{d['reject']['se']:.3f}" +
                            (f" [{d['reject_and_localised']['rate']:.3f}]" if 'reject_and_localised' in d else "")) if d else "-"
            lines.append(f"{a:<4} {n:<7d} {fm(e['asym_raw']):<21} {fm(e['asym_std']):<21} "
                         f"{fm(e.get('perm_std')):<21} {fm(e.get('perm_raw'))}   (reps {len(rows)})")

    if tm:
        lines.append("")
        lines.append("Block D. Single-core wall-clock per test, seconds (median over reps)")
        lines.append("n       asymptotic   permutation K=99   exact-MC table (2000 draws)   one curve, every-t grid")
        res["timing"] = {}
        for n in NS:
            rows = [r for r in tm if int(r["n"]) == n]
            e = {k: float(np.median(f(rows, k))) for k in ("wall_asym", "wall_perm_K99", "wall_exactmc_2000_table",
                                                            "wall_one_curve_every_t")}
            res["timing"][n] = e
            lines.append(f"{n:<7d} {e['wall_asym']:10.2f}   {e['wall_perm_K99']:16.2f}   {e['wall_exactmc_2000_table']:12.1f}"
                         f"   {e['wall_one_curve_every_t']:10.2f}")
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    open(os.path.join(OUT, "results_tables.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["bias_constant", "exactmc", "level", "power", "timing", "aggregate", "all"])
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2000, help="level replicates (asymptotic)")
    ap.add_argument("--perm-reps", type=int, default=500, help="level replicates for permutation tests")
    ap.add_argument("--power-reps", type=int, default=500)
    ap.add_argument("--mc-draws", type=int, default=2000)
    ap.add_argument("--timing-reps", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    json.dump(vars(a) | dict(NS=NS, PERM_NS=PERM_NS, BASE=BASE, ALPHA=ALPHA, N_DRAW=N_DRAW, K=K,
                             grid="t=round(n*l), l=0.100:0.005:0.900", D=8),
              open(os.path.join(OUT, f"config_{a.phase}.json"), "w"), indent=1)
    t0 = time.time()
    if a.phase in ("bias_constant", "all"):
        phase_bias_constant()
    if a.phase in ("exactmc", "all"):
        phase_exactmc(a.procs, a.mc_draws)
    if a.phase in ("level", "all"):
        phase_level(a.procs, a.reps, a.perm_reps)
    if a.phase in ("power", "all"):
        phase_power(a.procs, a.power_reps)
    if a.phase in ("timing", "all"):
        phase_timing(a.timing_reps)
    if a.phase in ("aggregate", "all"):
        aggregate()
    print(f"done in {time.time() - t0:.0f}s")
