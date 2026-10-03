#!/usr/bin/env python
"""Weather stage two under season-restricted block permutation (Section 6.7).

Purpose
-------
On random real dates the unrestricted stage-two calibration, which permutes the order of six-week
super-blocks freely within the +/- 1 year window, rejects too often: 9.7% at nominal 0.05 for the
dependence test and 23.2% for the marginal-scale test (run_e8_validate.py; Supplementary
Section B.8). The cause is a seasonal cycle in the variability of the anomalies, which makes blocks
from different seasons non-exchangeable. Permuting super-blocks only among those of the same
calendar season (DJF, MAM, JJA, SON, assigned by the month of the super-block's middle hour) makes
the permutation group respect that structure; the test is then exact under within-season block
exchangeability. This script applies that restriction to every test. Every other setting is that
of run_e8_retest.py: the window (+/- 26 two-week intervals), grid (336-hour intervals), super-block
length (three intervals), minimum segment (three intervals), statistic, features (D = 8, seeds 2026 / 2027), and
permutation seeds (4242 + interval for the dependence test; 9090 + interval + crc32(variable) mod
97 for the marginal test, as in run_e8_retest.py and run_e8_marginal.py).

Parts
-----
  cand     all 27 stage-one candidates (e8/results.json): the dependence test and the
           marginal-scale test on each variable, both at K = 9999. Classification as in
           run_e8_marginal.py: dependence change at Benjamini-Hochberg q <= 0.10
           across the 27, marginal change at min(p_X, p_Y) <= 0.05; Benjamini-Yekutieli q-values
           alongside.
  randomdate  random-date calibration: the 720 random dates of run_e8_validate.py (20 per station-pair series,
           seed [20260926, series index], at least one year from either end, no stage one), both
           tests at K = 999.
  split    the split-sample analysis of run_e8_split.py with this stage two: the stage-one
           candidates of e8_split/results.json (selection on one half), tested on the other half's
           six-week retained blocks within +/- 52 weeks, the blocks permuted only within season,
           K = 9999, the split's permutation seeds; BH / BY within each direction.
  summary  combines the part records into results.json.

Run
---
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_e8_season.py --part cand --procs 6
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_e8_season.py --part randomdate --procs 12
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_e8_season.py --part split --procs 6
    python 00_SRC/run_e8_season.py --part summary

Outputs (04_DAOU/EXPERIMENT/e8_season/)
  records_{cand,randomdate,split}.csv.gz   one row per test: identifiers, p, observed statistic followed
                                        by all K replica maxima (every p-value regenerates)
  part_{cand,randomdate,split}.json        per-part metadata (K, wall-clock)
  results.json                          the tables: candidates with classes and q-values, class
                                        counts, random-date calibration, split-sample results
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys
import time
import zlib
from collections import Counter
from multiprocessing import Pool

import numpy as np
from scipy import stats

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.hourly import features, block_moments, prefix, domi_diff_curve  # noqa: E402
from dots.perm import ge  # noqa: E402
from run_e8_marginal import scale_curve, curve_from  # noqa: E402
from run_e8_summary import step_up_q  # noqa: E402
from run_e8_hourly import NAMES  # noqa: E402
import run_e8_split as SPL  # noqa: E402
import run_e8_validate as VAL  # noqa: E402

E8 = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8")
SPLIT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8_split")
TIES = os.environ.get("E8_TIES", "order")        # "order": ties broken in time order (as reported);
                                                 # "random": ties broken at random (Supplementary Section B.8)
OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8_season" + ("" if TIES == "order" else "_randomties"))
UNIT, SUPER, HALF, LO = 336, 3, 26, 3
K_CAND, K_RANDOMDATE, K_SPLIT = 9999, 999, 9999
SEASON = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}


def season_of(ts):
    return SEASON[int(str(ts)[4:6])]


def strat_order(n_units, block, seasons, rng):
    """Order of grid units after permuting whole blocks of `block` units only among blocks of the
    same season; a trailing remainder shorter than a block stays in place."""
    nb = n_units // block
    order = np.arange(nb)
    for s in range(4):
        g = np.where(seasons[:nb] == s)[0]
        if len(g) > 1:
            order[g] = g[rng.permutation(len(g))]
    idx = (order[:, None] * block + np.arange(block)[None, :]).ravel()
    return np.concatenate([idx, np.arange(nb * block, n_units)])


def perm_p(obs, R):
    A = np.vstack([obs[None, :], R])
    mu, sd = A.mean(0), A.std(0) + 1e-12
    T = np.nanmax((A - mu) / sd, axis=1)
    return (1 + int(np.sum(ge(T[1:], T[0])))) / len(T), T


def season_tests(x, y, t, edges, block, lo, K, dep_rng, marg_rngs):
    """Dependence and marginal tests on one window with season-restricted block permutation.
    `edges` are the grid-unit edges of the window, `t` its timestamps."""
    B = len(edges) - 1
    nb = B // block
    seasons = np.array([season_of(t[min(edges[k * block] + (edges[min((k + 1) * block, B)] -
                                                             edges[k * block]) // 2, len(t) - 1)])
                        for k in range(nb)])
    FX, FY, J = features(x, y, tie_seed=(None if TIES == "order" else 777))
    mx, my, mj, cnt = block_moments(FX, FY, J, edges)
    ts, obs = domi_diff_curve(prefix(mx, my, mj, cnt), B, lo=lo)
    R = np.empty((K, len(ts)))
    for k in range(K):
        o = strat_order(B, block, seasons, dep_rng)
        R[k] = domi_diff_curve(prefix(mx[o], my[o], mj[o], cnt[o]), B, lo=lo)[1]
    p_dep, Td = perm_p(obs, R)
    out = dict(p_dep=float(p_dep), T_dep=Td, season_blocks=np.bincount(seasons, minlength=4).tolist())
    for tag, s, rng in (("x", x, marg_rngs[0]), ("y", y, marg_rngs[1])):
        s1, s2, cn = scale_curve(s, edges)
        _, ob = curve_from(s1, s2, cn, lo=lo)
        Rm = np.empty((K, len(ob)))
        for k in range(K):
            o = strat_order(B, block, seasons, rng)
            Rm[k] = curve_from(s1[o], s2[o], cn[o], lo=lo)[1]
        pm, Tm = perm_p(ob, Rm)
        out[f"p_marg_{tag}"], out[f"T_marg_{tag}"] = float(pm), Tm
    out["p_marg_min"] = min(out["p_marg_x"], out["p_marg_y"])
    return out


# ---------------------------------------------------------------- full-record window (cand, randomdate)
def _full_job(args):
    part, stn, pair, c, date, K = args
    u, v = pair.split("-")
    a, b, tm = VAL.load(stn, u, v)
    sl, _ = VAL.window(a, c)
    x, y, t = a[sl], b[sl], tm[sl]
    ew = np.arange(0, len(x) + 1, UNIT)
    if ew[-1] != len(x):
        ew = np.append(ew, len(x))
    r = season_tests(x, y, t, ew, SUPER, LO, K, np.random.default_rng(4242 + c),
                     [np.random.default_rng(9090 + c + zlib.crc32(n.encode()) % 97) for n in (u, v)])
    return dict(part=part, stn=stn, pair=pair, interval=c, date=date, **r)


# ---------------------------------------------------------------- split-sample test half
def _split_job(args):
    stn, pair, sel, cand, seed = args
    u, v = pair.split("-")
    a, b, tm = SPL.load_pair(stn, u, v)
    test = "B" if sel == "A" else "A"
    blocks = [bk for bk in SPL.split_index(len(a))[test]
              if bk[0] >= cand["t_star"] - SPL.HALF_WIN and bk[-1] < cand["t_star"] + SPL.HALF_WIN]
    base = dict(part="split", stn=stn, pair=pair, select=sel, test=test, date=cand["date"],
                t_star=cand["t_star"], n_blocks=len(blocks))
    if len(blocks) < SPL.MIN_BLOCKS:
        return dict(base, tested=False)
    idx = np.concatenate(blocks)
    r = season_tests(a[idx], b[idx], tm[idx], SPL.unit_edges(len(idx)), SPL.UPB, SPL.UPB, K_SPLIT,
                     np.random.default_rng([seed, 1]),
                     [np.random.default_rng([seed, 2]), np.random.default_rng([seed, 3])])
    return dict(base, tested=True, window=f"{str(tm[idx[0]])[:8]}-{str(tm[idx[-1]])[:8]}", **r)


# ---------------------------------------------------------------- records
REC_KEYS = ["part", "stn", "pair", "select", "date", "interval", "t_star", "n_blocks", "tested",
            "season_blocks", "test", "p", "T"]


def write_records(part, res):
    with gzip.open(os.path.join(OUT, f"records_{part}.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REC_KEYS, extrasaction="ignore")
        w.writeheader()
        for r in res:
            base = {k: r.get(k, "") for k in REC_KEYS[:10]}
            base["season_blocks"] = json.dumps(r.get("season_blocks", ""))
            if r.get("tested", True) is False:
                w.writerow(base)
                continue
            u, v = r["pair"].split("-")
            for test, pk, tk in (("dependence", "p_dep", "T_dep"), (f"marginal:{u}", "p_marg_x", "T_marg_x"),
                                 (f"marginal:{v}", "p_marg_y", "T_marg_y")):
                w.writerow(dict(base, test=test, p=r[pk], T=" ".join(f"{t:.6f}" for t in r[tk])))


def read_records(part):
    with gzip.open(os.path.join(OUT, f"records_{part}.csv.gz"), "rt", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        key = (r["stn"], r["pair"], r["select"], r["date"], r["interval"], r["t_star"])
        e = out.setdefault(key, dict(stn=r["stn"], pair=r["pair"], select=r["select"], date=r["date"],
                                     interval=r["interval"], t_star=r["t_star"], n_blocks=r["n_blocks"],
                                     tested=r["tested"] != "False",
                                     season_blocks=json.loads(r["season_blocks"] or '""')))
        if not r["test"]:
            continue
        u, v = r["pair"].split("-")
        name = {"dependence": "p_dep", f"marginal:{u}": "p_marg_x", f"marginal:{v}": "p_marg_y"}[r["test"]]
        e[name] = float(r["p"])
    for e in out.values():
        if e["tested"]:
            e["p_marg_min"] = min(e["p_marg_x"], e["p_marg_y"])
    return list(out.values())


def classify(rows):
    p = np.array([r["p_dep"] for r in rows])
    H = float(np.sum(1.0 / np.arange(1, len(p) + 1)))
    bh, by = step_up_q(p), step_up_q(p, H)
    for r, q1, q2 in zip(rows, bh, by):
        r["bh_q"], r["by_q"] = round(float(q1), 4), round(float(q2), 4)
        d, m = q1 <= 0.10, r["p_marg_min"] <= 0.05
        r["cls"] = ("dependence change" if d and not m else "both change" if d and m
                    else "marginal-driven" if m else "undetermined")
    return dict(n=len(rows), n_p_le_05=int((p <= 0.05).sum()), n_bh_q_le_10=int((bh <= 0.10).sum()),
                n_bh_q_le_05=int((bh <= 0.05).sum()), n_by_q_le_10=int((by <= 0.10).sum()),
                n_by_q_le_05=int((by <= 0.05).sum()), class_counts=dict(Counter(r["cls"] for r in rows)))


def calib(p, ref05=0.05):
    p = np.asarray(p)
    n = len(p)
    row = {}
    for a in (0.10, 0.05, 0.01):
        k = int((p <= a).sum())
        row[f"le_{a}"] = dict(k=k, n=n, rate=k / n, binom_sf=float(stats.binom.sf(k - 1, n, a)))
    row["ks_uniform_p"] = float(stats.kstest(p, "uniform").pvalue)
    row["mean_p"] = float(p.mean())
    row["quantiles_01_05_10_25_50"] = [float(np.quantile(p, q)) for q in (0.01, 0.05, 0.1, 0.25, 0.5)]
    return row


# ---------------------------------------------------------------- main
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["cand", "randomdate", "split", "summary"])
    ap.add_argument("--procs", type=int, default=6)
    a_ = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a_.part != "summary":
        if a_.part == "cand":
            d = json.load(open(os.path.join(E8, "results.json")))
            jobs = [("cand", r["stn"], r["pair"], b["interval"], b["date"], K_CAND)
                    for r in d["results"] for b in r["breaks"]]
            fn = _full_job
        elif a_.part == "randomdate":
            jobs = [("randomdate", s, p, c, dt, K_RANDOMDATE) for s, p, c, dt, _ in VAL.randomdate_dates()]
            fn = _full_job
        else:
            d = json.load(open(os.path.join(SPLIT, "results.json")))
            jobs = []
            for r in d["stage1"]:
                for c in r["candidates"]:
                    seed = zlib.crc32(f"{r['stn']}{r['pair']}{r['select']}{c['unit']}".encode())
                    jobs.append((r["stn"], r["pair"], r["select"], c, seed))
            fn = _split_job
        print(f"part {a_.part}: {len(jobs)} windows", flush=True)
        t0 = time.time()
        with Pool(a_.procs, maxtasksperchild=4) as pool:
            res = pool.map(fn, jobs, chunksize=1)
        wall = round(time.time() - t0, 1)
        write_records(a_.part, res)
        json.dump(dict(part=a_.part, n=len(jobs), wall_clock_s=wall, procs=a_.procs,
                       K={"cand": K_CAND, "randomdate": K_RANDOMDATE, "split": K_SPLIT}[a_.part]),
                  open(os.path.join(OUT, f"part_{a_.part}.json"), "w"), indent=1)
        print(f"part {a_.part} done in {wall:.0f}s", flush=True)
        return

    # ---- summary
    pub_c = {(r["stn"], r["pair"], str(r["interval"])): r
             for r in json.load(open(os.path.join(E8, "retest_K999.json")))["results"]}
    pub_m = {(r["stn"], r["pair"], str(r["interval"])): r
             for r in json.load(open(os.path.join(E8, "marginal_diagnostic.json")))["results"]}
    cand = read_records("cand")
    cand_sum = classify(cand)
    for r in cand:
        k = (r["stn"], r["pair"], r["interval"])
        r.update(station=NAMES[r["stn"]], p_dep_unrestricted=pub_c[k]["p_block"],
                 bh_q_unrestricted=pub_c[k]["bh_q"], p_marg_min_unrestricted=pub_m[k]["p_marg_min"],
                 cls_unrestricted=pub_m[k]["class"])
    cand.sort(key=lambda r: r["p_dep"])
    rdate = read_records("randomdate")
    calib_tab = dict(dependence=calib([r["p_dep"] for r in rdate]),
                     marginal_min=calib([r["p_marg_min"] for r in rdate]),
                     marginal_each_variable=calib([r[k] for r in rdate for k in ("p_marg_x", "p_marg_y")]),
                     note="marginal_min is min(p_X, p_Y), the quantity the classification uses; its "
                          "reference rate at 0.05 is at most 0.0975 (0.05 if the two are identical)")
    split = read_records("split")
    split_dirs = {}
    for sel in ("A", "B"):
        rows = [r for r in split if r["select"] == sel and r["tested"]]
        split_dirs[f"select_{sel}"] = classify(rows)
    for r in split:
        r["station"] = NAMES[r["stn"]]
    parts = {p: json.load(open(os.path.join(OUT, f"part_{p}.json"))) for p in ("cand", "randomdate", "split")}
    res = dict(config=dict(permutation="season-restricted block permutation (DJF/MAM/JJA/SON by the "
                                       "month of the block's middle hour)",
                           K=dict(candidates=K_CAND, randomdate=K_RANDOMDATE, split=K_SPLIT),
                           window_intervals=HALF, unit_hours=UNIT, super_block_intervals=SUPER,
                           classification="dependence: BH q<=0.10 across candidates; marginal: "
                                          "min(p_X,p_Y)<=0.05 uncorrected"),
               candidates=cand, candidate_summary=cand_sum, randomdate_calibration=calib_tab,
               split_sample=dict(directions=split_dirs, candidates=split),
               wall_clock_s={p: v["wall_clock_s"] for p, v in parts.items()})
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(json.dumps(cand_sum))
    for r in cand:
        print(f"  {r['station']:<10}{r['pair']:<7}{r['date']:<10} p_dep={r['p_dep']:.4f} "
              f"(unrestricted {r['p_dep_unrestricted']:.3f}) q={r['bh_q']:.3f} BY={r['by_q']:.3f} px={r['p_marg_x']:.4f} py={r['p_marg_y']:.4f} "
              f"{r['cls']} (unrestricted {r['cls_unrestricted']})")
    print(json.dumps(calib_tab, indent=1))
    print(json.dumps(split_dirs, indent=1))
    for r in sorted([r for r in split if r["tested"]], key=lambda r: (r["select"], r["p_dep"]))[:12]:
        print(f"  {r['select']} {r['station']:<10}{r['pair']:<7}{r['date']:<10} p_dep={r['p_dep']:.4f} "
              f"q={r['bh_q']:.3f} BY={r['by_q']:.3f} pm={r['p_marg_min']:.4f} {r['cls']}")


if __name__ == "__main__":
    main()
