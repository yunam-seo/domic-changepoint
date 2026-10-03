#!/usr/bin/env python
"""Exploratory resampling check of the whole two-stage weather procedure (stage-one segmentation +
stage-two re-test), Section 6.7; Supplementary Section B.14.

Purpose
-------
Stage two of Section 4.7 tests windows that stage one selected from the same record, so the
stage-two p-values are nominal (Propositions P1 / P5 assume a window fixed in advance). Permuting
the whole procedure would address the selection. Here the stage-one penalty and bias correction
are held at their observed values, so the observed and permuted statistics are not computed by
the same rule and the result is exploratory: finite-sample validity and selection adjustment are
not established. For each of the 36 station-pair analyses the script gives
  (i)  a check of "no change anywhere in the analysis" (the permutation null below), and
  (ii) single-step Westfall-Young max-T adjusted p-values for the individual candidates. Max-T
       gives strong family-wise control only under subset pivotality, which the selection step
       makes doubtful; like (i), (ii) is exploratory.
Across the 36 analyses: Holm and Benjamini-Yekutieli on the per-analysis p-values of (i), and a
joint min-p test over all 36 analyses using the permutations common to all of them.

Pipeline (that of run_e8_hourly.py + run_e8_retest.py)
-------------------------------------------------------
Stage one: Holevo optimal partitioning on the 336-hour grid (B = 209 intervals), costs centered by
three block-permuted copies (Cp), penalty beta calibrated on 20 block-permuted copies; block
permutations of super-blocks of 6 intervals (12 weeks), the 5 trailing intervals held fixed.
Stage two, per candidate interval c: +/- 26 intervals (one year) around c, ranks and features
recomputed in the window, the weighted DOMI-difference curve over split points 3..Bw-3, studentized
by block-permutation replicas (3-interval, six-week super-blocks), T = max over split points of the
studentized curve -- exactly the statistic of run_e8_retest.py, with K_IN = 99 inner replicas.
T is the statistic, not the inner p-value: it is continuous, and on a common studentized scale
across candidates of one analysis (windows truncated at the record ends have fewer split points,
which only shifts T's null law slightly downward).

Outer permutation (the null)
----------------------------
Season-stratified permutations of the 34 twelve-week super-blocks of stage one: a super-block is
labeled by the calendar season (DJF/MAM/JJA/SON) of its middle interval, and super-blocks are
permuted only among those with the same label; the 5 trailing intervals stay in place. The null
hypothesis is that the sequence of super-blocks of the (X, Y) record is exchangeable within season
labels. Twelve-week blocks are used rather than stage two's six-week blocks for two reasons:
  * Every such permutation is one of the permutations from which stage one draws its own
    calibration copies (unstratified permutations of the same super-blocks). The observed Cp and
    beta are held fixed on the permuted records, which cuts a pipeline run from 84 stage-one cost
    matrices to 1: recomputing the calibration on every permuted record would take over two
    minutes per run (Supplementary Section B.9) against about 1.7 s with it held fixed
    (Supplementary Section B.14).
  * Section 6.7's block-length diagnostic recommends 4.0-22.2 weeks (median 6.6); twelve-week
    blocks meet it in 33 of 36 analyses, six-week blocks in 8.
The same permutation sigma_b is applied to all 36 analyses (their grids coincide up to a few
missing hours), which gives the joint min-p combination across analyses.

Data are the global ranks of each variable (ties broken by time order, as ranks01 does). Features
are functions of those ranks, so the permuted stage-one moments are the permuted interval moments.
The raw humidity and wind records contain ties; the pipeline breaks them by time order,
and the test treats the resulting ranks as the data (the within-tie reordering a permutation would
induce on the raw values moves a rank by at most 39/70,000).

Statistics
----------
T_ab = max_j T_abj over the candidates stage one returns on permuted record b of analysis a
(-inf if none). b = 0 is the observed record.
  per candidate:  p_adj_j = (1 + #{b >= 1: T_ab >= T_j}) / (B + 1)
  per analysis:   p_a = p_adj of its largest T_j (1 if stage one returned no candidate)
  across:         Holm and BY on the 36 p_a; joint min-p over analyses with q_ab the rank p-value
                  of T_ab within analysis a (b = 0..B), Q_b = min_a q_ab,
                  p_global = #{b: Q_b <= Q_0} / (B + 1), and the joint single-step adjusted
                  p for candidate j of analysis a: #{b: Q_b <= q_a(T_j)} / (B + 1).

Run
---
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python 00_SRC/run_e8_fullperm.py --B 199 --procs 8
Resumable and extensible: permutation b uses seed [20260925, b] for all analyses, chunks already
on disk are skipped, so running again with a larger --B extends the same test.
    python 00_SRC/run_e8_fullperm.py --B 199 --summarize-only     # rebuild results from chunks/

Inputs:  02_MART/WEATHER_HOURLY_ANOM.npz; 04_DAOU/EXPERIMENT/e8/{results.json,
         records_stage1_beta_perm.csv, retest_K999.json, retest_K9999.json}
Outputs: 04_DAOU/EXPERIMENT/e8_fullperm/
         calib/<stn>_<pair>.npz          observed stage one: Cp/3, beta, candidates (regenerated)
         chunks/*.csv                    per (analysis, permutation) rows, as computed
                                         (calib/ and chunks/ are written by a full run)
         records_fullperm.csv.gz         all rows b = 0..B: candidates, their T and inner p, T_ab
         candidates.csv, analyses.csv    the tables; results.json the summary
         season_labels.json              season label of each super-block
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
import run_e8_hourly as H  # noqa: E402
from dots.hourly import block_moments, prefix, domi_diff_curve, block_perm_order  # noqa: E402
from dots.domi import ranks01  # noqa: E402
from dots import pelt as P  # noqa: E402
from dots.perm import ge  # noqa: E402

E8 = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8")
OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8_fullperm")
SEED = 20260925
K_IN = 99            # inner replicas for the stage-two studentization
SUPER1 = H.SUPER     # 6 intervals = 12 weeks (stage one, and the outer permutation)
SUPER2 = 3           # 3 intervals = 6 weeks (stage two inner permutation)
HALF = 26            # +/- one year of two-week intervals
BETAS = np.geomspace(5, 40000, 80)
CHUNK = 20


# ----------------------------------------------------------------------------- pipeline pieces
def season_of_month(m):
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2}.get(m, 3)


def load(stn, u, v):
    z = np.load(H.ANOM)
    a, b = z[f"{stn}|{u}"], z[f"{stn}|{v}"]
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok], z[f"{stn}|tm"][ok]


def stage2_T(ua, ub, c, edges, B, rng, K=K_IN):
    """Stage-two statistic of run_e8_retest.py at candidate c (K inner replicas).
    Returns (T, inner p) or None if the window is too short (as in run_e8_hourly.py)."""
    lo, hi = max(0, c - HALF), min(B, c + HALF)
    if hi - lo < 12:
        return None
    sl = slice(edges[lo], edges[hi])
    Pw, Bw, FX, FY, J, ew = H.build(ua[sl], ub[sl])
    ts, obs = domi_diff_curve(Pw, Bw, lo=3)
    mx, my, mj, cnt = block_moments(FX, FY, J, ew)
    R = np.empty((K, len(ts)))
    for k in range(K):
        o = block_perm_order(Bw, SUPER2, rng)
        R[k] = domi_diff_curve(prefix(mx[o], my[o], mj[o], cnt[o]), Bw, lo=3)[1]
    A = np.vstack([obs[None, :], R])
    mu, sd = A.mean(0), A.std(0) + 1e-12
    T = [float(np.nanmax((A[i] - mu) / sd)) for i in range(K + 1)]
    return T[0], (1 + sum(ge(t, T[0]) for t in T[1:])) / (K + 1)


def outer_order(labels, B, rng):
    """Season-stratified permutation of the super-blocks -> interval order of length B."""
    nb = len(labels)
    order = np.arange(nb)
    for s in range(4):
        g = np.flatnonzero(labels == s)
        if len(g) > 1:
            order[g] = g[rng.permutation(len(g))]
    idx = np.concatenate([np.arange(k * SUPER1, (k + 1) * SUPER1) for k in order])
    return np.concatenate([idx, np.arange(nb * SUPER1, B)])


def analyses():
    d = json.load(open(os.path.join(E8, "results.json")))
    return [(i, r["stn"], r["pair"]) for i, r in enumerate(d["results"])], d


def calib_path(stn, pair):
    return os.path.join(OUT, "calib", f"{stn}_{pair}.npz")


# ----------------------------------------------------------------------------- setup (observed)
def _setup(args):
    ai, stn, pair, beta_stars = args
    u, v = pair.split("-")
    a, b, tm = load(stn, u, v)
    Pr, B, FX, FY, J, edges = H.build(a, b)
    rng = np.random.default_rng(20260914 + zlib.crc32((stn + u + v).encode()) % 9999)
    C = H.cost_matrix(Pr, B)
    Cp = np.zeros_like(C)
    for _ in range(3):                                   # the same three copies as run_e8_hourly.py
        Cp += H.cost_matrix(H.permuted_prefix(FX, FY, J, edges, B, rng), B)
    Cp3 = Cp / 3
    # beta regenerated from the 20 stored per-copy beta* (PELT is monotone in the penalty):
    fa = np.array([np.mean([bs > bb for bs in beta_stars]) for bb in BETAS])
    ok = np.flatnonzero(fa <= H.ALPHA)
    beta = float(BETAS[ok[0]]) if len(ok) else float(BETAS[-1])
    cps = P.pelt_from_costs(np.where(np.isfinite(C), C - Cp3, np.inf), beta)
    nb = B // SUPER1
    labels = np.array([season_of_month(int(tm[edges[k * SUPER1 + SUPER1 // 2]][4:6]))
                       for k in range(nb)])
    np.savez(calib_path(stn, pair), Cp3=Cp3, beta=beta, cps=np.array(cps, int), labels=labels,
             B=B, edges=edges)
    return ai, labels


# ----------------------------------------------------------------------------- permutation chunks
_CACHE = {}


def _prep(ai, stn, pair):
    if ai in _CACHE:
        return _CACHE[ai]
    u, v = pair.split("-")
    a, b, _ = load(stn, u, v)
    ua, ub = ranks01(a)[:, 0], ranks01(b)[:, 0]          # the data: tie-free global ranks
    cz = np.load(calib_path(stn, pair))
    B, edges = int(cz["B"]), cz["edges"]
    FX, FY, J = H.features(ua, ub)
    mom = block_moments(FX, FY, J, edges)
    _CACHE.clear()                                        # one analysis per worker at a time
    _CACHE[ai] = (ua, ub, mom, cz["Cp3"], float(cz["beta"]), [int(c) for c in cz["cps"]], B, edges)
    return _CACHE[ai]


def run_one(ai, stn, pair, b, labels):
    ua, ub, (mx, my, mj, cnt), Cp3, beta, cps_obs, B, edges = _prep(ai, stn, pair)
    if b == 0:
        o = np.arange(B)
        cps = cps_obs
    else:
        o = outer_order(labels, B, np.random.default_rng([SEED, b]))
        Cb = H.cost_matrix(prefix(mx[o], my[o], mj[o], cnt[o]), B)
        with np.errstate(invalid="ignore"):              # inf - inf below the diagonal
            cps = P.pelt_from_costs(np.where(np.isfinite(Cb), Cb - Cp3, np.inf), beta)
    hidx = np.concatenate([np.arange(edges[k], edges[k + 1]) for k in o])
    pa, pb = ua[hidx], ub[hidx]
    out = []
    for c in cps:
        r = stage2_T(pa, pb, int(c), edges, B, np.random.default_rng([SEED, ai, b, int(c)]))
        if r is not None:
            out.append((int(c), r[0], r[1]))
    Tmax = max((t for _, t, _ in out), default=-np.inf)
    return dict(ai=ai, stn=stn, pair=pair, b=b, n_cand=len(out),
                cands=";".join(f"{c}:{t:.6f}:{p:.4f}" for c, t, p in out), Tmax=Tmax)


FIELDS = ["ai", "stn", "pair", "b", "n_cand", "cands", "Tmax"]


def _chunk(args):
    ai, stn, pair, b0, b1, labels, path = args
    t0 = time.time()
    rows = [run_one(ai, stn, pair, b, labels) for b in range(b0, b1)]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return ai, b0, b1, time.time() - t0


# ----------------------------------------------------------------------------- multiplicity
def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.maximum.accumulate(np.minimum(1, (m - np.arange(m)) * p[o]))
    out = np.empty(m); out[o] = adj
    return out


def by(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    hm = np.sum(1.0 / np.arange(1, m + 1))
    q = p[o] * m * hm / np.arange(1, m + 1)
    q = np.minimum(1, np.minimum.accumulate(q[::-1])[::-1])
    out = np.empty(m); out[o] = q
    return out


def summarize(B, jobs, d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(OUT, "chunks", "*.csv"))):
        for r in csv.DictReader(open(f)):
            rows[(int(r["ai"]), int(r["b"]))] = r
    rows = list(rows.values())
    A = len(jobs)
    Tab = np.full((A, B + 1), np.nan)
    cand_obs = {}
    for r in rows:
        ai, b = int(r["ai"]), int(r["b"])
        if b > B:
            continue
        Tab[ai, b] = round(float(r["Tmax"]), 6)          # the precision of the per-candidate T
        if b == 0:
            cand_obs[ai] = [(int(c), float(t), float(p)) for c, t, p in
                            (x.split(":") for x in r["cands"].split(";") if x)]
    missing = np.argwhere(np.isnan(Tab))
    if len(missing):
        raise SystemExit(f"{len(missing)} (analysis, b) cells missing, e.g. {missing[:5].tolist()}")
    with gzip.open(os.path.join(OUT, "records_fullperm.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted((r for r in rows if int(r["b"]) <= B),
                           key=lambda r: (int(r["ai"]), int(r["b"]))))
    # nominal stage-two p-values (run_e8_retest.py)
    k999 = {(r["stn"], r["pair"], int(r["interval"])): r["p_block"]
            for r in json.load(open(os.path.join(E8, "retest_K999.json")))["results"]}
    k9999 = {(r["stn"], r["pair"], int(r["interval"])): r["p_dep"]
             for r in json.load(open(os.path.join(E8, "retest_K9999.json")))["results"]
             if r.get("K") == 9999}
    dates = {(r["stn"], r["pair"], b_["interval"]): b_["date"]
             for r in d["results"] for b_ in r["breaks"]}
    perm = Tab[:, 1:]
    # rank p-values within analysis over b = 0..B (for the joint min-p)
    q = np.empty_like(Tab)
    for a in range(A):
        q[a] = (Tab[a][None, :] >= Tab[a][:, None]).sum(1) / (B + 1)
    Q = q.min(0)
    p_global_joint = float(np.mean(Q <= Q[0]))
    # secondary global statistic: number of analyses in which stage one returns any candidate
    Nf = np.isfinite(Tab).sum(0)
    p_global_nfire = float(np.mean(ge(Nf, Nf[0])))
    # DESCRIPTIVE, not exact: stage-two T of the observed candidate against the pooled T of the
    # candidates stage one selected on permuted records of all analyses (selection-conditional)
    pooled = np.sort(perm[np.isfinite(perm)])
    ana, cand = [], []
    for ai, stn, pair in jobs:
        cs = cand_obs.get(ai, [])
        fire = int(np.sum(np.isfinite(perm[ai])))
        for c, T, pin in cs:
            padj = (1 + np.sum(ge(perm[ai], T))) / (B + 1)
            qa = np.sum(Tab[ai] >= T) / (B + 1)
            pcond = (1 + np.sum(ge(pooled, T))) / (1 + len(pooled))
            cand.append(dict(ai=ai, stn=stn, station=H.NAMES[stn], pair=pair, interval=c,
                             date=dates.get((stn, pair, c), ""), T_obs=round(T, 4),
                             p_inner_K99=pin, p_nominal_K999=k999.get((stn, pair, c)),
                             p_nominal_K9999=k9999.get((stn, pair, c)),
                             p_adj_wy=float(padj),
                             p_joint_minp=float(np.mean(Q <= qa)),
                             p_cond_pooled_descriptive=float(pcond)))
        pa = min((x["p_adj_wy"] for x in cand if x["ai"] == ai), default=1.0)
        ana.append(dict(ai=ai, stn=stn, station=H.NAMES[stn], pair=pair, n_cand_obs=len(cs),
                        T_max_obs=(max(t for _, t, _ in cs) if cs else None),
                        perm_fire_rate=fire / B, p_analysis=float(pa)))
    pa = np.array([x["p_analysis"] for x in ana])
    for x, h, y in zip(ana, holm(pa), by(pa)):
        x["holm"], x["by"] = float(h), float(y)
    top = {}
    for x in cand:
        if x["ai"] not in top or x["T_obs"] > top[x["ai"]]["T_obs"]:
            top[x["ai"]] = x
    for x in cand:
        a_ = ana[x["ai"]]
        is_top = top[x["ai"]] is x
        x["holm_analysis"] = a_["holm"] if is_top else None
        x["by_analysis"] = a_["by"] if is_top else None
    cand.sort(key=lambda x: (x["p_adj_wy"], -x["T_obs"]))
    res = dict(config=dict(B=B, K_in=K_IN, seed=SEED, super_blocks_outer=SUPER1,
                           super_blocks_inner=SUPER2, window_half_intervals=HALF,
                           outer="season-stratified permutation of 12-week super-blocks",
                           n_analyses=A, n_candidates_obs=len(cand)),
               p_global_joint_minp=p_global_joint,
               p_global_bonferroni=float(min(1.0, A * pa.min())),
               p_global_nfire=p_global_nfire, n_fire_obs=int(Nf[0]),
               n_fire_perm_mean=float(Nf[1:].mean()), n_fire_perm_max=int(Nf[1:].max()),
               n_pooled_perm_candidates=int(len(pooled)),
               n_padj_le_05=int(sum(x["p_adj_wy"] <= 0.05 for x in cand)),
               n_holm_le_05=int(sum(x["holm"] <= 0.05 for x in ana)),
               n_by_le_10=int(sum(x["by"] <= 0.10 for x in ana)),
               n_joint_le_05=int(sum(x["p_joint_minp"] <= 0.05 for x in cand)),
               median_perm_fire_rate=float(np.median([x["perm_fire_rate"] for x in ana])),
               candidates=cand, analyses=ana)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    for name, tab in (("candidates.csv", cand), ("analyses.csv", ana)):
        with open(os.path.join(OUT, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tab[0].keys()))
            w.writeheader(); w.writerows(tab)
    print(f"\nB={B}  global joint min-p={p_global_joint:.4f}  Bonferroni={res['p_global_bonferroni']:.4f}"
          f"  median permuted firing rate={res['median_perm_fire_rate']:.3f}")
    print(f"  {'station':<10}{'pair':<7}{'date':<10}{'T':>7}{'p_nom':>8}{'p_adj':>8}{'joint':>8}"
          f"{'holm':>8}{'BY':>8}")
    for x in cand:
        f_ = lambda v: f"{v:8.3f}" if v is not None else f"{'':>8}"
        pn = x["p_nominal_K9999"] if x["p_nominal_K9999"] is not None else x["p_nominal_K999"]
        print(f"  {x['station']:<10}{x['pair']:<7}{x['date']:<10}{x['T_obs']:7.2f}{f_(pn)}"
              f"{f_(x['p_adj_wy'])}{f_(x['p_joint_minp'])}{f_(x['holm_analysis'])}{f_(x['by_analysis'])}")
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=199)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--time-one", action="store_true", help="time one permuted pipeline run")
    ap.add_argument("--analyses", type=str, default="",
                    help="lo:hi -- run only analyses lo..hi-1 (to split the work across hosts that "
                         "share this file system); the summary is skipped until all are present")
    a_ = ap.parse_args()
    os.makedirs(os.path.join(OUT, "calib"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "chunks"), exist_ok=True)
    jobs, d = analyses()
    if a_.summarize_only:
        summarize(a_.B, jobs, d); return
    t0 = time.time()
    # ---- setup: regenerate the observed stage one
    bstars = {}
    for r in csv.DictReader(open(os.path.join(E8, "records_stage1_beta_perm.csv"))):
        bstars.setdefault((r["stn"], r["pair"]), []).append(float(r["beta_star"]))
    todo = [(ai, stn, pair, bstars[(stn, pair)])
            for ai, stn, pair in jobs if not os.path.exists(calib_path(stn, pair))]
    with Pool(a_.procs) as pool:
        for ai, _ in pool.imap_unordered(_setup, todo):
            print(f"  setup {jobs[ai][1]} {jobs[ai][2]} ok", flush=True)
    labs = [np.load(calib_path(s, p))["labels"] for _, s, p in jobs]
    ref = np.array([np.bincount([l[k] for l in labs], minlength=4).argmax() for k in range(len(labs[0]))])
    n_dis = sum(int((l != ref).any()) for l in labs)
    print(f"setup done ({time.time()-t0:.0f}s); season labels of the {len(ref)} super-blocks: "
          f"{''.join(map(str, ref))}; analyses whose own labels differ from the common ones: {n_dis}",
          flush=True)
    json.dump(dict(season_labels=ref.tolist(), n_analyses_with_differing_own_labels=n_dis,
                   group_sizes=np.bincount(ref, minlength=4).tolist()),
              open(os.path.join(OUT, "season_labels.json"), "w"), indent=1)
    if a_.time_one:
        ai, stn, pair = jobs[0]
        t1 = time.time(); run_one(ai, stn, pair, 1, ref)
        print(f"one permuted pipeline run: {time.time()-t1:.2f}s"); return
    # ---- permutations: b = 0 (observed) and b = 1..B, in chunks, analyses interleaved
    work = []
    starts = [0] + list(range(1, a_.B + 1, CHUNK))       # fixed chunk grid [1+20k, 21+20k):
    for b0 in starts:                                     # a larger --B reuses every chunk on disk
        b1 = 1 if b0 == 0 else b0 + CHUNK
        lo_, hi_ = (map(int, a_.analyses.split(":")) if a_.analyses else (0, len(jobs)))
        for ai, stn, pair in jobs[lo_:hi_]:
            path = os.path.join(OUT, "chunks", f"a{ai:02d}_b{b0:05d}_{b1:05d}.csv")
            if not os.path.exists(path):
                work.append((ai, stn, pair, b0, b1, ref, path))
    print(f"{len(work)} chunks to run", flush=True)
    done = 0
    with Pool(a_.procs, maxtasksperchild=20) as pool:
        for ai, b0, b1, dt in pool.imap_unordered(_chunk, work):
            done += 1
            if done % 10 == 0 or done == len(work):
                print(f"  {done}/{len(work)} chunks ({time.time()-t0:.0f}s; last {dt:.0f}s for "
                      f"{b1-b0} runs)", flush=True)
    if not a_.analyses:
        summarize(a_.B, jobs, d)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
