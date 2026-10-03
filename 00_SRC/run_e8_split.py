#!/usr/bin/env python
"""Split-sample, selection-valid version of the two-stage hourly weather analysis (Section 6.7).

Purpose
-------
In Section 6.7 stage one (Holevo partitioning) and stage two (the block-permutation DOMI test)
see the same observations. Propositions P1 / P5 assume a window fixed in advance, and the
stage-one selection is not invariant under the block permutations of stage two, so the
stage-two p-values of Section 6.7 are nominal. This script makes them valid by sample
splitting: the candidates and their windows are chosen on one half of each record and tested on the
other, disjoint half.

Split
-----
Each station-pair record (hourly anomalies, hours with both variables finite, as in
run_e8_hourly.py) is cut into consecutive six-week blocks of 1008 hours, the stage-two block length
of run_e8_retest.py. Blocks 1, 3, 5, ... form set A and blocks 2, 4, 6, ... set B. The first and
last 24 hours of every block are dropped, leaving 960 hours per block and a gap of at least two days
between any retained hour of A and any retained hour of B. A final block shorter than 1008 hours is
dropped.

Stage one (on the selection half)
---------------------------------
Holevo partitioning exactly as run_e8_hourly.py, applied to the concatenated retained hours of the half:
ranks and D = 8 random features (seeds 2026 / 2027) over the half, energy-weighted von Neumann
segment costs on a grid of 320-hour units (three per retained block, tiling it exactly; the
full-record analysis uses 336-hour units), the costs centered by three block-permuted copies,
super-blocks of six units (two retained blocks, twelve weeks of data, as in stage one of the full
analysis), the penalty the smallest value on the same geometric grid at which at most 5% of 20
block-permuted copies return any break.

Stage two (on the test half)
----------------------------
Each stage-one cut is mapped to calendar time: the midpoint between the last selection-half hour
before the cut and the first after it (the midpoint of the intervening test block when the cut
falls on a block boundary). The window consists of the test-half blocks lying entirely within
+/- 8736 hours (52 weeks) of that time, the +/- 1 year window of run_e8_retest.py; a window with
fewer than four blocks is not tested. On the window: the weighted DOMI-difference curve over the
320-hour grid (ranks and features recomputed within the window, minimum segment one block), block
permutation of the order of whole retained blocks (six weeks of data, the stage-two super-block),
symmetric studentization over all K + 1 curves, permutation p-value; and the marginal-scale test of
run_e8_marginal.py (|log sd_L - log sd_R|, weighted) on each variable with the same blocks.
K = 9999. Classification as in run_e8_marginal.py: dependence change at Benjamini-Hochberg
q <= 0.10 within the direction, marginal change at min(p_X, p_Y) <= 0.05. Benjamini-Yekutieli
q-values are reported alongside.

Both directions are run: select on A / test on B, and select on B / test on A. A candidate passes
in both directions (key passed_both) when the two directions produce candidates for the same
station and pair within 12 weeks of each other and both pass.

Because the test half is disjoint from the selection half and separated from it by at least two
days, the stage-two p-value is conditionally valid given the selection to the extent that the
two halves are independent -- exactly under independence across the gaps, approximately under the
serial dependence that remains across 48 hours (the same approximation block permutation already
makes at block boundaries).

Run
---
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_e8_split.py --procs 12

Inputs:  02_MART/WEATHER_HOURLY_ANOM.npz
Outputs: 04_DAOU/EXPERIMENT/e8_split/results.json         (per candidate and per direction)
         04_DAOU/EXPERIMENT/e8_split/results.csv          (one row per candidate)
         04_DAOU/EXPERIMENT/e8_split/records_stage1_beta_perm.csv.gz  (per permuted copy: the
                         smallest penalty with no break; regenerates every stage-one penalty)
         04_DAOU/EXPERIMENT/e8_split/records_stage2_perm.csv.gz (per candidate and test: the
                         observed statistic and all K replica maxima; regenerates every p-value)
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.hourly import features, block_moments, prefix, domi_diff_curve, block_perm_order  # noqa: E402
from dots import pelt as P  # noqa: E402
from dots.perm import ge  # noqa: E402
from run_e8_hourly import cost_matrix, NAMES, PAIRS  # noqa: E402
from run_e8_marginal import curve_from  # noqa: E402
from run_e8_summary import step_up_q  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8_split")
ANOM = os.path.join(ROOT, "02_MART", "WEATHER_HOURLY_ANOM.npz")
BLOCK, TRIM = 1008, 24                 # six-week blocks; drop 24 h at each end
KEEP = BLOCK - 2 * TRIM                # 960 retained hours per block
UNIT = 320                             # grid unit: three per retained block
UPB = KEEP // UNIT                     # units per block (3)
SUPER1 = 2 * UPB                       # stage-one super-block: two retained blocks (6 units)
HALF_WIN = 8736                        # +/- 52 weeks
MIN_BLOCKS = 4
K = 9999
ALPHA = 0.05
NPERM = 20
BETAS = list(np.geomspace(5, 40000, 80))
MATCH_H = 12 * 168                     # 12 weeks, for cross-direction matching
PAPER = {("146", "ta-hm"): "20191021", ("119", "hm-ws"): "20190826"}


def load_pair(stn, u, v):
    z = np.load(ANOM)
    a, b = z[f"{stn}|{u}"], z[f"{stn}|{v}"]
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok], z[f"{stn}|tm"][ok]


def split_index(n):
    """Original indices of the retained hours of set A (odd blocks) and set B (even blocks),
    each as a list of per-block index arrays."""
    nb = n // BLOCK
    blocks = [np.arange(k * BLOCK + TRIM, k * BLOCK + BLOCK - TRIM) for k in range(nb)]
    return {"A": blocks[0::2], "B": blocks[1::2]}


def unit_edges(m):
    return np.arange(0, m + 1, UNIT)


# ---------------------------------------------------------------- stage one
def _stage1(args):
    stn, u, v, sel = args
    a, b, tm = load_pair(stn, u, v)
    blocks = split_index(len(a))[sel]
    idx = np.concatenate(blocks)
    x, y = a[idx], b[idx]
    edges = unit_edges(len(x))
    B = len(edges) - 1
    FX, FY, J = features(x, y)
    mx, my, mj, cnt = block_moments(FX, FY, J, edges)
    Pr = prefix(mx, my, mj, cnt)
    rng = np.random.default_rng([20260925, zlib.crc32((stn + u + v + sel).encode())])

    def perm_prefix():
        o = block_perm_order(B, SUPER1, rng)
        return prefix(mx[o], my[o], mj[o], cnt[o])

    C = cost_matrix(Pr, B)
    Cp = sum(cost_matrix(perm_prefix(), B) for _ in range(3))
    Cbc = np.where(np.isfinite(C), C - Cp / 3, np.inf)
    beta_stars = []
    fa = {bb: 0 for bb in BETAS}
    for _ in range(NPERM):
        Cq = cost_matrix(perm_prefix(), B)
        Cqp = sum(cost_matrix(perm_prefix(), B) for _ in range(3))
        Cq = np.where(np.isfinite(Cq), Cq - Cqp / 3, np.inf)
        ks = {bb: len(P.pelt_from_costs(Cq, bb)) for bb in BETAS}
        for bb in BETAS:
            fa[bb] += ks[bb] > 0
        beta_stars.append(next((bb for bb in BETAS if ks[bb] == 0), float("inf")))
    beta = next((bb for bb in BETAS if fa[bb] / NPERM <= ALPHA), BETAS[-1])
    cuts = P.pelt_from_costs(Cbc, beta)
    cands = []
    for c in cuts:
        h = c * UNIT
        t_star = 0.5 * (idx[h - 1] + idx[h])          # calendar position (hours) of the cut
        cands.append(dict(unit=int(c), t_star=float(t_star),
                          date=str(tm[int(round(t_star))])[:8]))
    return dict(stn=stn, pair=f"{u}-{v}", select=sel, n_select=int(len(x)), n_units=int(B),
                beta=float(beta), beta_fa=fa[beta] / NPERM, candidates=cands,
                beta_stars=beta_stars)


# ---------------------------------------------------------------- stage two
def _perm_p(obs, R):
    A = np.vstack([obs[None, :], R])
    mu, sd = A.mean(0), A.std(0) + 1e-12
    T = np.nanmax((A - mu) / sd, axis=1)
    return (1 + int(np.sum(ge(T[1:], T[0])))) / len(T), T


def _stage2(args):
    stn, u, v, sel, cand, seed = args
    a, b, tm = load_pair(stn, u, v)
    test = "B" if sel == "A" else "A"
    blocks = [bk for bk in split_index(len(a))[test]
              if bk[0] >= cand["t_star"] - HALF_WIN and bk[-1] < cand["t_star"] + HALF_WIN]
    base = dict(stn=stn, pair=f"{u}-{v}", select=sel, test=test, date=cand["date"],
                t_star=cand["t_star"], n_blocks=len(blocks))
    if len(blocks) < MIN_BLOCKS:
        return dict(base, tested=False)
    idx = np.concatenate(blocks)
    x, y = a[idx], b[idx]
    edges = unit_edges(len(x))
    B = len(edges) - 1
    FX, FY, J = features(x, y)
    mx, my, mj, cnt = block_moments(FX, FY, J, edges)
    ts, obs = domi_diff_curve(prefix(mx, my, mj, cnt), B, lo=UPB)
    rng = np.random.default_rng([seed, 1])
    R = np.empty((K, len(ts)))
    for k in range(K):
        o = block_perm_order(B, UPB, rng)
        R[k] = domi_diff_curve(prefix(mx[o], my[o], mj[o], cnt[o]), B, lo=UPB)[1]
    p_dep, Td = _perm_p(obs, R)
    out = dict(base, tested=True, window=f"{str(tm[idx[0]])[:8]}-{str(tm[idx[-1]])[:8]}",
               p_dep=float(p_dep), T_dep=" ".join(f"{t:.6f}" for t in Td))
    for name, s, tag in ((u, x, "x"), (v, y, "y")):
        s1 = np.add.reduceat(s, edges[:-1]); s2 = np.add.reduceat(s ** 2, edges[:-1])
        c_ = np.diff(edges)
        _, ob = curve_from(s1, s2, c_, lo=UPB)
        r2 = np.random.default_rng([seed, 2 if tag == "x" else 3])
        Rm = np.empty((K, len(ob)))
        for k in range(K):
            o = block_perm_order(B, UPB, r2)
            Rm[k] = curve_from(s1[o], s2[o], c_[o], lo=UPB)[1]
        pm, Tm = _perm_p(ob, Rm)
        out[f"p_marg_{tag}"] = float(pm)
        out[f"T_marg_{tag}"] = " ".join(f"{t:.6f}" for t in Tm)
    out["p_marg_min"] = min(out["p_marg_x"], out["p_marg_y"])
    return out


def classify(rows):
    tested = [r for r in rows if r["tested"]]
    if not tested:
        return
    p = np.array([r["p_dep"] for r in tested])
    H = float(np.sum(1.0 / np.arange(1, len(p) + 1)))
    bh, by = step_up_q(p), step_up_q(p, H)
    for r, q1, q2 in zip(tested, bh, by):
        r["bh_q"], r["by_q"] = round(float(q1), 4), round(float(q2), 4)
        d, m = q1 <= 0.10, r["p_marg_min"] <= 0.05
        r["cls"] = ("dependence change" if d and not m else "both change" if d and m
                    else "marginal-driven" if m else "undetermined")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    a_ = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    z = np.load(ANOM)
    stns = sorted({k.split("|")[0] for k in z.files})
    t0 = time.time()
    jobs1 = [(s, u, v, sel) for sel in ("A", "B") for s in stns for (u, v) in PAIRS]
    with Pool(a_.procs, maxtasksperchild=2) as pool:
        s1 = pool.map(_stage1, jobs1, chunksize=1)
        t1 = time.time() - t0
        print(f"stage one: {len(jobs1)} analyses, {sum(len(r['candidates']) for r in s1)} "
              f"candidates ({t1:.0f}s)", flush=True)
        jobs2 = []
        for r in s1:
            u, v = r["pair"].split("-")
            for c in r["candidates"]:
                seed = zlib.crc32(f"{r['stn']}{r['pair']}{r['select']}{c['unit']}".encode())
                jobs2.append((r["stn"], u, v, r["select"], c, seed))
        s2 = pool.map(_stage2, jobs2, chunksize=1)
    wall = time.time() - t0
    with gzip.open(os.path.join(OUT, "records_stage1_beta_perm.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stn", "pair", "select", "copy", "beta_star"])
        w.writeheader()
        w.writerows([dict(stn=r["stn"], pair=r["pair"], select=r["select"], copy=i, beta_star=bs)
                     for r in s1 for i, bs in enumerate(r.pop("beta_stars"))])
    with gzip.open(os.path.join(OUT, "records_stage2_perm.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stn", "pair", "select", "date", "test", "p", "T"])
        w.writeheader()
        for r in s2:
            if not r["tested"]:
                continue
            for tag, pk, tk in (("dependence", "p_dep", "T_dep"),
                                (f"marginal:{r['pair'].split('-')[0]}", "p_marg_x", "T_marg_x"),
                                (f"marginal:{r['pair'].split('-')[1]}", "p_marg_y", "T_marg_y")):
                w.writerow(dict(stn=r["stn"], pair=r["pair"], select=r["select"], date=r["date"],
                                test=tag, p=r[pk], T=r.pop(tk)))
    directions = {}
    for sel in ("A", "B"):
        rows = [r for r in s2 if r["select"] == sel]
        classify(rows)
        tested = [r for r in rows if r["tested"]]
        from collections import Counter
        directions[f"select_{sel}_test_{'B' if sel == 'A' else 'A'}"] = dict(
            n_candidates=len(rows), n_tested=len(tested),
            n_p_le_05=int(sum(r["p_dep"] <= 0.05 for r in tested)),
            n_bh_q_le_10=int(sum(r["bh_q"] <= 0.10 for r in tested)),
            n_by_q_le_10=int(sum(r["by_q"] <= 0.10 for r in tested)),
            n_by_q_le_05=int(sum(r["by_q"] <= 0.05 for r in tested)),
            class_counts=dict(Counter(r["cls"] for r in tested)))
    for r in s2:
        r["station"] = NAMES[r["stn"]]
    # cross-direction matching, and the two strongest candidates of the unrestricted full-record stage two
    is_dep_change = lambda r: r.get("cls") in ("dependence change", "both change")  # noqa: E731
    both = []
    for ra in [r for r in s2 if r["select"] == "A" and r["tested"]]:
        for rb in [r for r in s2 if r["select"] == "B" and r["tested"]]:
            if (ra["stn"], ra["pair"]) == (rb["stn"], rb["pair"]) and\
                    abs(ra["t_star"] - rb["t_star"]) <= MATCH_H:
                both.append(dict(stn=ra["stn"], station=NAMES[ra["stn"]], pair=ra["pair"],
                                 date_A=ra["date"], date_B=rb["date"],
                                 p_dep_AtoB=ra["p_dep"], p_dep_BtoA=rb["p_dep"],
                                 cls_AtoB=ra["cls"], cls_BtoA=rb["cls"],
                                 passed_both=bool(is_dep_change(ra) and is_dep_change(rb))))
    paper = {}
    for (stn, pair), date in PAPER.items():
        a, b, tm = load_pair(stn, *pair.split("-"))
        t_pub = int(np.searchsorted(tm, date + "0000"))
        near = [dict(select=r["select"], date=r["date"], tested=r["tested"],
                     p_dep=r.get("p_dep"), p_marg_min=r.get("p_marg_min"), bh_q=r.get("bh_q"),
                     by_q=r.get("by_q"), cls=r.get("cls"),
                     weeks_from_unrestricted=round((r["t_star"] - t_pub) / 168, 1))
                for r in s2 if (r["stn"], r["pair"]) == (stn, pair)]   # all, with distance
        paper[f"{NAMES[stn]} {pair} {date}"] = near
    res = dict(config=dict(block_hours=BLOCK, trim_hours=TRIM, unit_hours=UNIT,
                           stage1_super_units=SUPER1, stage2_super_units=UPB,
                           half_window_hours=HALF_WIN, min_blocks=MIN_BLOCKS, K=K, alpha=ALPHA,
                           n_beta_perm=NPERM, feature_seeds=[2026, 2027],
                           classification="dependence: BH q<=0.10 within direction; marginal: min p<=0.05",
                           match_hours=MATCH_H),
               stage1=[{k: v for k, v in r.items()} for r in s1],
               directions=directions, candidates=s2, matched_across_directions=both,
               unrestricted_candidates=paper,
               wall_clock_s=dict(stage1=round(t1, 1), total=round(wall, 1)))
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    keys = ["select", "test", "stn", "station", "pair", "date", "n_blocks", "tested", "window",
            "p_dep", "bh_q", "by_q", "p_marg_x", "p_marg_y", "p_marg_min", "cls"]
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(s2, key=lambda r: (r["select"], r.get("p_dep", 2))))
    for d, v in directions.items():
        print(d, v)
    for r in sorted([r for r in s2 if r["tested"]], key=lambda r: (r["select"], r["p_dep"]))[:40]:
        print(f"  {r['select']}->{r['test']} {r['station']:<10}{r['pair']:<7}{r['date']:<10}"
              f"p={r['p_dep']:.4f} q={r['bh_q']:.3f} BY={r['by_q']:.3f} pm={r['p_marg_min']:.4f} {r['cls']}")
    print("matched:", both)
    print("reference candidates:", json.dumps(paper, indent=1))
    print(f"wall clock {wall:.0f}s (stage one {t1:.0f}s)")


if __name__ == "__main__":
    main()
