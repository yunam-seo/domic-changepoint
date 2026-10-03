#!/usr/bin/env python
"""Marginal-scale test for each candidate break of the weather analysis (Sections 4.7, 6.7).

Stage 2 re-tests each candidate for a dependence change. Not passing it is not evidence that a
break is marginal-only: it may be a dependence change too small to see. Testing the margins
directly turns that undetermined outcome into a classification (Section 4.7).

For each candidate break, a block-permutation test of a change in the marginal scale at the same
location, run separately on X and on Y, using the same block structure as the dependence test:
    statistic = |log sd_left - log sd_right|, weighted as sqrt(n_L n_R / n)
The two tests then classify each break:
    dependence passes,     margins stable   -> a dependence change
    dependence passes,     margins changed  -> both change
    dependence not passed, margins changed  -> consistent with a marginal-driven break
    dependence not passed, margins stable   -> undetermined; too weak to attribute

Run: python 00_SRC/run_e8_marginal.py --procs 12
Writes 04_DAOU/EXPERIMENT/e8/{marginal_diagnostic.json, records_marginal_perm.csv}.
"""
from __future__ import annotations
import json, os, sys, time
from multiprocessing import Pool
import zlib

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.hourly import block_perm_order  # noqa: E402
from dots.perm import ge  # noqa: E402

E8 = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8")
ANOM = os.path.join(ROOT, "02_MART", "WEATHER_HOURLY_ANOM.npz")
UNIT, K, SUPER = 336, 999, 3


def scale_curve(v, edges, lo=3):
    """Per-interval sums of v and v**2 and the interval counts (inputs of curve_from); lo is unused."""
    B = len(edges) - 1
    s1 = np.array([np.nansum(v[edges[k]:edges[k + 1]]) for k in range(B)])
    s2 = np.array([np.nansum(v[edges[k]:edges[k + 1]] ** 2) for k in range(B)])
    cnt = np.array([edges[k + 1] - edges[k] for k in range(B)])
    return s1, s2, cnt


def curve_from(s1, s2, cnt, lo=3):
    """Weighted |log sd_L - log sd_R| over interval-level split points, from interval sums."""
    B = len(cnt)
    c1 = np.concatenate([[0], np.cumsum(s1)]); c2 = np.concatenate([[0], np.cumsum(s2)])
    cn = np.concatenate([[0], np.cumsum(cnt)])
    n = cn[B]
    ts = np.arange(lo, B - lo + 1)
    out = np.empty(len(ts))
    for i, t in enumerate(ts):
        nl, nr = cn[t], n - cn[t]
        vl = c2[t] / nl - (c1[t] / nl) ** 2
        vr = (c2[B] - c2[t]) / nr - ((c1[B] - c1[t]) / nr) ** 2
        out[i] = np.sqrt(nl * nr / n) * abs(np.log(max(vl, 1e-12)) - np.log(max(vr, 1e-12))) / 2
    return ts, out


def _job(args):
    stn, pair, c, date = args
    u, v = pair.split("-")
    z = np.load(ANOM)
    a, b = z[f"{stn}|{u}"], z[f"{stn}|{v}"]
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    ef = np.arange(0, len(a) + 1, UNIT)
    if ef[-1] != len(a):
        ef = np.append(ef, len(a))
    B = len(ef) - 1
    lo, hi = max(0, c - 26), min(B, c + 26)
    sl = slice(ef[lo], ef[hi])
    out = {}
    perm_T = {}
    for name, series in ((u, a[sl]), (v, b[sl])):
        e2 = np.arange(0, len(series) + 1, UNIT)
        if e2[-1] != len(series):
            e2 = np.append(e2, len(series))
        s1, s2, cnt = scale_curve(series, e2)
        ts, obs = curve_from(s1, s2, cnt)
        rng = np.random.default_rng(9090 + c + zlib.crc32(name.encode()) % 97)
        R = np.empty((K, len(ts)))
        for k in range(K):
            o = block_perm_order(len(cnt), SUPER, rng)
            R[k] = curve_from(s1[o], s2[o], cnt[o])[1]
        A = np.vstack([obs[None, :], R])
        mu, sd = A.mean(0), A.std(0) + 1e-12
        T = [float(np.nanmax((A[i] - mu) / sd)) for i in range(K + 1)]
        out[name] = round((1 + sum(ge(t, T[0]) for t in T[1:])) / (K + 1), 4)
        perm_T[name] = T
    return dict(stn=stn, pair=pair, date=date, interval=c,
                p_marg_x=out[u], p_marg_y=out[v], p_marg_min=min(out.values()),
                _T={n: " ".join(f"{v:.6f}" for v in t) for n, t in perm_T.items()})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=12)
    a_ = ap.parse_args()
    d = json.load(open(os.path.join(E8, "results.json")))
    retest = {(r["stn"], r["pair"], r["interval"]): r for r in
            json.load(open(os.path.join(E8, "retest_K999.json")))["results"]}
    jobs = [(r["stn"], r["pair"], b["interval"], b["date"]) for r in d["results"] for b in r["breaks"]]
    names = {r["stn"]: r["station"] for r in d["results"]}
    t0 = time.time()
    with Pool(a_.procs, maxtasksperchild=4) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    for r in res:
        k = (r["stn"], r["pair"], r["interval"])
        r["p_dep"] = retest[k]["p_block"]; r["bh_q_dep"] = retest[k]["bh_q"]
        dep = r["bh_q_dep"] <= 0.10
        marg = r["p_marg_min"] <= 0.05
        r["class"] = ("dependence change" if dep and not marg else
                      "both change" if dep and marg else
                      "marginal-driven" if marg else "undetermined")
    res.sort(key=lambda r: r["p_dep"])
    print(f"  {'station':<11}{'pair':<7}{'date':<10}{'p_dep':>7}{'q_dep':>7}{'p_marg':>8}  class")
    for r in res:
        print(f"  {names[r['stn']]:<11}{r['pair']:<7}{r['date']:<10}{r['p_dep']:>7.3f}"
              f"{r['bh_q_dep']:>7.3f}{r['p_marg_min']:>8.3f}  {r['class']}")
    from collections import Counter
    cnt = Counter(r["class"] for r in res)
    print("\n  " + " | ".join(f"{k}: {v}" for k, v in cnt.most_common()))
    import csv as _csv
    with open(os.path.join(E8, "records_marginal_perm.csv"), "w", newline="") as f:
        rows = [dict(stn=r["stn"], pair=r["pair"], date=r["date"], test=n, T=t)
                for r in res for n, t in r.pop("_T", {}).items()]
        w = _csv.DictWriter(f, fieldnames=["stn", "pair", "date", "test", "T"])
        w.writeheader(); w.writerows(rows)
    json.dump(dict(config=dict(K=K, unit_hours=UNIT, super_blocks=SUPER), counts=dict(cnt), results=res),
              open(os.path.join(E8, "marginal_diagnostic.json"), "w"), indent=1)
    print(f"  ({time.time()-t0:.0f}s)")
