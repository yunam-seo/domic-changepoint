#!/usr/bin/env python
"""Stage two of E8 for the candidate breaks with K = 999 permutations.

At K = 99 the smallest attainable permutation p-value is 0.01, which caps what any multiplicity
procedure can do: with 27 candidates the best possible Benjamini-Hochberg q is 0.27 at rank one.
K = 999 lowers the floor to 0.001 and makes FDR control informative. Stage one (the segmentation) is
unchanged and is read from e8/results.json; only stage two is recomputed.

Run:    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/run_e8_retest.py --procs 12
Writes: 04_DAOU/EXPERIMENT/e8/{retest_K999.json, records_retest_perm.csv}
"""
from __future__ import annotations
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.hourly import (features, block_moments, prefix, domi_diff_curve, block_perm_order)  # noqa: E402
from dots.perm import ge  # noqa: E402

E8 = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8")
ANOM = os.path.join(ROOT, "02_MART", "WEATHER_HOURLY_ANOM.npz")
UNIT, K, SUPER = 336, 999, 3


def build(x, y):
    edges = np.arange(0, len(x) + 1, UNIT)
    if edges[-1] != len(x):
        edges = np.append(edges, len(x))
    FX, FY, J = features(x, y)
    return prefix(*block_moments(FX, FY, J, edges)), len(edges) - 1, FX, FY, J, edges


def _job(args):
    stn, pair, c, date = args
    u, v = pair.split("-")
    z = np.load(ANOM)
    a, b = z[f"{stn}|{u}"], z[f"{stn}|{v}"]
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    edges_full = np.arange(0, len(a) + 1, UNIT)
    if edges_full[-1] != len(a):
        edges_full = np.append(edges_full, len(a))
    B = len(edges_full) - 1
    lo, hi = max(0, c - 26), min(B, c + 26)
    sl = slice(edges_full[lo], edges_full[hi])
    Pw, Bw, FX, FY, J, ew = build(a[sl], b[sl])
    ts, obs = domi_diff_curve(Pw, Bw, lo=3)
    mx, my, mj, cnt = block_moments(FX, FY, J, ew)
    rng = np.random.default_rng(4242 + c)
    R = np.empty((K, len(ts)))
    for k in range(K):
        o = block_perm_order(Bw, SUPER, rng)
        R[k] = domi_diff_curve(prefix(mx[o], my[o], mj[o], cnt[o]), Bw, lo=3)[1]
    A = np.vstack([obs[None, :], R])
    mu, sd = A.mean(0), A.std(0) + 1e-12
    T = [float(np.nanmax((A[i] - mu) / sd)) for i in range(K + 1)]
    p = (1 + sum(ge(t, T[0]) for t in T[1:])) / (K + 1)
    return dict(stn=stn, pair=pair, date=date, interval=c, p_block=round(p, 4),
                T_obs=T[0], T_rep=" ".join(f"{v:.6f}" for v in T[1:]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=12)
    a_ = ap.parse_args()
    d = json.load(open(os.path.join(E8, "results.json")))
    jobs = [(r["stn"], r["pair"], b["interval"], b["date"]) for r in d["results"] for b in r["breaks"]]
    print(f"re-testing {len(jobs)} candidate breaks at K={K}", flush=True)
    t0 = time.time()
    with Pool(a_.procs, maxtasksperchild=4) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    p = np.array([r["p_block"] for r in res]); m = len(p)
    order = np.argsort(p)
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    for rank, i in enumerate(order):
        res[i]["bh_q"] = round(float(q[rank]), 4)
    res.sort(key=lambda r: r["p_block"])
    names = {r["stn"]: r2["station"] for r2 in d["results"] for r in [r2]}
    print(f"\n  {'station':<11}{'pair':<7}{'date':<10}{'p':>8}{'BH q':>8}")
    for r in res[:12]:
        print(f"  {names.get(r['stn'], r['stn']):<11}{r['pair']:<7}{r['date']:<10}"
              f"{r['p_block']:>8.3f}{r['bh_q']:>8.3f}")
    import csv as _csv
    with open(os.path.join(E8, "records_retest_perm.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["stn", "pair", "date", "interval", "p_block",
                                           "T_obs", "T_rep"])
        w.writeheader()
        w.writerows([{k: r[k] for k in w.fieldnames} for r in res])
    for r in res:
        r.pop("T_obs", None); r.pop("T_rep", None)
    out = dict(config=dict(K=K, unit_hours=UNIT, super_blocks=SUPER, n_candidates=m),
               n_p_le_05=int((p <= 0.05).sum()), n_bh_le_10=int((q <= 0.10).sum()),
               n_bh_le_05=int((q <= 0.05).sum()), results=res)
    json.dump(out, open(os.path.join(E8, "retest_K999.json"), "w"), indent=1)
    print(f"\n  p<=0.05: {out['n_p_le_05']} | BH q<=0.10: {out['n_bh_le_10']} | "
          f"BH q<=0.05: {out['n_bh_le_05']}   ({time.time()-t0:.0f}s)", flush=True)
