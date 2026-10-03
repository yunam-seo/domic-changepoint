#!/usr/bin/env python
"""Ablation of the entropy, Frobenius and spectral-only functionals (Supplementary Section B.2).

Compares the entropy-based (von Neumann) functional on the random-feature density operator with a
Frobenius functional on the same features, a spectrum-only variant and MMD at several bandwidths.

Methods (all global, same grid):
  kQDg-Holevo  entropy (Holevo) functional of the feature moment (RFF D=64, median-heuristic gamma)
  kFrob        ||M_L - M_R||_F * sqrt(t(n-t)/n), M = mean phi phi^T (same RFF)
  kSpecHolevo  Holevo on sorted spectra only (commuting part)
  MMD x{0.25,0.5,1,2,4} median bandwidth multipliers (exact kernel); Figure B.1 shows the best of them
Scenarios: N1, N2, N3, S1, S3 of the auxiliary set in dots/synth.py (these
are not the article's S1-S4).  200 replicates.

Run:  python 00_SRC/run_ablation.py
Output: 04_DAOU/ABLATION/{results_long.csv, thresholds.json, config.json, records_*.csv.gz}
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

from dots.synth import SCENARIOS, DEFAULT_W, generate  # noqa: E402
from dots.detect import Context, qd_global, sqdist, prefix2d, block_sum  # noqa: E402
from dots.extras import rff  # noqa: E402
from dots.evaluate import summarize_alt, aggregate  # noqa: E402
from dots.persist import save_records  # noqa: E402

DAOU = os.path.join(ROOT, "04_DAOU", "ABLATION")
# m_embed is recorded in config.json but not used by this runner. w is the generator's whole DEFAULT_W map;
# the stored config.json lists the map as it stood at run time (including auxiliary scenarios not run
# here). The five scenarios run here all have w = 60.
CFG = dict(n=600, tau=300, reps=200, tol=30, base_seed=20260822, fpr=0.05, m_embed=10,
           scenarios=["N1", "N2", "N3", "S1", "S3"], w=DEFAULT_W, mults=[0.25, 0.5, 1.0, 2.0, 4.0])


def mmd_global_multi(Z, grid, mults, n, k2=True):
    D2 = sqdist(Z)
    med = np.median(D2[np.triu_indices(n, 1)])
    med = med if med > 0 else 1.0
    out = {}
    for m in mults + (["k2"] if k2 else []):
        gamma = (2.0 / med) if m == "k2" else 1.0 / (med * m)
        P = prefix2d(np.exp(-gamma * D2))
        v = np.empty(len(grid))
        for i, t in enumerate(grid):
            kLL = block_sum(P, 0, t, 0, t) / (t * t)
            kRR = block_sum(P, t, n, t, n) / ((n - t) ** 2)
            kLR = block_sum(P, 0, t, t, n) / (t * (n - t))
            v[i] = t * (n - t) / n * max(kLL + kRR - 2 * kLR, 0.0)
        out[f"MMDx{m}|g" if m != "k2" else "MMD-k2|g"] = v
    return out


def kfrob_global(ctxf):
    ca = ctxf.cache_raw()
    n = ctxf.n
    v = np.empty(len(ctxf.grid))
    for i, t in enumerate(ctxf.grid):
        v[i] = np.sqrt(t * (n - t) / n) * np.linalg.norm(ca.second_moment(0, t) - ca.second_moment(t, n), "fro")
    return v


def all_stats(sample, w, cfg):
    Z = sample["Z"]
    n = Z.shape[0]
    st = {}
    F = rff(Z, D=64, seed=12345)
    ctxf = Context(F, w, c_aug=0.0)
    qg = qd_global(ctxf)
    st["kQDg-Holevo|g"] = qg["QDg-Holevo"]
    st["kSpecHolevo|g"] = qg["QDg-SpecHolevo"]
    st["kFrob|g"] = kfrob_global(ctxf)
    grid = ctxf.grid
    st.update(mmd_global_multi(Z, grid, cfg["mults"], n, k2=False))
    return st, grid


def _null_job(a):
    s, r, cfg = a
    smp = generate(s, SCENARIOS[s]["levels"][0], r, n=cfg["n"], tau=cfg["tau"], null=True, base_seed=cfg["base_seed"], level_idx=0)
    st, _ = all_stats(smp, cfg["w"][s], cfg)
    return {k: float(np.nanmax(v)) for k, v in st.items()}


def _alt_job(a):
    s, li, r, cfg, thr = a
    smp = generate(s, SCENARIOS[s]["levels"][li], r, n=cfg["n"], tau=cfg["tau"], null=False, base_seed=cfg["base_seed"], level_idx=li)
    st, grid = all_stats(smp, cfg["w"][s], cfg)
    return summarize_alt(st, grid, smp["tau"], cfg["w"][s], cfg["n"], thr, cfg["tol"])


def run(cfg, procs):
    os.makedirs(DAOU, exist_ok=True)
    rows, thresholds = [], {}
    with Pool(procs) as pool:
        for s in cfg["scenarios"]:
            t0 = time.time()
            nulls = pool.map(_null_job, [(s, r, cfg) for r in range(cfg["reps"])], chunksize=2)
            save_records(DAOU, "records_null.csv", nulls, {"scen": s})
            thr = {k: float(np.quantile([d[k] for d in nulls], 1 - cfg["fpr"])) for k in nulls[0]}
            thresholds[s] = thr
            print(f"[{time.strftime('%H:%M:%S')}] {s} null {time.time()-t0:.0f}s", flush=True)
            for li, level in enumerate(SCENARIOS[s]["levels"]):
                recs = pool.map(_alt_job, [(s, li, r, cfg, thr) for r in range(cfg["reps"])], chunksize=2)
                # per-replicate records feed the paired comparison (run_ablation_paired.py)
                save_records(DAOU, "records_alt.csv", recs,
                             {"scen": s, "level": level, "level_idx": li})
                agg = aggregate(recs, thr)
                for k, v in agg.items():
                    name, mode = k.split("|")
                    rows.append(dict(scen=s, level=level, level_idx=li, method=name, mode=mode, key=k, **v))
                top = sorted(agg.items(), key=lambda kv: -kv[1]["power"])[:6]
                print(f"[{time.strftime('%H:%M:%S')}] {s} level={level}: " + ", ".join(f"{k}={v['power']:.2f}" for k, v in top), flush=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(DAOU, "results_long.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    json.dump(thresholds, open(os.path.join(DAOU, "thresholds.json"), "w"), indent=1)
    json.dump(cfg, open(os.path.join(DAOU, "config.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--procs", type=int, default=28)
    a = ap.parse_args()
    cfg = dict(CFG)
    cfg["reps"] = a.reps
    run(cfg, a.procs)
