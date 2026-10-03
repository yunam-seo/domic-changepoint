#!/usr/bin/env python
"""Gram form of DOMI (matrix-based mutual information) on the single-break designs of Section 5.

The Gram form computes the mutual information of a segment from Gram matrices of the Gaussian kernel
on the rank inputs, with the joint entropy from the entrywise product of the two Gram matrices
(Sanchez Giraldo, Rao and Principe, 2015; Yu et al., 2020). It is run on the same rank inputs,
candidate grid, studentization and max-over-grid null calibration as the random-feature DOMI
difference, which is recomputed in the same run. Method keys in the outputs: "DOMI-diff" is the
random-feature DOMI difference, "matmi-a1" the Gram form (von Neumann entropy), "matmi-r2" its
order-2 Renyi variant.

The kernel bandwidth is the median heuristic that the random features approximate
(dots.extras.rff), so both forms act on the same kernel. The article's Gram form uses every row of
each segment (--msub 0); --msub 256 draws a fresh 256-row subsample per segment instead
(Supplementary Section B.10 reports what that does to localization). --matched calibrates each level
of S3/S4 against its own pre-change regime, as in Table 1.

Runs behind the article (see also run_gram_full.sh):
    python 00_SRC/run_matmi_baseline.py --scenarios D2,D4,M1,D1,D3
    python 00_SRC/run_matmi_baseline.py --scenarios D3,D4 --matched --tag matched_D8
        (these two supply the random-feature D = 8 cells of Table B.3; their Gram form uses 256-row
        subsamples and is replaced in the tables by the full-segment runs of run_gram_full.sh)
    python 00_SRC/run_matmi_baseline.py --scenarios D2,D4 --D 16 --no-matmi --tag D16
    python 00_SRC/run_matmi_baseline.py --scenarios D3,D4 --D 16 --no-matmi --matched --tag matched_D16
        (random-feature form only, D = 16 cells of Table B.3)
    bash 00_SRC/run_gram_full.sh        (full-segment Gram form, every cell of Tables 1 and B.3)
Scenario codes: D1-D4 = S1-S4 of the article.
Outputs: 04_DAOU/EXPERIMENT/matmi_baseline[_<tag>]/{results.csv, summary.json, records_*.csv.gz, run.log}
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

from dots.synth import SCENARIOS, DEFAULT_W, generate  # noqa: E402
from dots.domi import DOMIContext, domi_stats, _segments, DEP_BASELINES  # noqa: E402
from dots.extras import studentize  # noqa: E402
from dots.evaluate import summarize_alt, aggregate  # noqa: E402
from dots.persist import save_records  # noqa: E402

BASE = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "matmi_baseline")
# Same calibration constants as run_experiment.CFG (Table 1 settings).
CFG = dict(n=600, tau=300, tol=30, base_seed=20260825, fpr=0.05, D=8)
MSUB = 256           # subsample cap per segment (feasibility)
SUBSEED = 20260924   # base for the deterministic per-segment subsample rng
SCEN_ID = {"D1": 1, "D2": 2, "D3": 3, "D4": 4, "M1": 5}


# ---------------------------------------------------------------- matrix-based entropy tools
def eig_unit(G):
    lam = np.linalg.eigvalsh((G + G.T) / 2.0)
    lam = np.clip(lam, 0.0, None)
    s = lam.sum()
    return lam / s if s > 0 else lam


def H(lam):
    lam = lam[lam > 1e-15]
    return float(-(lam * np.log(lam)).sum())


def Hr(lam, a):
    lam = lam[lam > 1e-15]
    return float(np.log((lam ** a).sum()) / (1.0 - a))


def gram(Z, g):
    Z = Z if Z.ndim == 2 else Z[:, None]
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    return np.exp(-g * d2)


# ---------------------------------------------------------------- bandwidth matched to the RFF
def rff_gamma(Z, seed):
    """Median-heuristic gamma EXACTLY as dots.extras.rff computes it (W ~ N(0, 2*gamma))."""
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    n = Z.shape[0]
    idx = np.random.default_rng(seed + 1).choice(n, size=min(n, 400), replace=False)
    S = Z[idx]
    sq = (S * S).sum(1)
    D2 = np.clip(sq[:, None] + sq[None, :] - 2 * S @ S.T, 0, None)
    med = np.median(D2[np.triu_indices(len(idx), 1)])
    return 1.0 / (med if med > 0 else 1.0)


# ---------------------------------------------------------------- matrix-based MI curves
SUBMODE = "perseg"   # "perseg": fresh draw per segment (default); "bottomk": see _sub
_KEYS = {}


def _keys(base, b):
    """Per-series random key for every time index, drawn once (prefix-consistent in length)."""
    k = tuple(base)
    if k not in _KEYS or len(_KEYS[k]) < b:
        _KEYS[k] = np.random.default_rng(list(base) + [7]).random(max(b, 1024))
    return _KEYS[k]


def _sub(a, b, base):
    """Deterministic subsample of rows [a,b) to at most MSUB.

    perseg  -- a fresh draw per (series, segment): the subsample of [0,t) and of [0,t+1) are
               unrelated, so the curve over t carries subsampling noise from candidate to candidate.
    bottomk -- bottom-k sampling: every time index carries one random key drawn once per series,
               and a segment keeps its MSUB indices with the smallest keys. Moving a boundary by one
               index changes the subsample by at most one row, so the curve over t is not re-randomized
               at every candidate. The keys attach to time positions, not to data, so the observed
               series and any permuted replica are subsampled identically (Proposition P1 applies).
    """
    if b - a <= MSUB:
        return np.arange(a, b)
    if SUBMODE == "bottomk":
        kk = _keys(base, b)[a:b]
        return a + np.sort(np.argpartition(kk, MSUB - 1)[:MSUB])
    rng = np.random.default_rng(base + [int(a), int(b)])
    return a + rng.choice(b - a, MSUB, replace=False)


def matmi_seg(ux, uy, gx, gy):
    """(alpha=1, Renyi-2) matrix-based MI on aligned subsampled segment rows."""
    Gx = gram(ux, gx)
    Gy = gram(uy, gy)
    Gxy = Gx * Gy                     # Hadamard joint (Yu et al.)
    lx, ly, lxy = eig_unit(Gx), eig_unit(Gy), eig_unit(Gxy)
    a1 = H(lx) + H(ly) - H(lxy)
    r2 = Hr(lx, 2.0) + Hr(ly, 2.0) - Hr(lxy, 2.0)
    return a1, r2


def matmi_curves(ctx, mode, gx, gy, base):
    """Per-grid diff statistic sqrt(wt)*|MI_L - MI_R| for alpha=1 and Renyi-2 (same diff pattern
    as domi_stats / hsic_diff)."""
    UX, UY = ctx.UX[:, 0], ctx.UY[:, 0]
    a1 = np.empty(len(ctx.grid))
    r2 = np.empty(len(ctx.grid))
    for i, t, (a1s, b1), (a2s, b2), wt in _segments(ctx, mode):
        il = _sub(a1s, b1, base)
        ir = _sub(a2s, b2, base)
        La1, Lr2 = matmi_seg(UX[il], UY[il], gx, gy)
        Ra1, Rr2 = matmi_seg(UX[ir], UY[ir], gx, gy)
        a1[i] = np.sqrt(wt) * abs(La1 - Ra1)
        r2[i] = np.sqrt(wt) * abs(Lr2 - Rr2)
    return a1, r2


# ---------------------------------------------------------------- one replicate: all stat curves
def stats_one(smp, w, cfg, base):
    Z = smp["Z"]
    X, Y = Z[:, smp["blocks"][0]], Z[:, smp["blocks"][1]]
    ctx = DOMIContext(X, Y, w, D=cfg["D"], seed=2026, n_perm=0)
    gx = rff_gamma(ctx.UX, 2026)
    gy = rff_gamma(ctx.UY, 2027)
    st = {}
    for mode, suf in [("global", "g")]:
        st[f"DOMI-diff|{suf}"] = domi_stats(ctx, mode)["DOMI-diff"]
        if cfg.get("deps", False):   # classical dependence baselines on the same rank inputs
            for k, f in DEP_BASELINES.items():
                st[f"{k}|{suf}"] = f(ctx, mode)
        if not cfg.get("matmi", True):
            continue          # random-feature-only D-sweep: matmi is D-independent, computed once at D=8
        a1, r2 = matmi_curves(ctx, mode, gx, gy, base)
        st[f"matmi-a1|{suf}"] = a1
        st[f"matmi-r2|{suf}"] = r2
    return st, ctx.grid, (gx, gy)


def _base_for(scen, li, r, null):
    return [SUBSEED, SCEN_ID[scen], li, r, int(null)]


# ---------------------------------------------------------------- workers
def _null_job(a):
    """Null replicate. (s, r, cfg) draws at level 0; (s, r, cfg, li) draws
    the null at the alternative's own level li -- the pre-change regime held throughout, which is the
    matched null of run_e1_matched_null.py and the protocol of the printed Table 1 for D3/D4."""
    s, r, cfg = a[:3]
    li = a[3] if len(a) > 3 else 0
    smp = generate(s, SCENARIOS[s]["levels"][li], r, null=True, base_seed=cfg["base_seed"], level_idx=li)
    return stats_one(smp, DEFAULT_W[s], cfg, _base_for(s, li, r, True))[0]


def _alt_job(a):
    s, li, r, cfg, thr, mu0, sd0 = a
    smp = generate(s, SCENARIOS[s]["levels"][li], r, null=False, base_seed=cfg["base_seed"], level_idx=li)
    st, grid, _ = stats_one(smp, DEFAULT_W[s], cfg, _base_for(s, li, r, False))
    for k in list(st):
        if k.endswith("|g"):
            st[k[:-1] + "gs"] = studentize(st[k], mu0[k], sd0[k])
    return summarize_alt(st, grid, smp["tau"], DEFAULT_W[s], cfg["n"], thr, cfg["tol"])


# ---------------------------------------------------------------- runner
def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(os.path.join(BASE, "run.log"), "a") as f:
        f.write(line + "\n")


def run(cfg, procs, reps, scenarios):
    rows = []
    bandwidths = {}
    with Pool(procs, maxtasksperchild=30) as pool:
        for s in scenarios:
            matched = cfg.get("matched", False)
            wanted = cfg.get("levels", {}).get(s)          # None -> every level
            have_null = False
            for li, level in enumerate(SCENARIOS[s]["levels"]):
                include = wanted is None or any(abs(level - w) < 1e-9 for w in wanted)
                if matched and not include:
                    continue
                if matched or not have_null:
                    have_null = True
                    # null calibration: once per scenario at level 0, or per level (matched)
                    t0 = time.time()
                    nulls = pool.map(_null_job, [(s, r, cfg, li if matched else 0) for r in range(reps)],
                                     chunksize=4)
                    save_records(BASE, "records_null.csv", nulls,
                                 {"scen": s, "null_level": level if matched else SCENARIOS[s]["levels"][0]})
                    keys = list(nulls[0].keys())
                    half = reps // 2
                    mu0 = {k: np.mean([d[k] for d in nulls[:half]], axis=0) for k in keys if k.endswith("|g")}
                    sd0 = {k: np.std([d[k] for d in nulls[:half]], axis=0) for k in keys if k.endswith("|g")}
                    thr = {k: float(np.quantile([np.nanmax(d[k]) for d in nulls], 1 - cfg["fpr"])) for k in keys}
                    for k in keys:
                        if k.endswith("|g"):
                            thr[k[:-1] + "gs"] = float(np.quantile(
                                [np.nanmax(studentize(d[k], mu0[k], sd0[k])) for d in nulls[half:]],
                                1 - cfg["fpr"]))
                    say(f"{s}: null{' @'+str(level) if matched else ''} {reps} reps {time.time()-t0:.0f}s")
                if not include:
                    continue
                t1 = time.time()
                recs = pool.map(_alt_job, [(s, li, r, cfg, thr, mu0, sd0) for r in range(reps)], chunksize=4)
                save_records(BASE, "records_alt.csv", recs, {"scen": s, "level": level})
                agg = aggregate(recs, thr)
                for k, v in agg.items():
                    name, mode = k.split("|")
                    rows.append(dict(scen=s, level=level, method=name, mode=mode, key=k,
                                     power=round(v["power"], 4), detect_rate=round(v["detect_rate"], 4),
                                     threshold=v["threshold"], n_reps=v["n_reps"]))
                top = sorted(agg.items(), key=lambda kv: -kv[1]["power"])[:6]
                say(f"{s} level={level}: {time.time()-t1:.0f}s | "
                    + ", ".join(f"{k}={v['power']:.2f}" for k, v in top))
            # record bandwidth actually used (deterministic; take rep 0 alt)
            smp = generate(s, SCENARIOS[s]["levels"][0], 0, null=False, base_seed=cfg["base_seed"], level_idx=0)
            _, _, (gx, gy) = stats_one(smp, DEFAULT_W[s], cfg, _base_for(s, 0, 0, False))
            bandwidths[s] = dict(gamma_x=round(gx, 5), gamma_y=round(gy, 5))
    # results.csv
    keyset = []
    for r in rows:
        for k in r:
            if k not in keyset:
                keyset.append(k)
    with open(os.path.join(BASE, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keyset)
        wr.writeheader()
        wr.writerows(rows)
    return rows, bandwidths


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="D2,D4,M1,D1,D3")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-matmi", action="store_true", help="random-feature DOMI only (feature-dimension runs)")
    ap.add_argument("--matched", action="store_true",
                    help="calibrate each level against its own null (pre-change regime held throughout)")
    ap.add_argument("--subsample", choices=["perseg", "bottomk"], default="perseg")
    ap.add_argument("--msub", type=int, default=256, help="rows per segment for the Gram form; 0 = every row")
    ap.add_argument("--levels", default="", help="e.g. 'D1:0.35;D2:0.7,0.9' (default: every level)")
    a = ap.parse_args()
    if a.tag:
        BASE = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "matmi_baseline_" + a.tag)
    os.makedirs(BASE, exist_ok=True)
    cfg = dict(CFG)
    cfg["D"] = a.D
    cfg["matmi"] = not a.no_matmi
    cfg["matched"] = a.matched
    global SUBMODE
    SUBMODE = a.subsample
    cfg["subsample"] = a.subsample
    global MSUB
    MSUB = a.msub if a.msub > 0 else 10 ** 9
    cfg["msub"] = a.msub
    if a.levels:
        cfg["levels"] = {k: [float(x) for x in v.split(",")] for k, v in
                         (item.split(":") for item in a.levels.split(";"))}
    t0 = time.time()
    scenarios = a.scenarios.split(",")
    rows, bandwidths = run(cfg, a.procs, a.reps, scenarios)
    summary = dict(
        config=dict(cfg, reps=a.reps, scenarios=scenarios, MSUB=MSUB, subsample_seed_base=SUBSEED,
                    grid_stride=1, D_domi=cfg["D"]),
        method_note=("matrix-based MI (Sanchez Giraldo et al. 2015; Yu et al. 2020); "
                     "Hadamard joint entropy; bandwidth = RFF median-heuristic gamma on whole rank series "
                     "(exact same kernel DOMI's RFF at D=8 approximates); per-segment subsample to "
                     f"m=min(n_seg,{MSUB}); DOMI-diff recomputed in same run for head-to-head; "
                     "identical grid / window+global / studentised / max-over-grid null calibration."),
        bandwidths=bandwidths,
        runtime_sec=round(time.time() - t0, 1),
        rows=rows)
    json.dump(summary, open(os.path.join(BASE, "summary.json"), "w"), indent=1, ensure_ascii=False)
    say(f"done {summary['runtime_sec']}s -> {os.path.join(BASE, 'results.csv')}")


if __name__ == "__main__":
    main()
