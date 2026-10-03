#!/usr/bin/env python
"""Two design choices shared by every experiment: the random-feature draw and the break location.

H1  Random-feature draw. Every reported number uses one pair of RFF seeds (2026 for X, 2027 for Y).
    The statistic is a function of that draw; how much an answer moves when the draw changes is
    measured here on synthetic data (power across draws, and per-replicate agreement
    of the decision) and on the real weather pair whose result the paper discusses.

H2  Break location. Every synthetic scenario places the break at the midpoint, tau = n/2, so every
    reported power is a center-break power. Measured here at tau/n = 0.2, 0.33, 0.5, 0.67, 0.8,
    separating DETECTION (does the test reject) from LOCALIzATION (is the estimate within 30), and
    comparing four candidate location estimators plus the Holevo-partitioning segmentation route.

Writes 04_DAOU/EXPERIMENT/design_checks/results.json
"""
from __future__ import annotations
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.synth import generate            # noqa: E402
from dots.domi import ranks01, unit_rff     # noqa: E402
from dots.encode import MomentCache        # noqa: E402
from dots import pelt as P                 # noqa: E402
from dots.perm import ge  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "design_checks")
N, W, D, ALPHA = 600, 60, 8, 0.05
SEEDS = [(2026, 2027), (11, 12), (777, 778), (31415, 31416), (5, 6), (98765, 98766)]


def ent(M):
    l = np.clip(np.linalg.eigvalsh(M), 1e-300, None); l = l[l > 1e-14]
    return float(-(l * np.log(l)).sum())


def curve(x, y, sx, sy, n=None, w=W):
    n = len(x) if n is None else n
    FX = unit_rff(ranks01(np.asarray(x, float)[:, None]), D, sx)
    FY = unit_rff(ranks01(np.asarray(y, float)[:, None]), D, sy)
    J = np.einsum("ti,tj->tij", FX, FY).reshape(n, D * D)
    cx = np.zeros((n + 1, D, D)); cy = np.zeros((n + 1, D, D)); cj = np.zeros((n + 1, D * D, D * D))
    np.cumsum(np.einsum("ti,tj->tij", FX, FX), axis=0, out=cx[1:])
    np.cumsum(np.einsum("ti,tj->tij", FY, FY), axis=0, out=cy[1:])
    np.cumsum(np.einsum("ti,tj->tij", J, J), axis=0, out=cj[1:])
    def I(a, b):
        m = b - a
        return ent((cx[b] - cx[a]) / m) + ent((cy[b] - cy[a]) / m) - ent((cj[b] - cj[a]) / m)
    grid = np.arange(w, n - w + 1)
    return grid, np.array([np.sqrt(t * (n - t) / n) * abs(I(0, t) - I(t, n)) for t in grid])


# ---------------------------------------------------------------- H1 synthetic
def _h1(args):
    rep, null, sx, sy = args
    smp = generate("D2", 0.9, rep, n=N, tau=N // 2, null=null, base_seed=20260908, level_idx=2)
    Z = smp["Z"]
    return curve(Z[:, 0], Z[:, 1], sx, sy)[1]


REC = []


def h1_synthetic(pool, reps=200):
    out = {}
    per_rep = {}
    global REC
    for sx, sy in SEEDS:
        nulls = pool.map(_h1, [(r, True, sx, sy) for r in range(reps)], chunksize=4)
        alts = pool.map(_h1, [(r, False, sx, sy) for r in range(reps)], chunksize=4)
        nulls = np.array(nulls); alts = np.array(alts)
        half = reps // 2
        mu, sd = nulls[:half].mean(0), nulls[:half].std(0) + 1e-12
        thr = float(np.quantile([np.nanmax((c - mu) / sd) for c in nulls[half:]], 1 - ALPHA))
        grid = np.arange(W, N - W + 1)
        dec, pw = [], 0
        for c in alts:
            T = (c - mu) / sd
            d = bool(np.nanmax(T) > thr and abs(int(grid[int(np.nanargmax(T))]) - N // 2) <= 30)
            dec.append(d); pw += d
        out[f"{sx}/{sy}"] = round(pw / reps, 3)
        per_rep[f"{sx}/{sy}"] = dec
        REC += [dict(part="H1", config=f"{sx}/{sy}", rep=i, value=int(d)) for i, d in enumerate(dec)]
        print(f"  H1 seeds {sx}/{sy}: power={pw/reps:.3f}", flush=True)
    M = np.array([per_rep[k] for k in per_rep])
    agree = float(np.mean(M.all(0) | (~M).all(0)))
    return dict(power_by_seed=out, unanimous_fraction=round(agree, 3),
                power_min=min(out.values()), power_max=max(out.values()))


# ---------------------------------------------------------------- H2 break location
def _h2(args):
    rep, null, frac = args
    tau = int(round(frac * N))
    smp = generate("D2", 0.9, rep, n=N, tau=tau, null=null, base_seed=20260909, level_idx=2)
    Z = smp["Z"]
    return curve(Z[:, 0], Z[:, 1], 2026, 2027)[1]


def h2_location(pool, reps=200):
    """Detection vs localization at five break positions, for four location estimators."""
    nulls = np.array(pool.map(_h2, [(r, True, 0.5) for r in range(reps)], chunksize=4))
    half = reps // 2
    grid = np.arange(W, N - W + 1)
    wgt = np.sqrt(grid * (N - grid) / N)
    unw_null = nulls / wgt                                  # the weight cancels under studentization
    muw, sdw = nulls[:half].mean(0), nulls[:half].std(0) + 1e-12
    muu = unw_null[:half].mean(0)
    thr = float(np.quantile([np.nanmax((c - muw) / sdw) for c in nulls[half:]], 1 - ALPHA))
    out = {}
    for frac in (0.2, 0.33, 0.5, 0.67, 0.8):
        tau = int(round(frac * N))
        alts = np.array(pool.map(_h2, [(r, False, frac) for r in range(reps)], chunksize=4))
        rej = stud = raww = cenw = cenu = 0
        for ri, c in enumerate(alts):
            T = (c - muw) / sdw
            hit = lambda v: abs(int(grid[int(np.nanargmax(v))]) - tau) <= 30
            det = bool(np.nanmax(T) > thr)
            REC.append(dict(part="H2", config=f"tau/n={frac}", rep=ri,
                            value=int(det), extra=int(det and hit(T))))
            if not det:
                continue
            rej += 1
            stud += hit(T)                 # studentized (deployed); identical weighted/unweighted
            raww += hit(c)                 # raw weighted
            cenw += hit(c - muw)           # centered, weighted
            cenu += hit(c / wgt - muu)     # centered, unweighted
        out[f"tau/n={frac}"] = dict(reject=round(rej / reps, 3), loc_studentised=round(stud / reps, 3),
                                    loc_raw_weighted=round(raww / reps, 3),
                                    loc_centred_weighted=round(cenw / reps, 3),
                                    loc_centred_unweighted=round(cenu / reps, 3))
        print(f"  H2 tau/n={frac}: reject={rej/reps:.3f} loc(stud)={stud/reps:.3f} "
              f"raw-w={raww/reps:.3f} cen-w={cenw/reps:.3f} cen-u={cenu/reps:.3f}", flush=True)
    return out


def _pelt(args):
    rep, frac, beta, null = args
    tau = int(round(frac * N))
    Z = generate("D2", 0.9, rep, n=N, tau=tau, null=null, base_seed=20260911, level_idx=2)["Z"]
    FX = unit_rff(ranks01(Z[:, 0][:, None]), D, 2026); FY = unit_rff(ranks01(Z[:, 1][:, None]), D, 2027)
    J = np.einsum("ti,tj->tij", FX, FY).reshape(N, D * D)
    grid = np.arange(0, N + 1, 10)
    rng = np.random.default_rng(1000 + rep)
    C = P.cost_matrix_vn(MomentCache(J), grid)
    Cp = np.zeros_like(C)
    for _ in range(3):
        Cp += P.cost_matrix_vn(MomentCache(J[rng.permutation(N)]), grid)
    cps = P.pelt_from_costs(C - Cp / 3, beta)
    locs = [int(grid[c]) for c in cps]
    return dict(k=len(locs), hit=any(abs(l - tau) <= 30 for l in locs))


def h2_pelt(pool, reps=100):
    """The same break positions, through the segmentation route the applications actually use (Holevo
    partitioning; the part tag "H2_pelt" and result key "H2_break_location_pelt" refer to this route)."""
    for beta in (2.0, 4.0, 6.0, 8.0, 12.0):
        ks = pool.map(_pelt, [(r, 0.5, beta, True) for r in range(60)], chunksize=2)
        fa = float(np.mean([k["k"] > 0 for k in ks]))
        print(f"  Holevo partitioning beta={beta}: null any-break rate={fa:.3f}", flush=True)
        if fa <= 0.05:
            break
    out = {"beta": beta}
    for frac in (0.2, 0.33, 0.5, 0.67, 0.8):
        res = pool.map(_pelt, [(r, frac, beta, False) for r in range(reps)], chunksize=2)
        REC.extend(dict(part="H2_pelt", config=f"tau/n={frac}", rep=i, value=int(r["hit"]),
                        extra=r["k"]) for i, r in enumerate(res))
        out[f"tau/n={frac}"] = dict(found_within_30=round(float(np.mean([r["hit"] for r in res])), 3),
                                    exactly_one=round(float(np.mean([r["k"] == 1 for r in res])), 3))
        print(f"  Holevo partitioning tau/n={frac}: found={out[f'tau/n={frac}']['found_within_30']:.3f} "
              f"k=1 in {out[f'tau/n={frac}']['exactly_one']:.3f}", flush=True)
    return out


def _h1e(args):
    """Statistic averaged over all SEEDS draws (an ensemble of feature maps)."""
    rep, null = args
    smp = generate("D2", 0.9, rep, n=N, tau=N // 2, null=null, base_seed=20260908, level_idx=2)
    Z = smp["Z"]
    return np.mean([curve(Z[:, 0], Z[:, 1], sx, sy)[1] for sx, sy in SEEDS], axis=0)


def h1_ensemble(pool, reps=200):
    """Does averaging the statistic over draws remove the draw-to-draw variation?"""
    nulls = np.array(pool.map(_h1e, [(r, True) for r in range(reps)], chunksize=2))
    alts = np.array(pool.map(_h1e, [(r, False) for r in range(reps)], chunksize=2))
    half = reps // 2
    mu, sd = nulls[:half].mean(0), nulls[:half].std(0) + 1e-12
    thr = float(np.quantile([np.nanmax((c - mu) / sd) for c in nulls[half:]], 1 - ALPHA))
    grid = np.arange(W, N - W + 1)
    pw = sum(bool(np.nanmax((c - mu) / sd) > thr and abs(int(grid[int(np.nanargmax((c - mu) / sd))]) - N // 2) <= 30)
             for c in alts)
    print(f"  H1 ensemble of {len(SEEDS)} draws: power={pw/reps:.3f}", flush=True)
    return round(pw / reps, 3)


# ---------------------------------------------------------------- H1 on real data
def h1_real():
    import csv
    path = os.path.join(ROOT, "02_MART", "WEATHER_DAILY.csv")
    rows = list(csv.DictReader(open(path)))
    ser = {}
    for r in rows:
        if r.get("stn") != "108":
            continue
        ser.setdefault("date", []).append(r["date"])
        for k in ("ta", "ws"):
            ser.setdefault(k, []).append(float(r[k]) if r[k] not in ("", "None") else np.nan)
    d = np.array(ser["date"]); ta = np.array(ser["ta"]); ws = np.array(ser["ws"])
    sel = (d >= "20220101") & (d <= "20231231")
    ta, ws = ta[sel], ws[sel]
    ok = ~(np.isnan(ta) | np.isnan(ws)); ta, ws = ta[ok], ws[ok]
    # anomalies against the mean of each position modulo 365 (the 2022-2023 window has no leap day
    # and, for station 108, no missing day, so position modulo 365 is the day of the year)
    doy = np.arange(len(ta)) % 365
    for arr in (ta, ws):
        for k in range(365):
            m = doy == k
            if m.sum() > 1:
                arr[m] -= arr[m].mean()
    n = len(ta)
    res = {}
    for sx, sy in SEEDS:
        rng = np.random.default_rng(20260910)
        g, obs = curve(ta, ws, sx, sy, n=n)
        R = []
        for k in range(99):
            nb = n // 20
            order = rng.permutation(nb)
            idx = np.concatenate([np.arange(j * 20, (j + 1) * 20) for j in order])
            if nb * 20 < n:
                idx = np.concatenate([idx, np.arange(nb * 20, n)])
            R.append(curve(ta[idx], ws[idx], sx, sy, n=n)[1])
        A = np.vstack([obs[None, :], np.array(R)])
        mu, sd = A.mean(0), A.std(0) + 1e-12
        T = [float(np.nanmax((A[i] - mu) / sd)) for i in range(100)]
        p = (1 + sum(ge(t, T[0]) for t in T[1:])) / 100
        res[f"{sx}/{sy}"] = round(p, 3)
        print(f"  H1-real Seoul ta-ws block20 seeds {sx}/{sy}: p={p:.3f}", flush=True)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with Pool(a.procs, maxtasksperchild=30) as pool:
        t0 = time.time(); print("H2 break location:", flush=True)
        h2 = h2_location(pool, a.reps)
        print("H2 through Holevo partitioning:", flush=True)
        h2p = h2_pelt(pool, 100)
        print(f"H1 feature draw ({time.time()-t0:.0f}s so far):", flush=True)
        h1 = h1_synthetic(pool, a.reps)
        h1e = h1_ensemble(pool, a.reps)
    print("H1 on real data:", flush=True)
    real = h1_real()
    import csv as _csv
    with open(os.path.join(OUT, "records_reps.csv"), "w", newline="") as f:
        wcsv = _csv.DictWriter(f, fieldnames=["part", "config", "rep", "value", "extra"])
        wcsv.writeheader()
        wcsv.writerows([{**dict(extra=""), **r} for r in REC])
    json.dump(dict(config=dict(n=N, w=W, D=D, alpha=ALPHA, seeds=SEEDS, reps=a.reps),
                   H2_break_location=h2, H2_break_location_pelt=h2p, H1_feature_draw=h1, H1_ensemble_power=h1e, H1_real_weather=real),
              open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print("design_checks done", flush=True)
