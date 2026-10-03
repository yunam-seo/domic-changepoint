#!/usr/bin/env python
"""Block-exchangeability diagnostics for the weather application (Section 6.7).

The block-permutation calibration is finite-sample exact under block exchangeability of the blocks
actually permuted. Exchangeability is a symmetry of the joint distribution, not a marginal
assumption, so it cannot be established from a single record; what can be done is to look for major,
observable departures relevant to the permutation scheme in use. Two stages, two scopes:

  stage one -- twelve-week super-blocks over the WHOLE eight-year record, because that is what the
      segmentation and its penalty calibration see. The diagnostic looks for sources of
      non-exchangeability that would be present under the null regardless of whether a coupling
      change exists -- above all residual seasonality in the block MEAN, which the anomaly transform
      is meant to have removed. Seasonal heterogeneity in the block VARIANCE is a different matter:
      the transform deliberately leaves the diurnal amplitude free to vary with season, and a
      twelve-week block is close to a season long, so heterogeneity there is expected. It is still
      a departure from the identical-distribution half of block exchangeability, and it is
      measured here.

  stage two -- six-week super-blocks inside each candidate's +/-1 year re-test window. Six
      weeks average over a narrower part of the seasonal cycle than twelve, so the same
      season-linked variation is if anything more visible, and here it bears on the finite-sample
      exactness of the re-test itself. Windows around different candidates overlap heavily,
      so pooling their blocks would be pseudoreplication: a per-window effect size is computed and
      its distribution across candidates reported.

Deliberately low-dimensional. These are not exhaustive tests of joint block exchangeability, and
non-rejection is read as no detected departure rather than as independence.

The p-values are PERMUTATION p-values, not chi-square ones. With about 35 blocks (stage one) or 17
(stage two) the chi-square approximation behind Ljung-Box is not trustworthy, and it does not have
to be used: the hypothesis under test is exactly that the block sequence is exchangeable, so
permuting the block order and recomputing the statistic is exact under that null with no asymptotics
-- the same argument that licenses the block calibration itself (P5).

    OMP_NUM_THREADS=1 python 00_SRC/run_block_exch_diag.py
Writes 04_DAOU/EXPERIMENT/block_exch_diag/{stage1.csv,stage2.csv,summary.json,run.log}.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
from scipy.stats import chi2, rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "00_SRC"))
from dots.perm import ge  # noqa: E402
ANOM = os.path.join(ROOT, "02_MART", "WEATHER_HOURLY_ANOM.npz")
E8 = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8", "results.json")
OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "block_exch_diag")

WEEK = 7 * 24
STAGE1_BLOCK = 12 * WEEK          # 2016 h, as deployed for the cost correction and the penalty
STAGE2_BLOCK = 6 * WEEK           # 1008 h, as deployed inside the re-test window
WINDOW_HALF = 365 * 24            # +/- 1 year


def acf(x, L):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    d = (x * x).sum()
    return [float((x[:n - k] * x[k:]).sum() / d) if d > 0 else np.nan for k in range(1, L + 1)]


def ljung_box(x, L):
    """Q = n(n+2) sum r_k^2/(n-k) ~ chi2_L (Ljung-Box statistic)."""
    n = len(x)
    r = acf(x, L)
    Q = n * (n + 2) * sum((rk ** 2) / (n - k - 1) for k, rk in enumerate(r) if np.isfinite(rk))
    return float(Q), float(1 - chi2.cdf(Q, L)), r


def lb_perm_p(v, L, K=4999, seed=20260831):
    """Exact-under-block-exchangeability p-value for the Ljung-Box statistic on a block summary."""
    rng = np.random.default_rng(seed)
    obs = ljung_box(v, L)[0]
    n_ge = sum(ge(ljung_box(v[rng.permutation(len(v))], L)[0], obs) for _ in range(K))
    return float((1 + n_ge) / (K + 1))


def season_of(ts):
    m = int(str(ts)[4:6])
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}[m]


def doy_frac(ts):
    s = str(ts)
    mo, da = int(s[4:6]), int(s[6:8])
    cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    return (cum[mo - 1] + da - 1) / 365.0


def blockify(x, tm, blen):
    """Consecutive blocks of blen hours: per-block mean, variance, season and year-fraction."""
    nb = len(x) // blen
    out = []
    for b in range(nb):
        seg = x[b * blen:(b + 1) * blen]
        if np.isfinite(seg).sum() < blen // 2:
            continue
        mid = tm[b * blen + blen // 2]
        out.append(dict(idx=b, mean=float(np.nanmean(seg)), var=float(np.nanvar(seg)),
                        season=season_of(mid), frac=doy_frac(mid)))
    return out


def harmonic_amplitude(frac, logvar):
    """Amplitude of a fitted one-year-period harmonic on log block variance."""
    A = np.column_stack([np.ones(len(frac)), np.cos(2 * np.pi * np.asarray(frac)),
                         np.sin(2 * np.pi * np.asarray(frac))])
    coef, *_ = np.linalg.lstsq(A, np.asarray(logvar), rcond=None)
    return float(np.hypot(coef[1], coef[2]))


def season_ratio(blocks):
    """Largest / smallest season-group mean block variance."""
    g = {}
    for b in blocks:
        g.setdefault(b["season"], []).append(b["var"])
    m = [np.mean(v) for v in g.values() if len(v) >= 2]
    return float(max(m) / min(m)) if len(m) >= 2 and min(m) > 0 else np.nan


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return np.nan
    return float(np.corrcoef(rankdata(a[ok]), rankdata(b[ok]))[0, 1])


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "run.log"), "w")

    def say(m):
        print(m, flush=True)
        log.write(m + "\n")

    z = np.load(ANOM, allow_pickle=True)
    res = json.load(open(E8))["results"]

    # ---------------- stage one: whole record, twelve-week blocks ----------------
    say("=== stage 1 -- full record, 12-week blocks (per station-variable) ===")
    s1 = []
    seen = set()
    for r in res:
        stn = r["stn"]
        for v in r["pair"].split("-"):
            if (stn, v) in seen:
                continue
            seen.add((stn, v))
            x, tm = z[f"{stn}|{v}"], z[f"{stn}|tm"]
            bl = blockify(x, tm, STAGE1_BLOCK)
            if len(bl) < 8:
                continue
            mu = np.array([b["mean"] for b in bl])
            lv = np.log(np.array([b["var"] for b in bl]))
            sd = float(np.nanstd(x))          # the series' own scale, to make amplitudes comparable
            Qm, _, rm = ljung_box(mu, 5)
            Qv, _, rv = ljung_box(lv, 5)
            pm, pv = lb_perm_p(mu, 5), lb_perm_p(lv, 5)
            s1.append(dict(stn=stn, station=r["station"], var=v, n_blocks=len(bl), series_sd=round(sd, 4),
                           mean_lb_p=round(pm, 4), mean_acf1=round(rm[0], 3),
                           mean_harm_amp_rel=round(harmonic_amplitude([b["frac"] for b in bl], mu) / sd, 4),
                           logvar_lb_p=round(pv, 4), logvar_acf1=round(rv[0], 3),
                           logvar_harm_amp=round(harmonic_amplitude([b["frac"] for b in bl], lv), 4),
                           var_season_ratio=round(season_ratio(bl), 3)))
    say(f"  station-variables {len(s1)}, blocks {min(d['n_blocks'] for d in s1)}-{max(d['n_blocks'] for d in s1)}")
    for lab, key in (("residual seasonality of block means (harmonic amp / series sd)", "mean_harm_amp_rel"),
                     ("block-mean LB permutation p", "mean_lb_p"),
                     ("log block variance harmonic amplitude", "logvar_harm_amp"),
                     ("block-variance season-group max/min ratio", "var_season_ratio")):
        v = np.array([d[key] for d in s1], float)
        say(f"  {lab:52s} median {np.nanmedian(v):8.3f}   range {np.nanmin(v):.3f}-{np.nanmax(v):.3f}")
    nrej = sum(d["mean_lb_p"] < 0.05 for d in s1)
    say(f"  block-mean LB (permutation) rejects at 0.05: {nrej}/{len(s1)}")
    nrv = sum(d["logvar_lb_p"] < 0.05 for d in s1)
    say(f"  log-block-variance LB (permutation) rejects at 0.05: {nrv}/{len(s1)}")

    # ---------------- stage two: re-test windows, six-week blocks ----------------
    say("\n=== stage 2 -- +/-1y re-test windows, 6-week blocks (per candidate) ===")
    s2 = []
    for r in res:
        stn = r["stn"]
        tms = z[f"{stn}|tm"]
        ds = np.array([str(t)[:8] for t in tms])
        for br in r["breaks"]:
            i = int(np.argmax(ds >= br["date"])) if (ds >= br["date"]).any() else len(ds) // 2
            a, b = max(0, i - WINDOW_HALF), min(len(ds), i + WINDOW_HALF)
            rec = dict(stn=stn, station=r["station"], pair=r["pair"], date=br["date"],
                       p_block=br["p_block"], passed=br["passed"],
                       window_hours=int(b - a),
                       clipped=bool((b - a) < 2 * WINDOW_HALF))
            amps, ratios, lbp, lvp = [], [], [], []
            for v in r["pair"].split("-"):
                bl = blockify(z[f"{stn}|{v}"][a:b], tms[a:b], STAGE2_BLOCK)
                if len(bl) < 6:
                    continue
                lv = np.log(np.array([bb["var"] for bb in bl]))
                mu = np.array([bb["mean"] for bb in bl])
                amps.append(harmonic_amplitude([bb["frac"] for bb in bl], lv))
                ratios.append(season_ratio(bl))
                lbp.append(lb_perm_p(mu, 3))
                lvp.append(lb_perm_p(lv, 3))
                rec["n_blocks"] = len(bl)
            if not amps:
                continue
            # rank-correlation stability: the two halves either side of the candidate, separately
            xv, yv = r["pair"].split("-")
            X, Y = z[f"{stn}|{xv}"], z[f"{stn}|{yv}"]
            hl = [spearman(X[a:i], Y[a:i]), spearman(X[i:b], Y[i:b])]
            rec.update(logvar_harm_amp=round(float(np.nanmax(amps)), 4),
                       var_season_ratio=round(float(np.nanmax(ratios)), 3),
                       mean_lb_p_min=round(float(np.nanmin(lbp)), 4),
                       logvar_lb_p_min=round(float(np.nanmin(lvp)), 4),
                       rho_left=round(hl[0], 4), rho_right=round(hl[1], 4),
                       rho_shift=round(abs(hl[1] - hl[0]), 4))
            s2.append(rec)
    nclip = sum(d["clipped"] for d in s2)
    say(f"  candidates {len(s2)}, blocks per window {min(d['n_blocks'] for d in s2)}-{max(d['n_blocks'] for d in s2)}"
        f"; windows clipped at the record edge {nclip} (full +/-1y: {len(s2)-nclip})")
    for lab, key in (("log block variance harmonic amplitude", "logvar_harm_amp"),
                     ("block-variance season-group max/min ratio", "var_season_ratio"),
                     ("block-mean LB permutation p (min over variables)", "mean_lb_p_min"),
                     ("log-block-variance LB permutation p (min over variables)", "logvar_lb_p_min"),
                     ("rank-correlation shift across the candidate |d rho|", "rho_shift")):
        v = np.array([d[key] for d in s2], float)
        say(f"  {lab:52s} median {np.nanmedian(v):8.3f}   range {np.nanmin(v):.3f}-{np.nanmax(v):.3f}")
    say(f"  block-mean LB (permutation) rejects at 0.05 (either variable): "
        f"{sum(d['mean_lb_p_min'] < 0.05 for d in s2)}/{len(s2)}")
    say(f"  log-block-variance LB (permutation) rejects at 0.05 (either variable): "
        f"{sum(d['logvar_lb_p_min'] < 0.05 for d in s2)}/{len(s2)}")

    for name, rows in (("stage1.csv", s1), ("stage2.csv", s2)):
        with open(os.path.join(OUT, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    json.dump({"stage1_block_hours": STAGE1_BLOCK, "stage2_block_hours": STAGE2_BLOCK,
               "n_series": len(s1), "n_candidates": len(s2)},
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    say(f"\nwritten: {OUT}")
    log.close()


if __name__ == "__main__":
    main()
