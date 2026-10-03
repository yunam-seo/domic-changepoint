#!/usr/bin/env python
"""Exchangeability diagnostic protocol for choosing the permutation calibration.

The exactness of pair permutation (Proposition P1) holds under exchangeability of the pairs;
that of block permutation (Proposition P5) under exchangeability of blocks. Exchangeability cannot be established from one
realization, but its two practically consequential violations — serial correlation in the
levels and serial dependence in the scale, exactly the two mechanisms Section 4.3 shows to
break the level — are testable on the rank sequences. After ranking, the margins are
identical by construction, so serial independence of the ranks is the content of the i.i.d.
sufficient condition.

Protocol (per bivariate segment):
  1. Centered ranks z_t = u_t - 1/2 for each margin.
  2. Ljung-Box portmanteau at L lags on (i) z_x, (ii) z_y  [levels],
     (iii) z_x^2, (iv) z_y^2 [scale], and (v) a cross-portmanteau on rho_xy(k), 0<|k|<=L.
     Bonferroni over the five statistics at level alpha.
  3. No rejection  -> pair permutation.
     Any rejection -> block permutation, block length b_hat = ceil(5 * tau_hat), where
     tau_hat is the largest integrated autocorrelation time (initial-positive-sequence
     truncation) over the four univariate sequences. The factor 5 makes the dependence
     carried across one block boundary ~ e^-5 for AR-type decay.

Caveats (Section 4.3): non-rejection establishes nothing, and because the scheme and block
length are chosen from the same data, which Propositions P1 and P5 do not cover, the level of
the protocol is not proved (Supplementary Section B.17).

Runs:
  A. Synthetic validation on the Section 4.3 null designs (iid / ar1 / garch, same
     generator and seeds as run_block_perm_check.py): protocol decision rates and the
     distribution of recommended block lengths.
  B. Weather (hourly anomalies, 12 stations x 3 pairs): decisions, tau_hat, b_hat in hours.
  C. Finance: daily stock-bond pair on the 2021-2022 window, and the weekly aggregate
     of the full 15y series (the object segmented in Section 6.8).

Writes 04_DAOU/EXPERIMENT/exch_diag/results.json
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
from scipy import stats

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.perm import ge as perm_ge  # noqa: E402

from run_block_perm_check import gen_null, CFG  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "exch_diag")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- core statistics
def _ranks01(x):
    n = len(x)
    return (stats.rankdata(x) / (n + 1.0)) - 0.5


def _acf(z, L):
    """Autocorrelations r_1..r_L via FFT."""
    z = np.asarray(z, float) - np.mean(z)
    n = len(z)
    f = np.fft.rfft(z, 2 * n)
    s = np.fft.irfft(f * np.conj(f))[: L + 1]
    return s[1:] / s[0]


def _ljung_box(z, L):
    n = len(z)
    r = _acf(z, L)
    q = n * (n + 2.0) * np.sum(r**2 / (n - np.arange(1, L + 1)))
    return float(q), float(stats.chi2.sf(q, L))


def _cross_portmanteau(zx, zy, L):
    """Portmanteau on cross-correlations at lags 1..L in both directions (~chi2_{2L}).

    FFT-based (zero-padded, linear cross-covariance) so that permutation calibration
    stays feasible on 70k-hour series.
    """
    n = len(zx)
    zx = zx - zx.mean()
    zy = zy - zy.mean()
    sx = np.sqrt(np.sum(zx**2))
    sy = np.sqrt(np.sum(zy**2))
    m = 2 * n
    fx = np.fft.rfft(zx, m)
    fy = np.fft.rfft(zy, m)
    cc = np.fft.irfft(fx * np.conj(fy), m)
    ks = np.arange(1, L + 1)
    r_pos = cc[1: L + 1] / (sx * sy)          # one direction
    r_neg = cc[m - L:][::-1] / (sx * sy)      # the other
    q = float(np.sum(n * (n + 2.0) * (r_pos**2 + r_neg**2) / (n - ks)))
    return q, float(stats.chi2.sf(q, 2 * L))


def _tau_int(z, kmax):
    """Integrated autocorrelation time, initial-positive-sequence truncation."""
    r = _acf(z, kmax)
    s = 0.0
    for rk in r:
        if rk <= 0:
            break
        s += rk
    return 1.0 + 2.0 * s


def _five_stats(zx, zy, L):
    """The five portmanteau statistics (values only) on centered-rank sequences."""
    return np.array([
        _ljung_box(zx, L)[0],
        _ljung_box(zy, L)[0],
        _ljung_box(zx**2, L)[0],
        _ljung_box(zy**2, L)[0],
        _cross_portmanteau(zx, zy, L)[0],
    ])


def exch_diag(x, y, L=20, alpha=0.05, kmax=None, K=199, seed=0):
    """Run the protocol on one bivariate segment. Returns decision and block length.

    The five statistics are calibrated by joint pair permutation, which is exact under
    the exchangeable null by the same group argument as Proposition P1 (any statistic's permutation
    distribution is its exact null distribution under exchangeability). Rejection is
    therefore finite-sample-exact evidence against pair exchangeability; non-rejection
    establishes nothing. Chi-square asymptotic p-values are kept as a cross-check.
    """
    n = len(x)
    kmax = kmax or min(n // 4, 5000)
    zx, zy = _ranks01(x), _ranks01(y)
    names = ["lb_level_x", "lb_level_y", "lb_scale_x", "lb_scale_y", "lb_cross"]
    obs = _five_stats(zx, zy, L)
    # permutation calibration: permute the PAIRS jointly (ranks of a permuted series
    # are the permuted ranks, so permuting z is enough)
    rng = np.random.default_rng(20260828 + seed)
    ge = np.ones(5)                              # count of perm >= obs, starts at 1 (the identity)
    for _ in range(K):
        pi = rng.permutation(n)
        ge += perm_ge(_five_stats(zx[pi], zy[pi], L), obs)
    pperm = ge / (K + 1.0)
    # asymptotic cross-check
    pasy = np.array([
        _ljung_box(zx, L)[1], _ljung_box(zy, L)[1],
        _ljung_box(zx**2, L)[1], _ljung_box(zy**2, L)[1],
        _cross_portmanteau(zx, zy, L)[1],
    ])
    thr = alpha / 5.0
    reject = {nm: bool(p <= thr) for nm, p in zip(names, pperm)}
    any_reject = any(reject.values())
    tau = max(_tau_int(z, kmax) for z in (zx, zy, zx**2, zy**2))
    b_hat = int(np.ceil(5.0 * tau))
    return {
        "n": int(n), "L": int(L), "alpha": alpha, "K": int(K),
        "pvalues": {nm: float(p) for nm, p in zip(names, pperm)},
        "pvalues_asymptotic": {nm: float(p) for nm, p in zip(names, pasy)},
        "reject": reject,
        "decision": "block" if any_reject else "pair",
        "tau_hat": float(tau), "b_hat": b_hat,
    }


# ---------------------------------------------------------------- A. synthetic validation
def synth_validation(reps=200, L=20):
    out = {}
    rec = []
    for kind in ["iid", "ar1", "garch"]:
        decisions, bhats, rej_counts = [], [], {k: 0 for k in
            ["lb_level_x", "lb_level_y", "lb_scale_x", "lb_scale_y", "lb_cross"]}
        for rep in range(reps):
            x, y = gen_null(kind, rep, CFG)
            d = exch_diag(x, y, L=L, seed=rep)
            decisions.append(d["decision"])
            if d["decision"] == "block":
                bhats.append(d["b_hat"])
            for k, v in d["reject"].items():
                rej_counts[k] += int(v)
            rec.append(dict(kind=kind, rep=rep, decision=d["decision"],
                            b_hat=d.get("b_hat", ""), **{k: int(v) for k, v in d["reject"].items()}))
        block_rate = decisions.count("block") / reps
        out[kind] = {
            "reps": reps,
            "block_rate": block_rate,
            "reject_rates_by_test": {k: v / reps for k, v in rej_counts.items()},
            "b_hat_median": float(np.median(bhats)) if bhats else None,
            "b_hat_iqr": [float(np.percentile(bhats, 25)), float(np.percentile(bhats, 75))]
            if bhats else None,
        }
        print(f"[synth {kind:6s}] block-decision rate {block_rate:.3f}  "
              f"median b_hat {out[kind]['b_hat_median']}")
    with open(os.path.join(OUT, "records_synth_reps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec[0].keys())); w.writeheader(); w.writerows(rec)
    return out


# ---------------------------------------------------------------- B. weather
NAMES = {"105": "Gangneung", "108": "Seoul", "112": "Incheon", "119": "Suwon",
         "129": "Seosan", "133": "Daejeon", "143": "Daegu", "146": "Jeonju",
         "156": "Gwangju", "159": "Busan", "165": "Mokpo", "184": "Jeju"}
PAIRS = [("ta", "hm"), ("ta", "ws"), ("hm", "ws")]


def weather_diag(L=168):
    npz = np.load(os.path.join(ROOT, "02_MART", "WEATHER_HOURLY_ANOM.npz"), allow_pickle=True)
    rows = []
    for si, stn in enumerate(NAMES):
        for pj, (va, vb) in enumerate(PAIRS):
            ka, kb = f"{stn}|{va}", f"{stn}|{vb}"
            if ka not in npz or kb not in npz:
                continue
            x, y = npz[ka], npz[kb]
            m = np.isfinite(x) & np.isfinite(y)
            d = exch_diag(x[m], y[m], L=L, kmax=8760, seed=1000 + 10 * si + pj)
            rows.append({"station": NAMES[stn], "pair": f"{va}-{vb}", **d})
    dec = [r["decision"] for r in rows]
    bh = [r["b_hat"] for r in rows]
    summary = {
        "n_analyses": len(rows),
        "all_block": all(v == "block" for v in dec),
        "max_pvalue_any_test": max(max(r["pvalues"].values()) for r in rows),
        "b_hat_hours_min": int(min(bh)), "b_hat_hours_median": float(np.median(bh)),
        "b_hat_hours_max": int(max(bh)),
        "b_hat_weeks_median": float(np.median(bh)) / 168.0,
        "b_hat_weeks_max": max(bh) / 168.0,
    }
    print(f"[weather] {len(rows)} analyses, all block: {summary['all_block']}, "
          f"b_hat hours median {summary['b_hat_hours_median']:.0f} "
          f"(~{summary['b_hat_weeks_median']:.1f} wk), max {summary['b_hat_hours_max']} "
          f"(~{summary['b_hat_weeks_max']:.1f} wk)")
    return {"summary": summary, "rows": rows}


# ---------------------------------------------------------------- C. finance
def _load_close(name):
    p = os.path.join(ROOT, "01_ORG", "FINANCE", name)
    rows = list(csv.DictReader(open(p)))
    return {r["date"]: float(r["close"]) for r in rows}


def finance_diag(L=20):
    spx, tnx = _load_close("yahoo_gspc.csv"), _load_close("yahoo_tnx.csv")
    dates = sorted(set(spx) & set(tnx))
    px = np.array([spx[d] for d in dates])
    yl = np.array([tnx[d] for d in dates])
    r = np.diff(np.log(px))
    dy = np.diff(yl)
    ddates = dates[1:]

    out = {}
    # daily 2021-2022 window (the Section 6.8 single-break test)
    m = np.array([("20210101" <= d <= "20221231") for d in ddates])
    out["daily_2021_2022"] = exch_diag(r[m], dy[m], L=L, seed=2000)
    print(f"[finance daily 21-22] decision {out['daily_2021_2022']['decision']}, "
          f"pvals {out['daily_2021_2022']['pvalues']}, b_hat {out['daily_2021_2022']['b_hat']}")

    # weekly aggregates of the full series (the object segmented in Section 6.8)
    k = 5
    nw = len(r) // k
    rw = r[: nw * k].reshape(nw, k).sum(axis=1)
    dyw = dy[: nw * k].reshape(nw, k).sum(axis=1)
    out["weekly_full"] = exch_diag(rw, dyw, L=L, seed=2001)
    print(f"[finance weekly full] decision {out['weekly_full']['decision']}, "
          f"pvals {out['weekly_full']['pvalues']}, b_hat {out['weekly_full']['b_hat']}")
    return out


# ---------------------------------------------------------------- main
def main():
    res = {
        "protocol": {
            "tests": ["LB(rank levels) x2", "LB(rank squares) x2", "cross-portmanteau"],
            "bonferroni_alpha": 0.05,
            "block_length_rule": "b_hat = ceil(5 * tau_int), tau_int = initial-positive-sequence "
                                 "integrated autocorrelation time, max over the 4 univariate rank sequences",
        },
        "synthetic": synth_validation(),
        "weather": weather_diag(),
        "finance": finance_diag(),
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved", os.path.join(OUT, "results.json"))


if __name__ == "__main__":
    main()
