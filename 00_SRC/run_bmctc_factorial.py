#!/usr/bin/env python
"""Why is BMCTC more powerful than DOMI on S4? A 2 x 2 design over input and kernel rule.

BMCTC as reimplemented (dots/bmctc.py) differs from DOMI in its input and in its kernel rule. This run changes
one factor at a time, with everything else fixed (60-point moving windows, step 1, Renyi orders 1.01 and 2,
the Bernaola-Galvan statistic, the replicates and seeds of run_bmctc.py, and the calibration of Table B.9:
500 nulls, raw and studentized thresholds, localization tolerance 30):

  input   raw   the observed values (as BMCTC)
          rank  whole-window ranks rank/(n + 1) (as DOMI)
  kernel  bmctc each 60-point window standardized to mean 0 and sd 1, Gaussian width 1.06 * 60^(-1/5)
                (Silverman's rule; as BMCTC)
          domi  no window standardization; Gaussian exp(-gamma d^2) with the median-heuristic gamma that
                DOMI's random features use, computed once on the whole input (run_matmi_baseline.rff_gamma,
                seeds 2026 and 2027)

  A = raw/bmctc (BMCTC itself; checked against run_bmctc.py), B = raw/domi, C = rank/bmctc, D = rank/domi.
Cells: S4 r = 0.7 and 0.85 (matched nulls) and M1 s = 2 (false-alarm control).

Run:    python 00_SRC/run_bmctc_factorial.py --procs 12
Writes 04_DAOU/EXPERIMENT/bmctc_factorial/{config.json, tc/<variant>_<set>.npz, results.csv, summary.json}
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
import run_bmctc as RB  # noqa: E402
from dots import bmctc as B  # noqa: E402
from dots.domi import ranks01  # noqa: E402
from dots.extras import studentize  # noqa: E402
from run_matmi_baseline import rff_gamma  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "bmctc_factorial")
TC = os.path.join(OUT, "tc")
CELLS = ["S4_r0.7", "S4_r0.85", "M1_s2"]
VARIANTS = {"A": ("raw", "bmctc"), "B": ("raw", "domi"), "C": ("rank", "bmctc"), "D": ("rank", "domi")}
CFG = dict(RB.CFG, cells=CELLS, variants={k: list(v) for k, v in VARIANTS.items()})


def tc_series(x, y, inp, kern, L=B.L_WIN, alphas=B.ALPHAS, chunk=256):
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    if inp == "rank":
        x, y = ranks01(x[:, None])[:, 0], ranks01(y[:, None])[:, 0]
    Wx, Wy = sliding_window_view(x, L), sliding_window_view(y, L)
    if kern == "bmctc":
        sig = 1.06 * L ** (-0.2)
        gx = gy = 1.0 / (2.0 * sig * sig)
    else:
        gx, gy = rff_gamma(x[:, None], 2026), rff_gamma(y[:, None], 2027)
    out = {a: np.empty(len(Wx)) for a in alphas}
    for s in range(0, len(Wx), chunk):
        zx, zy = Wx[s:s + chunk], Wy[s:s + chunk]
        if kern == "bmctc":
            zx = (zx - zx.mean(1, keepdims=True)) / zx.std(1, ddof=1, keepdims=True)
            zy = (zy - zy.mean(1, keepdims=True)) / zy.std(1, ddof=1, keepdims=True)
        dx = zx[:, :, None] - zx[:, None, :]
        dy = zy[:, :, None] - zy[:, None, :]
        Kx, Ky = np.exp(-gx * dx * dx), np.exp(-gy * dy * dy)
        lx, ly, lxy = np.linalg.eigvalsh(Kx), np.linalg.eigvalsh(Ky), np.linalg.eigvalsh(Kx * Ky)
        for a in alphas:
            out[a][s:s + chunk] = B._renyi(lx, a) + B._renyi(ly, a) - B._renyi(lxy, a)
    return out


def _job(a):
    var, setname, r = a
    x, y = RB.draw(setname, r)
    return tc_series(x, y, *VARIANTS[var], CFG["L"], tuple(CFG["alphas"]))


def sets():
    out = []
    for c in CELLS:
        if RB.null_set(c) not in out:
            out.append(RB.null_set(c))
    return out + [f"alt_{c}" for c in CELLS]


def compute(procs):
    os.makedirs(TC, exist_ok=True)
    with Pool(procs) as pool:
        for v in VARIANTS:
            for s in sets():
                p = os.path.join(TC, f"{v}_{s}.npz")
                if os.path.exists(p):
                    continue
                res = pool.map(_job, [(v, s, r) for r in range(CFG["reps"])], chunksize=5)
                tmp = os.path.join(TC, f".{v}_{s}.tmp.npz")
                np.savez_compressed(tmp, **{f"a{a}": np.array([d[a] for d in res]) for a in CFG["alphas"]})
                os.replace(tmp, p)
                print(f"{v} {s} done", flush=True)


def check_A():
    """Variant A must reproduce the stored BMCTC sequences of run_bmctc.py exactly."""
    worst = 0.0
    for s in sets():
        mine = np.load(os.path.join(TC, f"A_{s}.npz"))
        ref = np.load(os.path.join(RB.TC, f"{s}.npz"))
        for a in CFG["alphas"]:
            worst = max(worst, float(np.max(np.abs(mine[f"a{a}"] - ref[f"a{a}"]))))
    return worst


def evaluate():
    n, w, tol, fpr, tau = CFG["n"], CFG["w"], CFG["tol"], CFG["fpr"], CFG["tau"]
    grid = np.arange(w, n - w + 1)
    half = CFG["reps"] // 2
    rows = []
    for v in VARIANTS:
        for c in CELLS:
            load = lambda s: {a: RB.curves(np.load(os.path.join(TC, f"{v}_{s}.npz"))[f"a{a}"]) for a in CFG["alphas"]}
            cur0, cur1 = load(RB.null_set(c)), load(f"alt_{c}")
            for a in CFG["alphas"]:
                mu, sd = cur0[a][:half].mean(0), cur0[a][:half].std(0)
                for form in ("raw", "studentised"):
                    if form == "raw":
                        thr = float(np.quantile(cur0[a].max(1), 1 - fpr)); C = cur1[a]
                    else:
                        thr = float(np.quantile([np.max(studentize(z, mu, sd)) for z in cur0[a][half:]], 1 - fpr))
                        C = np.array([studentize(z, mu, sd) for z in cur1[a]])
                    mx, th = C.max(1), grid[C.argmax(1)]
                    det = mx > thr; hit = det & (np.abs(th - tau) <= tol)
                    rows.append(dict(variant=v, input=VARIANTS[v][0], kernel=VARIANTS[v][1], cell=c, stat=RB.STAT[a],
                                     form=form, detect_rate=round(float(det.mean()), 4),
                                     power_localised=round(float(hit.mean()), 4), n_reps=len(C)))
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); wr.writeheader(); wr.writerows(rows)
    dA = check_A()
    json.dump(dict(config=CFG, variant_A_max_abs_diff_vs_run_bmctc=dA), open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print("variant A vs stored BMCTC sequences, max |diff| =", dA)
    for x in rows:
        print(f"{x['variant']} {x['input']:4s} {x['kernel']:5s} {x['cell']:9s} {x['stat']:12s} {x['form']:12s} "
              f"detect {x['detect_rate']:.3f} localized {x['power_localised']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cp = os.path.join(OUT, "config.json")
    if not os.path.exists(cp):
        json.dump(CFG, open(cp, "w"), indent=1)
    elif json.load(open(cp)) != json.loads(json.dumps(CFG)):
        sys.exit("config.json differs from CFG: use a new output folder")
    if not a.evaluate:
        compute(a.procs)
    evaluate()


if __name__ == "__main__":
    main()
