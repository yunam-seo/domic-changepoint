#!/usr/bin/env python
"""Wall-clock cost of the two computational forms of DOMI against the series length n (Table 2).

One replicate of scenario D2 (the article's S2, level 0.7) per n; the curve is the global weighted
difference over the full candidate grid (window w = 60, stride 1), without permutation replicas
(calibration multiplies every form's cost by the same factor K + 1).

Forms timed:
  primal-D8, primal-D16   random-feature form: density operators from prefix sums of the segment
                          moments (one O(n D^4) pass, then two D x D and one D^2 x D^2
                          eigendecomposition per segment, independent of the segment length)
  dual-full               Gram form of the article: exact Gaussian kernel on every row of the
                          segment (per-candidate cost grows with the segment length)
  dual-m256               Gram form on a deterministic subsample of at most 256 rows per segment,
                          reported only in Supplementary Sections B.9 and B.10

Per-candidate cost: median over 15 candidates spread over the grid. Full-scan cost: measured
directly where affordable (FULL_SCAN_MAX below), otherwise per-candidate median times grid size,
marked scan_measured = False (an asterisk in Table 2).

Run:  python 00_SRC/run_scaling.py
Output: 04_DAOU/EXPERIMENT/scaling/{scaling.csv, run.log}
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, csv, sys, time  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_matmi_baseline as R  # noqa: E402
from dots.domi import DOMIContext  # noqa: E402
from dots.synth import generate  # noqa: E402

ROOT = R.ROOT
OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "scaling")
W = 60
FULL_SCAN_MAX = {"primal-D8": 10 ** 9, "primal-D16": 10 ** 9, "dual-m256": 4800, "dual-full": 600}
PERCAND_MAX = {"dual-full": 4800}


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(os.path.join(OUT, "run.log"), "a") as f:
        f.write(line + "\n")


def sample(n):
    smp = generate("D2", 0.7, 0, n=n, tau=n // 2, null=False, base_seed=20260825, level_idx=1)
    Z = smp["Z"]
    return Z[:, smp["blocks"][0]], Z[:, smp["blocks"][1]]


def primal_setup(X, Y, D):
    t0 = time.perf_counter()
    ctx = DOMIContext(X, Y, W, D=D, seed=2026, n_perm=0)
    return ctx, time.perf_counter() - t0


def primal_cand(ctx, t):
    ctx._ent.clear()
    return ctx.domi(0, t) - ctx.domi(t, ctx.n)


def dual_cand(ux, uy, gx, gy, t, n, base):
    a = R.matmi_seg(ux[R._sub(0, t, base)], uy[R._sub(0, t, base)], gx, gy)[0]
    b = R.matmi_seg(ux[R._sub(t, n, base)], uy[R._sub(t, n, base)], gx, gy)[0]
    return a - b


def time_form(form, n, X, Y, cands):
    """Returns (setup_sec, per-candidate median sec, full-scan sec, scan_measured)."""
    grid = np.arange(W, n - W + 1)
    if form.startswith("primal"):
        ctx, setup = primal_setup(X, Y, int(form.split("D")[1]))
        f = lambda t: primal_cand(ctx, t)  # noqa: E731
    else:
        t0 = time.perf_counter()
        ctx = DOMIContext(X, Y, W, D=8, seed=2026, n_perm=0)          # ranks only
        ux, uy = ctx.UX[:, 0], ctx.UY[:, 0]
        gx, gy = R.rff_gamma(ctx.UX, 2026), R.rff_gamma(ctx.UY, 2027)
        setup = time.perf_counter() - t0
        R.MSUB = 256 if form == "dual-m256" else 10 ** 9
        base = [R.SUBSEED, 99, 0, n, 0]
        f = lambda t: dual_cand(ux, uy, gx, gy, t, n, base)  # noqa: E731
    per = []
    for t in cands:
        t0 = time.perf_counter()
        f(int(t))
        per.append(time.perf_counter() - t0)
    per_med = float(np.median(per))
    if n <= FULL_SCAN_MAX[form]:
        t0 = time.perf_counter()
        for t in grid:
            f(int(t))
        return setup, per_med, setup + time.perf_counter() - t0, True
    return setup, per_med, setup + per_med * len(grid), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="600,1200,2400,4800,9600,19200")
    ap.add_argument("--forms", default="primal-D8,primal-D16,dual-m256,dual-full")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    say(f"ns={a.ns} forms={a.forms}")
    rows = []
    for n in map(int, a.ns.split(",")):
        X, Y = sample(n)
        grid = np.arange(W, n - W + 1)
        cands = grid[np.linspace(0, len(grid) - 1, 15).astype(int)]
        for form in a.forms.split(","):
            if n > PERCAND_MAX.get(form, 10 ** 9):
                continue
            setup, per, scan, measured = time_form(form, n, X, Y, cands)
            r = dict(n=n, form=form, grid=len(grid), setup_sec=round(setup, 4),
                     per_candidate_ms=round(per * 1e3, 3), scan_sec=round(scan, 2),
                     scan_measured=measured)
            rows.append(r)
            say(" ".join(f"{k}={v}" for k, v in r.items()))
    with open(os.path.join(OUT, "scaling.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    say("done")


if __name__ == "__main__":
    main()
