#!/usr/bin/env python
"""Order-2 Renyi Gram form of DOMI by prefix sums: cost against n.

Purpose
-------
The von Neumann (alpha = 1) Gram form needs an eigendecomposition per segment (Section 4.8,
Table 2). The order-2 Renyi variant reported in Supplementary Tables B.3 / B.4 ("matmi-r2" in the
code) admits 2-D prefix sums: the order-2 entropy is a function of the sum of squared Gram entries,
and block sums of a fixed matrix are differences of 2-D prefix sums. This script measures the time
and memory of the prefix-sum computation against n (Supplementary Section B.9).

Algebra
-------
For a segment of m rows with Gaussian Gram matrix K (unit diagonal, so tr K = m), the normalized
spectrum is lambda = eig(K)/m, and

    H_2(K/m) = -log sum_k lambda_k^2 = -log tr[(K/m)^2] = -log( sum_{ij} K_ij^2 / m^2 ),

since K is symmetric. With K_X, K_Y the Gram matrices of the two blocks and K_X o K_Y the Gram
matrix of the product states (Hadamard product, also unit diagonal),

    I_2(segment) = H_2(K_X/m) + H_2(K_Y/m) - H_2(K_X o K_Y / m)
                 = log( S_XY * m^2 / (S_X * S_Y) ),
    S_X = sum_{i,j in seg} K_X,ij^2,   S_Y likewise,   S_XY = sum_{i,j in seg} (K_X,ij K_Y,ij)^2.

The kernel is exp(-gamma (u_i - u_j)^2) with gamma the median-heuristic value of
run_matmi_baseline.rff_gamma, so K^2 is the Gaussian kernel at 2 gamma. The difference curve is
sqrt(t(n-t)/n) |I_2(0,t) - I_2(t,n)| over the grid, the statistic "matmi-r2|g" of
run_matmi_baseline.

Two implementations
-------------------
  prefix2d   2-D prefix sums of K_X^2, K_Y^2 and (K_X K_Y)^2: any segment [a,b) in O(1); O(n^2) time, three n x n float64 arrays.
  stream     global split only: the sums over [0,t)^2 and over [t,n)^2 need only the running
             diagonal-block sums Q(t) = sum_{i,j<t} and the running row totals, since
             sum_{[t,n)^2} = total - 2 sum_{i<t, all j} + Q(t). Rows are generated in chunks, so the
             time is O(n^2) and the memory O(n x chunk).

Run
---
    python 00_SRC/gram_r2_prefix.py timing         # n = 600 ... 19200, one curve, single core
Outputs: 04_DAOU/EXPERIMENT/gram_r2_prefix/{timing.csv, summary.json}
timing.csv carries wall-clock seconds and peak resident memory (each (n, form) in a fresh process),
with the matching rows of scaling/scaling.csv (Table 2) copied alongside for comparison.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import run_matmi_baseline as R  # noqa: E402

OUT = os.path.join(R.ROOT, "04_DAOU", "EXPERIMENT", "gram_r2_prefix")
W = 60


# ------------------------------------------------------------------ prefix2d
def _sq_kernel_prefix(u, g):
    """Squared Gaussian Gram matrix K^2 = exp(-2 g (u_i - u_j)^2) (prefix-summed by _cum2)."""
    A = np.subtract.outer(u, u)
    np.square(A, out=A)
    A *= -2.0 * g
    np.exp(A, out=A)
    return A


def _cum2(A):
    np.cumsum(A, axis=0, out=A)
    np.cumsum(A, axis=1, out=A)
    return A


def _block(P, a, b):
    """Sum over [a,b) x [a,b) from an inclusive 2-D prefix array P (P[i,j] = sum_{<=i, <=j})."""
    s = P[b - 1, b - 1]
    if a > 0:
        s = s - P[a - 1, b - 1] - P[b - 1, a - 1] + P[a - 1, a - 1]
    return s


def r2_prefix2d(ux, uy, gx, gy, n, grid, mode="global", w=W):
    A = _sq_kernel_prefix(ux, gx)            # K_X^2
    B = _sq_kernel_prefix(uy, gy)            # K_Y^2
    C = A * B                                # (K_X o K_Y)^2
    _cum2(A)
    _cum2(B)
    _cum2(C)

    def I2(a, b):
        m = b - a
        return float(np.log(_block(C, a, b) * m * m / (_block(A, a, b) * _block(B, a, b))))

    out = np.empty(len(grid))
    for i, t in enumerate(grid):
        t = int(t)
        if mode == "global":
            (a1, b1), (a2, b2), wt = (0, t), (t, n), t * (n - t) / n
        else:
            (a1, b1), (a2, b2), wt = (t - w, t), (t, t + w), 1.0
        out[i] = np.sqrt(wt) * abs(I2(a1, b1) - I2(a2, b2))
    return out


# ------------------------------------------------------------------ stream (global only)
def r2_stream(ux, uy, gx, gy, n, grid, chunk=256):
    Q = np.zeros((3, n + 1))       # Q[:, t] = sum over [0,t)^2
    Rw = np.zeros((3, n + 1))      # Rw[:, t] = sum over rows < t, all columns
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        ex = np.exp(-2.0 * gx * (ux[i0:i1, None] - ux[None, :]) ** 2)
        ey = np.exp(-2.0 * gy * (uy[i0:i1, None] - uy[None, :]) ** 2)
        for q, M in enumerate((ex, ey, ex * ey)):
            tot = M.sum(1)
            cs = np.cumsum(M, axis=1)
            rows = np.arange(i0, i1)
            below = np.where(rows > 0, cs[np.arange(i1 - i0), np.maximum(rows - 1, 0)], 0.0)
            diag = M[np.arange(i1 - i0), rows]
            for r, t in enumerate(rows):                    # Q(t+1) = Q(t) + 2 sum_{j<t} + K_tt
                Q[q, t + 1] = Q[q, t] + 2.0 * below[r] + diag[r]
                Rw[q, t + 1] = Rw[q, t] + tot[r]
    total = Q[:, n]
    grid = np.asarray(grid)
    L = Q[:, grid]
    Rr = total[:, None] - 2.0 * Rw[:, grid] + Q[:, grid]
    mL, mR = grid, n - grid
    IL = np.log(L[2] * mL ** 2 / (L[0] * L[1]))
    IR = np.log(Rr[2] * mR ** 2 / (Rr[0] * Rr[1]))
    return np.sqrt(grid * (n - grid) / n) * np.abs(IL - IR)


# ------------------------------------------------------------------ timing
def one(n, form):
    """Time one full global curve at length n (fresh process; prints one JSON line)."""
    import run_scaling as S
    from dots.domi import DOMIContext
    X, Y = S.sample(n)
    t0 = time.perf_counter()
    ctx = DOMIContext(X, Y, W, D=8, seed=2026, n_perm=0)          # ranks (as run_scaling's setup)
    ux, uy = ctx.UX[:, 0].copy(), ctx.UY[:, 0].copy()
    gx, gy = R.rff_gamma(ctx.UX, 2026), R.rff_gamma(ctx.UY, 2027)
    grid = np.arange(W, n - W + 1)
    del ctx
    setup = time.perf_counter() - t0
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t1 = time.perf_counter()
    c = r2_prefix2d(ux, uy, gx, gy, n, grid) if form == "r2-prefix2d" else r2_stream(ux, uy, gx, gy, n, grid)
    scan = time.perf_counter() - t1
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss            # KiB on Linux
    print(json.dumps(dict(n=n, form=form, grid=len(grid), setup_sec=round(setup, 4),
                          curve_sec=round(scan, 3), scan_sec=round(setup + scan, 3),
                          peak_rss_MiB=round(peak / 1024, 1), rss_before_curve_MiB=round(rss0 / 1024, 1),
                          curve_max=float(c.max()), argmax=int(grid[int(np.argmax(c))]))))


def timing(ns=(600, 1200, 2400, 4800, 9600, 19200), forms=("r2-stream", "r2-prefix2d")):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for n in ns:
        for form in forms:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", str(n), form],
                               capture_output=True, text=True, env=dict(os.environ))
            if p.returncode != 0:
                rows.append(dict(n=n, form=form, error=p.stderr.strip().splitlines()[-1][:200]))
            else:
                rows.append(json.loads(p.stdout.strip().splitlines()[-1]))
            print(rows[-1], flush=True)
    ref = [r for r in csv.DictReader(open(os.path.join(R.ROOT, "04_DAOU", "EXPERIMENT", "scaling",
                                                       "scaling.csv")))]
    fields = ["n", "form", "grid", "setup_sec", "curve_sec", "scan_sec", "peak_rss_MiB", "rss_before_curve_MiB",
              "curve_max", "argmax", "error", "per_candidate_ms", "scan_measured", "source"]
    allrows = [dict(r, source="this run") for r in rows] + \
              [dict(n=r["n"], form=r["form"], grid=r["grid"], setup_sec=r["setup_sec"],
                    per_candidate_ms=r["per_candidate_ms"], scan_sec=r["scan_sec"],
                    scan_measured=r["scan_measured"], source="scaling/scaling.csv (Table 2)") for r in ref]
    with open(os.path.join(OUT, "timing.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, restval="")
        wr.writeheader()
        wr.writerows(allrows)
    gib = 2 ** 30
    mem = dict(
        bytes_per_nxn_float64="8 n^2",
        n_one_array_exceeds_16GiB=int(np.floor(np.sqrt(16 * gib / 8))),
        n_three_arrays_exceed_16GiB=int(np.floor(np.sqrt(16 * gib / 24))),
        note="prefix2d holds three n x n float64 prefix arrays; the stream form holds O(n x chunk).")
    return rows, mem


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "timing"
    if cmd == "one":
        one(int(sys.argv[2]), sys.argv[3])
    elif cmd == "timing":
        ns = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else \
            (600, 1200, 2400, 4800, 9600, 19200)
        rows, mem = timing(ns)
        sp = os.path.join(OUT, "summary.json")
        s = json.load(open(sp)) if os.path.exists(sp) else {}
        s["timing"] = rows
        s["memory"] = mem
        json.dump(s, open(sp, "w"), indent=1)
