#!/usr/bin/env python
"""Weekly financial Holevo partitioning (Section 6.8) with block-permutation calibration.

The protocol of run_exch_diag.py rejects pair exchangeability for the WEEKLY aggregates
(scale statistics; volatility clustering survives weekly summation), so pair-permutation
penalty calibration of the weekly segmentation is not valid there. This run mirrors stage one
of the weather analysis (Section 6.7): block permutation for BOTH the cost bias correction and the
penalty calibration, with block length taken from the protocol recommendation.

Writes 04_DAOU/EXPERIMENT/e6/results_block.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from dots.segtests import joint_features, D, SEED, _perm_index  # noqa: E402
import dots.segtests as _SEG  # noqa: E402
from run_e6_finance import load, align_pair, weekly_sum  # noqa: E402
from run_exch_diag import exch_diag  # noqa: E402
from dots.encode import MomentCache  # noqa: E402
from dots import pelt as P  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e6")


def bias_corrected_cost_block(J, grid, rng, block, n_perm=5):
    C = P.cost_matrix_vn(MomentCache(J), grid)
    Cp = np.zeros_like(C)
    for _ in range(n_perm):
        idx = _perm_index(len(J), rng, block=block)
        Cp += P.cost_matrix_vn(MomentCache(J[idx]), grid)
    return C - Cp / n_perm


def pelt_segment_block(x, y, step, betas, block, n_beta_perm=20, rng=None):
    """Holevo partitioning with block-permutation bias correction and penalty calibration."""
    rng = rng or np.random.default_rng(SEED)
    n = len(x)
    grid = np.arange(0, n + 1, step)
    J = joint_features(x, y, D)
    Cobs = bias_corrected_cost_block(J, grid, rng, block)
    obs = {b: [int(grid[i]) for i in P.pelt_from_costs(Cobs, b)] for b in betas}
    fa = {b: 0 for b in betas}
    for k in range(n_beta_perm):
        idx = _perm_index(n, rng, block=block)
        Jp = joint_features(x[idx], y[idx], D)
        Cp = bias_corrected_cost_block(Jp, grid, rng, block)
        ks = {b: len(P.pelt_from_costs(Cp, b)) for b in betas}
        for b in betas:
            fa[b] += ks[b] > 0
        _SEG.PERM_LOG.append(dict(tag=_SEG.PERM_TAG, kind="pelt_block_beta", label="",
                                  copy=k, value=next((b for b in betas if ks[b] == 0),
                                                     float("inf"))))
    return obs, {b: fa[b] / n_beta_perm for b in betas}


def main():
    res = {}
    spx, tnx = load("yahoo_gspc.csv"), load("yahoo_tnx.csv")
    ksp, fx = load("yahoo_kospi.csv"), load("yahoo_usdkrw.csv")
    pairs = {
        "SPX-TNX": align_pair(spx, tnx, ("logret", "diff")),
        "KOSPI-USDKRW": align_pair(ksp, fx, ("logret", "logret")),
    }
    betas = list(np.linspace(0.5, 12, 24))
    for name, (dates, x, y) in pairs.items():
        xw, yw = weekly_sum(x), weekly_sum(y)
        wk_dates = dates[::5][: len(xw)]
        diag = exch_diag(xw, yw, L=20, seed=3000 + len(name))
        block = diag["b_hat"]
        t0 = time.time()
        _SEG.PERM_TAG = f"e6a_block_{name}"
        obs, fa = pelt_segment_block(xw, yw, step=4, betas=betas, block=block)
        beta = next((b for b in betas if fa[b] <= 0.05), betas[-1])
        res[f"e6a_block_{name}"] = dict(
            n_weeks=len(xw), diag_decision=diag["decision"],
            diag_pvalues=diag["pvalues"], block=block, beta=beta, fa=fa[beta],
            cps_dates=[wk_dates[min(c, len(wk_dates) - 1)] for c in obs[beta]],
        )
        print(f"weekly block-calibrated {name}: {time.time()-t0:.0f}s block={block} "
              f"beta={beta:.1f} fa={fa[beta]:.2f} "
              f"CPs {res[f'e6a_block_{name}']['cps_dates']}", flush=True)
    _SEG.flush_perm_log(os.path.join(OUT, "records_perm_block.csv"))
    with open(os.path.join(OUT, "results_block.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("saved", os.path.join(OUT, "results_block.json"))


if __name__ == "__main__":
    main()
