#!/usr/bin/env python
"""Arithmetic summary of the unrestricted stage-two re-test (K = 999) of the weather application
(Supplementary Section B.8). The q-values reported in Section 6.7 (season-restricted stage two)
come from run_e8_season.py, which reuses step_up_q from this file.

Reads the 27 stage-two block-permutation p-values from 04_DAOU/EXPERIMENT/e8/retest_K999.json
(written by 00_SRC/run_e8_retest.py) and computes
  - the number of candidates with p <= 0.05 and the number expected under the global null (27 x 0.05);
  - P(X >= that number) for X ~ Binomial(27, 0.05), i.e. treating the candidates as independent;
  - Benjamini-Hochberg q-values and the counts at q <= 0.10 and q <= 0.05;
  - Benjamini-Yekutieli q-values (BH multiplied by the harmonic number H_27, capped at 1) and the
    strongest candidate's BY q-value.

Run:    python 00_SRC/run_e8_summary.py
Writes: 04_DAOU/EXPERIMENT/e8/retest_summary.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E8 = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8")
ALPHA = 0.05


def step_up_q(p, factor=1.0):
    """Adjusted q-values of the Benjamini-Hochberg step-up procedure, times `factor`."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    q = p[order] * m * factor / np.arange(1, m + 1)
    q = np.minimum(np.minimum.accumulate(q[::-1])[::-1], 1.0)
    out = np.empty(m)
    out[order] = q
    return out


def main():
    res = json.load(open(os.path.join(E8, "retest_K999.json")))["results"]
    p = np.array([r["p_block"] for r in res])
    m = len(p)
    k = int((p <= ALPHA).sum())
    h = float(np.sum(1.0 / np.arange(1, m + 1)))
    q_bh = step_up_q(p)
    q_by = step_up_q(p, h)
    i0 = int(np.argmin(p))
    out = {"n_candidates": m,
           "n_p_le_05": k,
           "expected_p_le_05_under_global_null": m * ALPHA,
           "binomial_tail_P_ge_k": float(stats.binom.sf(k - 1, m, ALPHA)),
           "n_bh_q_le_10": int((q_bh <= 0.10).sum()),
           "n_bh_q_le_05": int((q_bh <= 0.05).sum()),
           "harmonic_number_H_m": h,
           "n_by_q_le_10": int((q_by <= 0.10).sum()),
           "strongest": {"stn": res[i0]["stn"], "pair": res[i0]["pair"], "date": res[i0]["date"],
                         "p": float(p[i0]), "bh_q": float(q_bh[i0]), "by_q": float(q_by[i0])},
           "per_candidate": [{"stn": r["stn"], "pair": r["pair"], "date": r["date"],
                              "p": float(p[i]), "bh_q": float(q_bh[i]), "by_q": float(q_by[i])}
                             for i, r in enumerate(res)]}
    with open(os.path.join(E8, "retest_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print({k_: v for k_, v in out.items() if k_ != "per_candidate"})
    print("saved", os.path.join(E8, "retest_summary.json"))


if __name__ == "__main__":
    main()
