"""Feature-map fact quoted in Supplementary Section A.8 ([NV-3]).

The deployed map divides sqrt(2/D) cos(uW + b) by its own norm; the quotient is smooth on [0,1]
iff that norm is bounded below there. Reported: min and max of the unnormalized norm over u in
[0,1] (20,000-point grid) for the deployed draws (seeds 2026, 2027) at D = 4 and 8, at the
median-heuristic bandwidth of rank inputs.

Run:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python 00_SRC/check_rank_correction_algebra.py
Writes 04_DAOU/EXPERIMENT/theory_rank_process/rank_correction_algebra.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dots.domi import ranks01  # noqa: E402
from dots.extras import rff  # noqa: E402
from check_rank_process_boundary import deployed_gamma  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "theory_rank_process")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20260925)
    res = {}
    ug = (np.arange(20000) + 0.5) / 20000          # quadrature on [0,1]
    gamma = deployed_gamma(ranks01(rng.uniform(size=600)), 2026)
    res["gamma_rank_inputs_n600"] = gamma

    # norm range of the unnormalized random-phase map
    nb = {}
    for Dd in (4, 8):
        def rng_norm(seed):
            F = rff(ug[:, None], D=Dd, seed=seed, gamma=gamma)
            nn = np.linalg.norm(F, axis=1)
            return nn.min(), nn.max()
        dep = {s: rng_norm(s) for s in (2026, 2027)}
        nb[f"D{Dd}"] = {"deployed_seed2026_min_max": dep[2026], "deployed_seed2027_min_max": dep[2027]}
    res["unnormalised_norm_on_unit_interval"] = nb

    with open(os.path.join(OUT, "rank_correction_algebra.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
