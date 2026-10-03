"""Power and localization summaries of a statistic curve at a calibrated threshold.
"""
from __future__ import annotations

import numpy as np


def summarize_alt(stats: dict, grid: np.ndarray, tau: int, w: int, n: int, thresholds: dict, tol: int) -> dict:
    out = {}
    for k, v in stats.items():
        h = thresholds.get(k, np.inf)
        mx = float(np.nanmax(v))
        tau_hat = int(grid[int(np.nanargmax(v))])
        detected = mx > h
        loc_ok = abs(tau_hat - tau) <= tol
        rec = dict(max=mx, tau_hat=tau_hat, detected=detected, power_hit=bool(detected and loc_ok),
                   loc_err=abs(tau_hat - tau) if detected else np.nan)
        out[k] = rec
    return out


def aggregate(alt_records: list[dict], thresholds: dict) -> dict:
    """alt_records: list over reps of dict method -> rec. Returns method -> summary."""
    methods = alt_records[0].keys()
    agg = {}
    for k in methods:
        recs = [r[k] for r in alt_records]
        power = float(np.mean([r["power_hit"] for r in recs]))
        det = float(np.mean([r["detected"] for r in recs]))
        loc = [r["loc_err"] for r in recs if r["detected"]]
        s = dict(power=power, detect_rate=det, loc_err_median=float(np.median(loc)) if loc else np.nan,
                 threshold=thresholds.get(k, np.nan), n_reps=len(recs))
        agg[k] = s
    return agg
