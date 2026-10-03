#!/usr/bin/env python
"""Mechanism behind the low power on the copula-shape scenario (Section 6.4 and Supplementary
Section A.8; paper S3 = code D3). Parts (1) and (2) are quoted in the article; part (3) is a
supporting check whose values are not quoted.

(1) Population-level DOMI of each regime (n=2e5 per segment, pipeline convention: global ranks over
    the whole series, a single RFF draw shared by both segments), for code D3 and D2.
(2) Sampling standard deviation of the plug-in DOMI at segment length m=300 under independence and
    under a Gaussian copula at tau=0.5 -> the noise term of the signal-to-noise argument.
(3) Median-heuristic bandwidth gamma on rank inputs vs raw inputs -> the claim that on ranks the
    bandwidth is distribution-free up to subsampling noise.

Writes 04_DAOU/EXPERIMENT/s3_mechanism/results.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.synth import generate  # noqa: E402
from dots.domi import ranks01, unit_rff  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "s3_mechanism")
NSEG = 200_000
M_SEG = 300
REPS = 400
D = 8


REC = []


def ent(M):
    lam = np.clip(np.linalg.eigvalsh(M), 1e-300, None)
    lam = lam[lam > 1e-14]
    return float(-(lam * np.log(lam)).sum())


def domi_from_features(fx, fy):
    rX = fx.T @ fx / len(fx)
    rY = fy.T @ fy / len(fy)
    P = np.einsum("ti,tj->tij", fx, fy).reshape(len(fx), -1)
    return ent(rX) + ent(rY) - ent(P.T @ P / len(P))


def population_domi(scen, levels):
    """Segment DOMI of each regime under the pipeline convention (global ranks, shared RFF draw)."""
    rows = []
    for lv in levels:
        Z = generate(scen, lv, 0, n=2 * NSEG, tau=NSEG, null=False, level_idx=1)["Z"]
        FX = unit_rff(ranks01(Z[:, 0][:, None]), D, 2026)
        FY = unit_rff(ranks01(Z[:, 1][:, None]), D, 2027)
        pre = domi_from_features(FX[:NSEG], FY[:NSEG])
        post = domi_from_features(FX[NSEG:], FY[NSEG:])
        rows.append(dict(level=lv, I_pre=round(pre, 4), I_post=round(post, 4), abs_diff=round(abs(pre - post), 4)))
    return rows


def _regime_draw(scen, level, rep, half):
    """One segment of length M_SEG drawn from the pre- ('L') or post-change ('R') regime of a
    scenario, using the project's own generators so the regimes match the power experiments."""
    Z = generate(scen, level, 10_000 + rep, n=2 * M_SEG, tau=M_SEG, null=False, level_idx=1)["Z"]
    sl = slice(0, M_SEG) if half == "L" else slice(M_SEG, 2 * M_SEG)
    return Z[sl, 0], Z[sl, 1]


def sampling_sd():
    """Sampling sd of the plug-in segment DOMI in each regime, at segment length m = M_SEG.

    Reported per regime and, for the difference statistic actually used, as sqrt(sd_L^2 + sd_R^2):
    the change statistic differences two independent segments, so this is the like-for-like noise.
    """
    out = {}
    global REC
    for tag, scen, level in (("S2_pre_independent", "D2", 0.7), ("S2_post_a0.7", "D2", 0.7),
                             ("S3_pre_gaussian_tau0.5", "D3", 0.5), ("S3_post_clayton_tau0.5", "D3", 0.5)):
        half = "L" if "pre" in tag else "R"
        v = []
        for r in range(REPS):
            x, y = _regime_draw(scen, level, r, half)
            fx = unit_rff(ranks01(np.asarray(x, float)[:, None]), D, 2026)
            fy = unit_rff(ranks01(np.asarray(y, float)[:, None]), D, 2027)
            v.append(domi_from_features(fx, fy))
        v = np.asarray(v)
        REC += [dict(tag=tag, rep=i, domi=float(x)) for i, x in enumerate(v)]
        out[tag] = dict(mean=round(float(v.mean()), 4), sd=round(float(v.std()), 4))
    for pair, name in ((("S2_pre_independent", "S2_post_a0.7"), "S2_diff_sd"),
                       (("S3_pre_gaussian_tau0.5", "S3_post_clayton_tau0.5"), "S3_diff_sd")):
        sl, sr = out[pair[0]]["sd"], out[pair[1]]["sd"]
        out[name] = round(float(np.hypot(sl, sr)), 4)
    out["diff_sd_ratio_S3_over_S2"] = round(out["S3_diff_sd"] / out["S2_diff_sd"], 2)
    return out


def gamma_stability():
    """Median heuristic gamma (as implemented in dots.extras.rff) on rank vs raw inputs."""
    def gamma_of(Z, seed):
        Z = np.asarray(Z, float)
        idx = np.random.default_rng(seed + 1).choice(len(Z), size=min(len(Z), 400), replace=False)
        S = Z[idx]
        sq = (S * S).sum(1)
        D2 = np.clip(sq[:, None] + sq[None, :] - 2 * S @ S.T, 0, None)
        med = np.median(D2[np.triu_indices(len(idx), 1)])
        return 1.0 / (med if med > 0 else 1.0)

    rng = np.random.default_rng(0)
    gr, graw = [], []
    for s in range(40):
        raw = rng.standard_normal((600, 1)) * rng.choice([1, 3])
        gr.append(gamma_of(ranks01(raw), s))
        graw.append(gamma_of(raw, s))
    f = lambda a: dict(mean=round(float(np.mean(a)), 3), sd=round(float(np.std(a)), 3),
                       cv=round(float(np.std(a) / np.mean(a)), 3))
    return dict(rank_inputs=f(gr), raw_inputs=f(graw))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = dict(
        config=dict(n_per_segment=NSEG, D=D, m_segment=M_SEG, reps=REPS,
                    note="article S1-S4 are code D1-D4; code S0-S6 are auxiliary scenarios of the ablation"),
        population_domi_D3=population_domi("D3", (0.3, 0.5, 0.7)),
        population_domi_D2=population_domi("D2", (0.5, 0.7, 0.9)),
        sampling_sd=sampling_sd(),
        gamma_stability=gamma_stability(),
    )
    d3 = [r for r in res["population_domi_D3"] if r["level"] == 0.5][0]
    d2 = [r for r in res["population_domi_D2"] if r["level"] == 0.7][0]
    sd = res["sampling_sd"]
    res["snr"] = dict(
        note="population contrast divided by the like-for-like sd of the segment difference",
        S3_tau0_5=round(d3["abs_diff"] / sd["S3_diff_sd"], 2),
        S2_a0_7=round(d2["abs_diff"] / sd["S2_diff_sd"], 2),
    )
    import csv as _csv
    with open(os.path.join(OUT, "records_sampling_draws.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["tag", "rep", "domi"]); w.writeheader(); w.writerows(REC)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
