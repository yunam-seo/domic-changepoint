#!/usr/bin/env python
"""Figures of the article and its Supplementary Material, drawn from 04_DAOU/EXPERIMENT outputs.

    python 00_SRC/make_paper_figures.py          (writes $FIGDIR, default figures/)

File -> figure: figure_2 = Figure 2; figure_B3 = Figure B.3; figure_B4 = Figure B.4;
figure_B5 = Figure B.5; figure_B1 = Figure B.1; figure_B2 = Figure B.2.
Displayed labels call the statistic DOMI; the data keys are the code's
"DOMI-..." names, and "matmi-a1" is the Gram form of DOMI."""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.synth import SCENARIOS, generate  # noqa: E402
from dots.domi import DOMIContext, ranks01, unit_rff  # noqa: E402
from dots.encode import MomentCache  # noqa: E402
from dots import pelt as P  # noqa: E402

BASE = os.path.join(ROOT, "04_DAOU", "EXPERIMENT")
FIG = os.environ.get("FIGDIR", os.path.join(ROOT, "figures"))
os.makedirs(FIG, exist_ok=True)
# Drawn at print size (full text width 6.85 in, one column 3.35 in); all type is 7 pt or larger.
FULL, HALF = 6.85, 3.35          # inches: full text width and one column
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,  # TrueType glyphs, no Type 3 fonts (IEEE PDF check)
    
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.frameon": True, "legend.framealpha": 0.9, "figure.dpi": 110,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
    "lines.markersize": 2.5,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(name, "saved", flush=True)


def load_csv(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in list(r):
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError):
                pass
    return rows


# ---------------------------------------------------------------- Figure 2: single-break power
COL = {"DOMI-diff|gs": ("#c0392b", "DOMI, random features"), "matmi-a1|gs": ("#e67e22", "DOMI, Gram form"),
       "Holevo-joint|g": ("#7d3c98", "joint-state Holevo"),
       "HSIC-diff|gs": ("#117a65", "HSIC"), "dCor-diff|gs": ("#1f618d", "dCor"),
       "Spearman-diff|g": ("#5dade2", "Spearman"), "CvM-sub|g": ("#b7950b", "CvM (copula test)"),
       "MMDx1.0|g": ("#45b39d", "MMD"), "GaussLR|g": ("#154360", "Gaussian LR")}


def figure_2():
    # Same selection rule as the printed Table 1 (see rebuild_table1.py): D1/D2 have a
    # level-independent null, so the level-0 null run applies; every D3/D4 level comes from
    # the matched-null run, whose threshold is drawn at the level's own null.
    rows = [r for r in load_csv(os.path.join(BASE, "e1", "results.csv"))
            if r["scen"] not in ("D3", "D4")]
    rows += [r for r in load_csv(os.path.join(BASE, "e1_matched_null", "results.csv"))
             if r["scen"] in ("D3", "D4")]
    # Gram form of DOMI: full-segment runs of run_matmi_baseline.py (run_gram_full.sh)
    for tag in ("full_T1a", "full_T1b", "full_T1c", "full_T1d", "full_T1e"):
        rows += [r for r in load_csv(os.path.join(BASE, "matmi_baseline_" + tag, "results.csv"))
                 if r["key"] == "matmi-a1|gs"]
    # copula test of [7] with ranks recomputed within each segment (run_cvm_subsample.py)
    rows += [r for r in load_csv(os.path.join(BASE, "cvm_subsample", "results.csv"))
             if r["key"] == "CvM-sub|g" and r["scen"] in ("D1", "D2", "D3", "D4")]
    scens = ["D1", "D2", "D3", "D4"]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.30), sharey=True)
    fig.subplots_adjust(bottom=0.30)
    for ax, s in zip(axes, scens):
        for key, (c, lab) in COL.items():
            pts = sorted([(r["level"], r["power"]) for r in rows if r["scen"] == s and r["key"] == key])
            if pts:
                joint = key in ("Holevo-joint|g", "MMDx1.0|g", "GaussLR|g")
                ax.plot(*zip(*pts), marker="s" if joint else "o", ms=3, color=c, ls="--" if joint else "-",
                        lw=2.0 if key in ("DOMI-diff|gs", "matmi-a1|gs", "Holevo-joint|g") else 1.0, label=lab)
        ax.set_title({"D1": "S1 correlation", "D2": "S2 uncorrelated", "D3": "S3 copula shape",
                      "D4": "S4 sign-mixed"}[s], fontsize=8)
        sym = {"r": "$r$", "a": "$a$", "tau": r"Kendall's $\tau$", "s": "$s$"}
        ax.set_xlabel(sym.get(SCENARIOS[s]["param"], SCENARIOS[s]["param"]))
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("localized power")
    # the legend goes below the panels: inside S4 it would cover the two curves that rise there
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.13),
               frameon=False, handlelength=1.4, columnspacing=1.2, handletextpad=0.4)
    save(fig, "figure_2")


# ---------------------------------------------------------------- Figure B.3: kernel (Section 3.4)
def figure_B3():
    """The three functionals' weighting of an eigenvalue pair, and the entropic amplification.

    Section 3.4 argues in algebra that the entropy functional weights low-eigenvalue directions
    more heavily than the Bures/quantum-Fisher form and than the flat Frobenius one, and that the
    advantage grows only logarithmically. The two panels are that argument drawn: the kernels
    themselves, and the ratio between the first two, whose values at eigenvalue ratios of 1e6 and 1e10
    (6.9 and 11.5, labeled to the nearest integer) are the numbers quoted in the text.
    """
    r = np.logspace(1e-6, 12, 400)                    # eigenvalue ratio a/b, with a = 1; the
    b = 1.0 / r                                       # ratio starts just above 1, where both
    inv_log = np.log(r) / (1.0 - b)                   # kernels take the common value 1/a
    inv_ari = 2.0 / (1.0 + b)                         # 1 / arithmetic mean  (Bures / quantum Fisher)
    flat = np.ones_like(r)                            # Frobenius
    fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.4), layout="constrained")

    # direct labels instead of a legend box: the three curves are well separated on the
    # right (about 28, 2 and 1), and a box anywhere collides with the rising entropy curve
    ax[0].loglog(r, inv_log, color="#c0392b", lw=2.0)
    ax[0].loglog(r, inv_ari, color="#1b4b8f", lw=1.6, ls="--")
    ax[0].loglog(r, flat, color="#7f8c8d", lw=1.4, ls=":")
    ax[0].text(4, 26, "entropy (reciprocal log mean)", color="#c0392b", fontsize=7)
    ax[0].text(3e11, 2.25, "Bures / quantum Fisher\n(reciprocal arithmetic mean)",
               color="#1b4b8f", fontsize=7, ha="right", va="bottom")
    ax[0].text(3e11, 1.07, "Frobenius (constant)", color="#7f8c8d", fontsize=7,
               ha="right", va="bottom")
    ax[0].set_xlabel(r"eigenvalue ratio $\lambda_i/\lambda_j$")
    ax[0].set_ylabel("weight on direction $(i,j)$")
    ax[0].grid(alpha=.3, which="both")

    amp = inv_log / inv_ari
    ax[1].semilogx(r, amp, color="#c0392b", lw=2.0)
    for rr in (1e6, 1e10):
        a_ = float(np.log(rr) * (rr + 1) / (2 * (rr - 1)))     # exact ratio at the marked point
        txt = f"{a_:.0f}×"
        ax[1].plot([rr], [a_], "o", ms=4, color="#c0392b")
        ax[1].annotate(txt, (rr, a_), textcoords="offset points", xytext=(5, -8), fontsize=7)
    ax[1].set_xlabel(r"eigenvalue ratio $\lambda_i/\lambda_j$")
    ax[1].set_ylabel("entropy weight /\nBures weight")
    ax[1].grid(alpha=.3, which="both")
    save(fig, "figure_B3")


# ---------------------------------------------------------------- Figure B.4: multiple breaks
def figure_B4():
    e2 = load_csv(os.path.join(BASE, "e2", "results.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4), gridspec_kw={"width_ratios": [1, 1.6]},
                             layout="constrained")
    ms = ["Holevo-partition", "PELT-Gauss", "BinSeg-MMD"]       # data keys in e2/results.csv
    x = np.arange(len(ms))
    for off, lv, alpha in [(-0.2, 0.8, 0.45), (0.2, 0.9, 1.0)]:
        vals = [next(r["k_correct"] for r in e2 if r["method"] == m and r["level"] == lv) for m in ms]
        axes[0].bar(x + off, vals, 0.38, color=["#c0392b", "#154360", "#45b39d"], alpha=alpha, label=f"a={lv}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Holevo\npartitioning", "rank-Gauss\nPELT", "BinSeg\nMMD"])
    axes[0].set_ylabel(r"$P(\hat{K}=3)$")
    from matplotlib.patches import Patch   # level is encoded by lightness across the three method colors
    axes[0].legend(handles=[Patch(facecolor="#7f7f7f", alpha=0.45, label="a=0.8"), Patch(facecolor="#7f7f7f", label="a=0.9")])
    axes[0].grid(alpha=0.3, axis="y")
    # example segmentation, rep 0, a=0.9
    smp = generate("P1", 0.9, 0, null=False, base_seed=20260825, level_idx=1)
    Z = smp["Z"]
    n = Z.shape[0]
    FX = unit_rff(ranks01(Z[:, [0]]), 8, 2026)
    FY = unit_rff(ranks01(Z[:, [1]]), 8, 2027)
    J = np.einsum("ti,tj->tij", FX, FY).reshape(n, -1)
    grid = np.arange(0, n + 1, 10)
    # the reported procedure on this replicate: e2's bias-correction stream and calibrated penalty
    rngl = np.random.default_rng([20260825, 0, 0, 1, 13])
    beta = next(r["beta"] for r in e2 if r["method"] == "Holevo-partition" and r["level"] == 0.9)
    C = P.cost_matrix_vn(MomentCache(J), grid)
    Cp = np.zeros_like(C)
    for _ in range(3):
        Cp += P.cost_matrix_vn(MomentCache(J[rngl.permutation(n)]), grid)
    cps = [int(grid[i]) for i in P.pelt_from_costs(C - Cp / 3, beta)]
    print("Figure B.4 example (rep 0, a=0.9, beta=%.3f): breaks %s" % (beta, cps))
    ctx = DOMIContext(Z[:, [0]], Z[:, [1]], 90, D=8, seed=2026, n_perm=1)
    qw = [ctx.domi_bc(t - 90, t) for t in range(90, n)]
    axes[1].plot(range(90, n), qw, color="#7d3c98", lw=1.1)
    for t0 in (450, 900, 1350):
        axes[1].axvline(t0, color="#999", lw=4, alpha=0.4)
    for c in cps:
        axes[1].axvline(c, color="k", ls="--", lw=1)
    axes[1].set_ylabel("bias-corrected\ntrailing-window DOMI")
    axes[1].set_xlabel("t")
    axes[1].grid(alpha=0.3)
    save(fig, "figure_B4")


# ---------------------------------------------------------------- Figure B.1: ablation
def figure_B1():
    rows = load_csv(os.path.join(ROOT, "04_DAOU", "ABLATION", "results_long.csv"))
    # all five ablation families, including the heavy-tail case where the ordering reverses
    # short panel tags; the caption names the five families in full
    scens = [("N3", "dependence"), ("S3", "rotation"),
             ("N1", "mixture"), ("N2", "heavy tails"), ("S1", "mean shift")]
    fig, axes = plt.subplots(1, 5, figsize=(FULL, 2.5), sharey=True)
    series = [("kQDg-Holevo|g", "#c0392b", "entropy (Holevo) of 2nd moment"),
              ("kFrob|g", "#1f618d", "Frobenius of same moment"),
              ("kSpecHolevo|g", "#e67e22", "spectral part only"),
              ("MMD-best", "#117a65", "MMD (oracle bandwidth)")]
    for k, (ax, (s, title)) in enumerate(zip(axes, scens)):
        rs = [r for r in rows if r["scen"] == s]
        levels = sorted({r["level"] for r in rs})
        for key, c, lab in series:
            if key == "MMD-best":
                vals = [max(r["power"] for r in rs if r["level"] == lv and str(r["method"]).startswith("MMDx")) for lv in levels]
            else:
                vals = [next(r["power"] for r in rs if r["level"] == lv and r["key"] == key) for lv in levels]
            ax.plot(levels, vals, "o-", ms=3, color=c, lw=2.0 if "entropy" in lab else 1.0, label=lab)
        ax.set_title(title)
        # each family has its own parameter (dots/synth.py): N3 a, S3 rotation angle, N1 a, N2 degrees of freedom, S1 shift
        ax.set_xlabel({"N3": "$a$", "S3": "angle (degrees)", "N1": "$a$", "N2": r"$\nu$", "S1": r"$\delta$"}[s])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("power")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.16),
               frameon=False, handlelength=1.4, columnspacing=1.6, handletextpad=0.4)
    fig.subplots_adjust(bottom=0.34)
    save(fig, "figure_B1")


def _window_domi(x, y, D=8):
    """DOMI of a single window, computed directly from its own ranks."""
    from dots.hourly import features
    from dots.hourly import _ent
    FX, FY, J = features(x, y, D=D)
    m = len(FX)
    return (_ent(FX.T @ FX / m) + _ent(FY.T @ FY / m) - _ent(J.T @ J / m))


def figure_B5():
    """Finance: rolling coupling of S&P 500 returns and 10-year yield changes, with the weekly
    Holevo partitioning breaks. Needs the daily closes (see README, Data)."""
    from run_e6_finance import load, align_pair
    from scipy.stats import spearmanr

    spx, tnx = load("yahoo_gspc.csv"), load("yahoo_tnx.csv")
    dates, x, y = align_pair(spx, tnx, ("logret", "diff"))
    yr = np.array([int(d[:4]) + (int(d[4:6]) - 1) / 12 for d in dates])
    W = 250
    mid, rho, domi = [], [], []
    for t in range(W, len(x) - W, 10):
        sl = slice(t - W, t + W)
        mid.append(yr[t])
        rho.append(spearmanr(x[sl], y[sl]).statistic)
        domi.append(_window_domi(x[sl], y[sl]))
    res = json.load(open(os.path.join(BASE, "e6", "results_block.json")))
    bx = [int(str(d)[:4]) + (int(str(d)[4:6]) - 1) / 12 for d in res["e6a_block_SPX-TNX"]["cps_dates"]]

    fig, axes = plt.subplots(2, 1, figsize=(FULL, 3.4), sharex=True, layout="constrained")
    axes[0].plot(mid, rho, color="#7f8c8d", lw=1.8)
    axes[0].axhline(0, color="k", lw=0.8, ls=":")
    axes[0].set_ylabel("rolling Spearman")
    axes[1].plot(mid, domi, color="#1b4b8f", lw=1.8)
    axes[1].set_ylabel("rolling DOMI"); axes[1].set_xlabel("year")
    # no legend: a single-entry box at the top reads as a panel title and covers the 2012-2015
    # stretch of the Spearman curve. The caption identifies the dashed line instead.
    for a_ in axes:
        for b_ in bx:
            a_.axvline(b_, color="#c0392b", ls="--", lw=1.4)
    save(fig, "figure_B5")


def figure_B2():
    """Finite-sample illustrations (Figure B.2): scaling of the plug-in under independence (rate m,
    P6), empirical rejection frequencies of the permutation test (P1), and the localization error of
    a permutation-studentized estimator over n (not a check of P3)."""
    tx = json.load(open(os.path.join(BASE, "theory_extras", "results.json")))
    th = json.load(open(os.path.join(BASE, "theory", "results.json")))
    rate = tx["X1_independence_degeneracy"]["rate"]
    ms = sorted(int(k) for k in rate)
    msd = [rate[str(m)]["m_times_sd"] for m in ms]
    ssd = [rate[str(m)]["sqrtm_times_sd"] for m in ms]

    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.5), layout="constrained")
    axes[0].plot(ms, msd, "o-", color="#1b4b8f", lw=2, ms=7, label=r"$m\cdot\mathrm{sd}(\hat I)$")
    axes[0].plot(ms, ssd, "s--", color="#c0392b", lw=2, ms=7, label=r"$\sqrt{m}\cdot\mathrm{sd}(\hat I)$")
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("segment length $m$")
    axes[0].set_xticks(ms); axes[0].set_xticklabels([str(m) for m in ms])
    axes[0].minorticks_off()
    axes[0].legend(loc="center left")

    lc = th["T2"]["level_check"]
    al = sorted(float(k) for k in lc)
    axes[1].plot([0, 0.22], [0, 0.22], "k:", lw=1.2, label="nominal")
    axes[1].plot(al, [lc[str(a_)] for a_ in al], "o-", color="#1b4b8f", lw=2, ms=4, label="empirical")
    axes[1].set_xlabel(r"$\alpha$"); axes[1].set_ylabel(r"$P(p\leq\alpha)$")
    axes[1].legend(loc="upper left")

    t3 = th["T3"]; ns = sorted(int(k) for k in t3)
    vals = [100 * t3[str(n)]["median_rel_err"] for n in ns]
    axes[2].plot(range(len(ns)), vals, "o-", color="#1b4b8f", lw=2, ms=4)
    axes[2].set_xticks(range(len(ns))); axes[2].set_xticklabels([str(n) for n in ns])
    axes[2].set_ylim(0, max(vals) * 1.45)
    axes[2].set_xlabel("$n$"); axes[2].set_ylabel(r"median $|\hat\tau-\tau|/n$ (%)")
    for i, v in enumerate(vals):
        axes[2].annotate(f"{v:.1f}", (i, v), textcoords="offset points", xytext=(0, 9),
                         ha="center", fontsize=7)
    save(fig, "figure_B2")


def main():
    figure_B3()
    figure_2()
    figure_B4()
    figure_B1()
    figure_B2()
    try:                    # needs the daily closes, which are third-party and not redistributed
        figure_B5()
    except FileNotFoundError as e:
        print("skipping the finance figure (Supplementary Figure B.5):", e)
    # The weather figure (Figure 3) comes from make_figure3_weather.py (hourly anomaly mart).
    print("all figures in", FIG)


if __name__ == "__main__":
    main()
