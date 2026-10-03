#!/usr/bin/env python
"""Figure 1 — schematic of DOMIC: representation, calibration and inference.

Three bands, read downward. Representation (Sections 4.1, 4.8): ranks feed two computational
forms of the segment DOMI, the random-feature form (solid outline; prefix sums, exact partial
traces) and the Gram form (dashed outline; exact kernel), joined by their spectral duality.
Calibration (Section 4.3): the exchangeability diagnostic chooses pair or block permutation,
which gives the permutation p-value for either form. Inference (Sections 4.2, 4.4, 4.7): the
single-break test and the two-stage segment-and-re-test procedure with its four classes.
The operator heat maps are the three operators of one simulated segment; only their shapes
are on display.

Run:  OMP_NUM_THREADS=1 python 00_SRC/make_fig1.py
Writes figure_1.{png,pdf} to figures/ (override with FIGDIR). Drawn at the width it is printed.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from dots.domi import DOMIContext  # noqa: E402

FIG = os.environ.get("FIGDIR", os.path.join(ROOT, "figures"))
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,  # TrueType glyphs, no Type 3 fonts (IEEE PDF check)
    "font.size": 7, "savefig.bbox": "tight"})

BLUE, PURPLE, TEAL, AMBER = "#2c5f8a", "#6a3d9a", "#2e7d6b", "#b9770e"
FILL = {BLUE: "#eef4fb", PURPLE: "#f3eef8", TEAL: "#eaf5f1", AMBER: "#fdf3e3"}
FS = 6.4                      # box text; printed at about 6.7 pt
W, H = 6.85, 3.72             # figure size in inches; axes coordinates are inches
N, TAU, D = 600, 300, 8


def operators(seed=11):
    """The three operators of one segment; only their shapes are on display."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, 1))
    e1, e2 = rng.standard_normal((N, 1)), rng.standard_normal((N, 1))
    Y = e1.copy()
    a = 0.9
    Y[TAU:] = np.sqrt(1 - a * a) * e1[TAU:] + a * np.abs(X[TAU:]) * e2[TAU:]
    c = DOMIContext(X, Y, 60, D=D, seed=2026, n_perm=0)
    return c.cj.rho(0, N), c.cx.rho(0, N), c.cy.rho(0, N)


def box(ax, x0, x1, y0, y1, text, ec, ls="-", fs=FS):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=FILL[ec], ec=ec, lw=1.0, ls=ls, zorder=2))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, linespacing=1.25)


def arrow(ax, p0, p1, color="#777", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=8, lw=1.0, ls=ls,
                                 color=color, shrinkA=0, shrinkB=0, zorder=1))


def heatmap(fig, M, x0, y0, s, title, below=None):
    a = fig.add_axes([x0 / W, y0 / H, s / W, s / H])
    a.imshow(M, cmap="magma"); a.set_xticks([]); a.set_yticks([])
    a.set_title(title, fontsize=FS, color=PURPLE, pad=2)
    for sp in a.spines.values():
        sp.set_color(PURPLE); sp.set_linewidth(1.0)
    if below:
        a.set_xlabel(below, fontsize=FS, color=PURPLE, labelpad=2)


def main():
    rho_xy, rho_x, rho_y = operators()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # ---------------- representation (4.1, 4.8) ----------------
    ym = 2.86                                            # center line of the input boxes
    box(ax, 0.35, 1.02, ym - 0.17, ym + 0.17, "blocks\n$X_t,\\ Y_t$", BLUE)
    box(ax, 1.17, 2.02, ym - 0.17, ym + 0.17, "ranks\n$U_X,U_Y\\in(0,1)$", BLUE)
    arrow(ax, (1.02, ym), (1.17, ym))

    # random-feature branch (solid)
    t0, t1 = 3.10, 3.44
    tm = (t0 + t1) / 2
    ax.text(2.22, 3.65, "random-feature form", fontsize=FS, color=BLUE, fontweight="bold",
            va="center")
    ax.text(3.46, 3.65, "cost per split independent of segment length; limit law (P6)",
            fontsize=FS, color=BLUE, style="italic", va="center")
    box(ax, 2.22, 3.10, t0, t1, "unit-norm RFF\n$\\varphi_X,\\varphi_Y\\in S^{D-1}$", BLUE)
    box(ax, 3.25, 4.18, t0, t1, "product states\n$\\psi_t=\\varphi_X\\otimes\\varphi_Y$", BLUE)
    box(ax, 4.33, 5.18, t0, t1, "prefix sums of\n$\\varphi\\varphi^{\\top}\\!,\\ \\psi\\psi^{\\top}$",
        BLUE)
    arrow(ax, (3.10, tm), (3.25, tm)); arrow(ax, (4.18, tm), (4.33, tm))
    arrow(ax, (2.02, ym + 0.06), (2.22, t0 + 0.05))

    # segment operators: joint and the two marginals its partial traces recover exactly
    s_big, s_small = 0.50, 0.27
    xb, yb = 5.40, tm - 0.04 - s_big / 2
    heatmap(fig, rho_xy, xb, yb, s_big, "$\\rho_{XY}$", "$D^2\\!\\times\\!D^2$")
    heatmap(fig, rho_x, 6.14, tm - s_small / 2, s_small, "$\\rho_X$")
    heatmap(fig, rho_y, 6.53, tm - s_small / 2, s_small, "$\\rho_Y$")
    ax.text(6.475, tm - s_small / 2 - 0.08, "$D\\!\\times\\!D$", fontsize=FS, color=PURPLE,
            ha="center", va="center")
    arrow(ax, (5.18, tm), (xb, tm))
    arrow(ax, (xb + s_big + 0.02, tm), (6.12, tm), color=PURPLE)
    ax.text(6.36, 2.83, "partial traces\nexact (Lemma 3)", fontsize=FS, color=PURPLE,
            ha="center", va="center", linespacing=1.15)

    # Gram branch (dashed)
    g0, g1 = 2.27, 2.63
    box(ax, 2.22, 4.30, g0, g1,
        "Gram matrices on segment rows ($m\\times m$)\n"
        "$K_X,\\ K_Y$ (exact Gaussian kernel), $K_X\\circ K_Y$", BLUE, ls="--")
    ax.text(2.22, 2.14, "Gram form", fontsize=FS, color=BLUE, fontweight="bold", va="center")
    ax.text(2.88, 2.14, "exact kernel; $\\alpha{=}1$: cost $O(m^3)$ per segment", fontsize=FS, color=BLUE,
            style="italic", va="center")
    arrow(ax, (2.02, ym - 0.06), (2.22, g1 - 0.05))

    # duality link
    arrow(ax, (3.715, t0), (3.715, g1), color=BLUE, style="<|-|>", ls=(0, (2, 1.5)))
    ax.text(3.80, (t0 + g1) / 2, "exact kernel in place of\nthe random-feature kernel (4.8);\nlimit as $D\\to\\infty$ per segment", fontsize=FS,
            color=BLUE, va="center", linespacing=1.15)

    # segment DOMI
    d0, d1 = 2.20, 2.58
    box(ax, 4.62, 6.80, d0, d1, "segment DOMI\n$I=S(\\rho_X)+S(\\rho_Y)-S(\\rho_{XY})$", PURPLE)
    arrow(ax, (4.30, (g0 + g1) / 2), (4.62, (g0 + g1) / 2), color=PURPLE)
    arrow(ax, (xb + s_big / 2, yb - 0.17), (xb + s_big / 2, d1), color=PURPLE)

    # ---------------- calibration (4.3) ----------------
    c0, c1 = 1.37, 1.75
    cm = (c0 + c1) / 2
    box(ax, 0.35, 1.72, c0, c1, "exchangeability diagnostic\nexact as a refutation test;\nconsistent (P7)", TEAL)
    box(ax, 2.30, 4.10, 1.60, 1.90, "pair permutation\nexact under pair exchangeability (P1)", TEAL)
    box(ax, 2.30, 4.10, 1.18, 1.48, "block permutation\nexact under block exchangeability (P5)",
        TEAL)
    arrow(ax, (1.72, cm + 0.06), (2.30, 1.75), color=TEAL)
    arrow(ax, (1.72, cm - 0.06), (2.30, 1.33), color=TEAL)
    ax.text(2.0, 1.83, "not refuted", fontsize=FS, color=TEAL, ha="center", va="center")
    ax.text(2.0, 1.25, "refuted", fontsize=FS, color=TEAL, ha="center", va="center")
    box(ax, 4.62, 6.80, c0, c1,
        "permutation p-value\n$p=(1+\\#\\{k:\\ T_k\\geq T_{\\mathrm{obs}}\\})\\,/\\,(K+1)$", TEAL)
    arrow(ax, (4.10, 1.75), (4.62, cm + 0.06), color=TEAL)
    arrow(ax, (4.10, 1.33), (4.62, cm - 0.06), color=TEAL)
    arrow(ax, (5.67, d0), (5.67, c1), color=PURPLE)
    ax.text(5.59, (d0 + c1) / 2, "either form: $T_{\\mathrm{obs}}$ and replicas $T_1,\\dots,T_K$",
            fontsize=FS, color=PURPLE, va="center", ha="right")

    # ---------------- inference (4.2, 4.4, 4.7) ----------------
    i0, i1 = 0.06, 0.84
    im = (i0 + i1) / 2
    bus = 1.02
    box(ax, 0.35, 1.75, i0, i1,
        "single break (4.2)\nweighted DOMI difference\n$Q(t)=\\sqrt{t(n-t)/n}\\,"
        "|\\hat I_{[0,t)}-\\hat I_{[t,n)}|$\nAlgorithm 1", AMBER)
    box(ax, 1.93, 3.30, i0, i1,
        "multiple breaks, stage 1:\nHolevo partitioning (4.4)\nrandom-feature form, cost\n"
        "$E_sS(\\rho_{XY,s})$, Holevo split gain", AMBER)
    box(ax, 3.48, 4.72, i0, i1,
        "stage 2: re-test (4.7)\neach candidate by the\nDOMI difference plus a\n"
        "marginal-scale test", AMBER)
    arrow(ax, (3.30, im), (3.48, im), color=AMBER)
    arrow(ax, (4.72, im), (4.90, im), color=AMBER)

    # permutation calibration feeds every test and the Holevo partitioning cost correction and penalty
    ax.plot([5.00, 5.00, 1.05], [c0, bus, bus], color=TEAL, lw=1.0, zorder=1,
            solid_joinstyle="miter")
    for x in (1.05, 2.615, 4.10):
        arrow(ax, (x, bus + 0.005), (x, i1), color=TEAL)

    # four classes of a candidate (Algorithm 2)
    tx0, rh, cw = 4.90, 0.50, 0.70
    ty1 = i1                                              # top of the header row
    ax.text((tx0 + 6.80) / 2, 0.93, "four classes (Algorithm 2)", fontsize=FS, color=AMBER,
            fontweight="bold", ha="center", va="center")
    hdr = 0.24
    xc = [tx0 + rh, tx0 + rh + cw, tx0 + rh + 2 * cw]      # column edges of the two class cells
    yr = [ty1 - hdr, ty1 - hdr - (ty1 - hdr - i0) / 2, i0]  # row edges
    ax.text((xc[0] + xc[1]) / 2, ty1 - hdr / 2, "no scale\nchange found", fontsize=FS, ha="center",
            va="center", linespacing=1.1)
    ax.text((xc[1] + xc[2]) / 2, ty1 - hdr / 2, "scale\nchange", fontsize=FS, ha="center",
            va="center", linespacing=1.1)
    ax.text(tx0 + rh / 2, (yr[0] + yr[1]) / 2, "DOMI\nrejects", fontsize=FS, ha="center",
            va="center", linespacing=1.1)
    ax.text(tx0 + rh / 2, (yr[1] + yr[2]) / 2, "DOMI\ndoes not", fontsize=FS, ha="center",
            va="center", linespacing=1.1)
    cells = [["dependence", "both"], ["undetermined", "scale"]]
    for r in range(2):
        for k in range(2):
            ax.add_patch(Rectangle((xc[k], yr[r + 1]), cw, yr[r] - yr[r + 1], fc=FILL[AMBER],
                                   ec=AMBER, lw=1.0, zorder=2))
            ax.text((xc[k] + xc[k + 1]) / 2, (yr[r] + yr[r + 1]) / 2, cells[r][k], fontsize=FS,
                    ha="center", va="center", zorder=3, linespacing=1.1)

    for y, txt, c in [(2.92, "representation (4.1, 4.8)", BLUE),
                      (1.55, "calibration (4.3)", TEAL),
                      (0.45, "inference\n(4.2, 4.4, 4.7)", AMBER)]:
        ax.text(0.12, y, txt, rotation=90, va="center", ha="center",
                fontsize=6.6, color=c, fontweight="bold", linespacing=1.15)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"figure_1.{ext}"), dpi=300, bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print("figure_1 saved")


if __name__ == "__main__":
    main()
