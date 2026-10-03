#!/usr/bin/env python
"""Figure 3 of the article: the hourly two-stage weather analysis.

Left: the strongest candidate (Suwon humidity-wind, 2019-08-26) as a rolling DOMI, with the
marginal standard deviations across it.
Right: the classes assigned by the two stages to all 27 candidates under the season-restricted
stage two (run_e8_season.py).

Run:  python 00_SRC/make_figure3_weather.py
Reads 02_MART/WEATHER_HOURLY_ANOM.npz and 04_DAOU/EXPERIMENT/e8_season/; writes
figure_3.{png,pdf} to figures/ (override with FIGDIR).
FIGURE3_LAYOUT=column draws the left panel alone at single-column width (the layout printed as
Figure 3); the class counts of the right panel are then given in the text of Section 6.7.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,  # TrueType glyphs, no Type 3 fonts (IEEE PDF check)
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.frameon": True, "legend.framealpha": 0.9,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
})   # as in make_paper_figures.py, without its figure.dpi and lines.markersize

SRC = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.hourly import features, block_moments, prefix, seg_domi  # noqa: E402

E8S = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "e8_season")
ANOM = os.path.join(ROOT, "02_MART", "WEATHER_HOURLY_ANOM.npz")
OUTP = os.environ.get("FIGDIR", os.path.join(ROOT, "figures"))
os.makedirs(OUTP, exist_ok=True)
UNIT = 336

if not os.path.exists(ANOM):
    raise SystemExit(f"missing {ANOM}: build it first -- see README, Data section")
z = np.load(ANOM)
a, b = z["119|hm"], z["119|ws"]
tm = z["119|tm"]
ok = np.isfinite(a) & np.isfinite(b)
a, b, tm = a[ok], b[ok], tm[ok]
edges = np.arange(0, len(a) + 1, UNIT)
if edges[-1] != len(a):
    edges = np.append(edges, len(a))
B = len(edges) - 1
FX, FY, J = features(a, b)
P = prefix(*block_moments(FX, FY, J, edges))

WIN = 26                                   # +/- one year of two-week intervals
mid, domi = [], []
for c in range(WIN, B - WIN):
    domi.append(seg_domi(P, c - WIN, c + WIN))
    mid.append(int(tm[min(edges[c], len(tm) - 1)][:4]) + (int(tm[min(edges[c], len(tm) - 1)][4:6]) - 1) / 12)
mid, domi = np.array(mid), np.array(domi)

sd_h, sd_w, mid2 = [], [], []
for c in range(WIN, B - WIN):
    sl = slice(edges[c - WIN], edges[c + WIN])
    sd_h.append(np.nanstd(a[sl])); sd_w.append(np.nanstd(b[sl]))
    mid2.append(mid[c - WIN])
sd_h, sd_w = np.array(sd_h), np.array(sd_w)

brk = 2019 + (8 - 1) / 12 + 26 / 365
COLUMN = os.environ.get("FIGURE3_LAYOUT", "") == "column"
if COLUMN:
    fig, ax0 = plt.subplots(1, 1, figsize=(3.45, 2.75), layout="constrained")
    ax = [ax0]
else:
    fig, ax = plt.subplots(1, 2, figsize=(6.85, 2.6), layout="constrained",
                           gridspec_kw={"width_ratios": [1.45, 1]})
ax[0].plot(mid, domi, color="#1b4b8f", lw=1.6, label="rolling DOMI (±1 yr)")
ax[0].axvline(brk, color="#c0392b", ls="--", lw=1.5, label="candidate break 2019-08-26")
ax[0].set_xlabel("year"); ax[0].set_ylabel("segment DOMI")
ax[0].set_ylim(top=float(np.nanmax(domi)) * 1.08)
ax[0].grid(alpha=.25)
axb = ax[0].twinx()
axb.plot(mid2, sd_h / np.nanmean(sd_h), color="#7f8c8d", lw=1.0, ls=":", label="humidity SD (relative)")
axb.plot(mid2, sd_w / np.nanmean(sd_w), color="#34495e", lw=1.0, ls="-.", label="wind SD (relative)")
axb.set_ylabel("rolling SD / its mean"); axb.set_ylim(0.7, 1.45)
# one legend for both axes, below the plot so that it covers neither curves nor axis labels
h0, l0 = ax[0].get_legend_handles_labels()
h1, l1 = axb.get_legend_handles_labels()
fig.legend(h0 + h1, l0 + l1, loc="outside lower center", ncol=2, frameon=False, fontsize=6.5,
           handlelength=1.8, columnspacing=1.0)

if COLUMN:
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(OUTP, f"figure_3.{ext}"), dpi=300)
    print("figure_3 saved (column layout)")
    sys.exit(0)

d = json.load(open(os.path.join(E8S, "results.json")))
cnt = {}
for c in d["candidates"]:
    cnt[c["cls"]] = cnt.get(c["cls"], 0) + 1
order = ["dependence change", "both change", "marginal-driven", "undetermined"]
vals = [cnt.get(k, 0) for k in order]
cols = ["#1b4b8f", "#5b8dd9", "#c0392b", "#bdc3c7"]
ax[1].barh(range(4), vals, color=cols)
ax[1].set_yticks(range(4)); ax[1].set_yticklabels(
    ["dependence only", "dependence and scale", "scale only", "undetermined"])   # the caption defines them
ax[1].invert_yaxis()
for i, v in enumerate(vals):
    ax[1].text(v + 0.3, i, str(v), va="center")
ax[1].set_xlabel("candidate breaks (of 27)")
ax[1].grid(alpha=.25, axis="x"); ax[1].set_xlim(0, max(vals) + 3.5)
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(OUTP, f"figure_3.{ext}"), dpi=300)
print("figure_3 saved")
