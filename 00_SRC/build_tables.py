#!/usr/bin/env python
"""Build the result tables of the article from the stored results, with a provenance record per cell.

Tables built (LaTeX fragments):
  t1_combined.tex   Table 1, single-break comparison: upper block S1-S4 and M1, lower block G1-G6
                    (the lower block is also written alone as t6_nongauss.tex)
  tb_jointref.tex   Supplementary Table B.6, joint-distribution references
  t5_timing.tex     Table 2, cost of the two computational forms (scaling/)
  tb10_gram.tex     Supplementary Table B.3, Gram form at alpha = 1 and 2 and the random-feature
                    form at D = 8, 16
  tb11_nongauss.tex Supplementary Table B.4, the full non-Gaussian comparison (all methods, raw and
                    studentized)
  tb_mv.tex B.1; tb_bandwidth.tex B.2; tb_blockperm.tex B.5; tb7_seg.tex B.7; tb_e3perm.tex B.8;
  tb_bmctc.tex B.9; tb_detect.tex B.10; tb_permcmp.tex B.11; tb_offcentre.tex B.12; tb_asym.tex B.13;
  tb_databased.tex B.14; tb_mctire.tex B.15; tb_cost.tex B.16; tb_signflip.tex B.17;
  tb_bmctc_factorial.tex B.18; tb_finance_seeds.tex B.19; tb_us_panel.tex B.20; tb_us_yearly.tex B.21
                    (Supplementary Tables)
and provenance.csv: one line per table number -> (file, scen, level, key, value). The file names
t5_*, t6_*, tb10_*, tb11_* and the provenance labels "T5" (Table 2) and "T6" (lower block of
Table 1) keep an earlier table numbering.

Sources: matmi_baseline/ (S1, S2, M1), matmi_baseline_matched_D8/ (S3, S4, matched null),
matmi_baseline_D16/ and matmi_baseline_matched_D16/ (D = 16), ng_suite_A/ and ng_suite_B/ (G1-G6),
e1_matched_null/table1.csv (Table 1 entries of the other methods). The Gram-form entries
are overlaid from the full-segment runs matmi_baseline_full_T1a, matmi_baseline_full_T1b and
ng_suite_gfull_G1 .. ng_suite_gfull_G6 (run_gram_full.sh).

Selection rule (Section 5): each method is reported at whichever of its two forms, the raw weighted
global difference "g" or the studentized one "gs", is the more powerful FOR THAT METHOD, one form
per method held across all columns of a table. "More powerful" is decided by the mean power over
the table's power columns.

The script stops if a table cell is missing from its source file or a run does not have
n_reps = 500.

Run:  python 00_SRC/build_tables.py
Output: 04_DAOU/EXPERIMENT/tables/
"""
from __future__ import annotations

import csv
import json
import re
import os
import sys
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "04_DAOU", "EXPERIMENT")
OUT = os.path.join(EXP, "tables")
PROV = []


def load(sub):
    """(scen, level, method, mode) -> row dict; the method key of the DOMI statistic is stored as DOMI."""
    d = {}
    path = os.path.join(EXP, sub, "results.csv")
    for r in csv.DictReader(open(path)):
        m = "DOMI" if r["method"] == "DOMI-diff" else r["method"]
        d[(r["scen"], round(float(r["level"]), 4), m, r["mode"])] = dict(r, _file=os.path.relpath(path, ROOT))
    return d


def overlay_gram(old, full_subs):
    """Replace the Gram-form entries (matmi-*) of `old` by those of the full-segment runs."""
    d = dict(old)
    for sub in full_subs:
        new = load(sub)
        for k, r in new.items():
            if k[2].startswith("matmi"):
                d[k] = r
    return d


def get(d, scen, level, method, mode, rate="power"):
    k = (scen, round(level, 4), method, mode)
    if k not in d:
        sys.exit(f"MISSING cell {k}")
    r = d[k]
    if int(r["n_reps"]) != 500:
        sys.exit(f"n_reps != 500 at {k}")
    v = float(r[rate])
    return v, r


def cell(d, scen, level, method, mode, rate="power", table=""):
    v, r = get(d, scen, level, method, mode, rate)
    PROV.append(dict(table=table, file=r["_file"], scen=scen, level=level, method=method, mode=mode,
                     rate=rate, value=v, printed=f"{v:.2f}"))
    return v


def pick_mode(d, method, cols):
    """One of g / gs for this method: the larger mean power over the table's power columns."""
    best = None
    for mode in ("g", "gs"):
        vals = [get(d, s, l, method, mode)[0] for s, l, rate in cols if rate == "power"]
        m = sum(vals) / len(vals)
        if best is None or m > best[1]:
            best = (mode, m)
    return best[0]


def fmt(v):
    return f"{v:.2f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    base = load("matmi_baseline")
    matched = load("matmi_baseline_matched_D8")
    # Gram form: full segments (no subsample), see run_gram_subsample_check.py
    base = overlay_gram(base, ["matmi_baseline_full_T1a"])
    matched = overlay_gram(matched, ["matmi_baseline_full_T1b"])

    # ---------------------------------------------------------------- Table 1 lower block / B.4: non-Gaussian
    ng = load("ng_suite_A")
    ng.update(load("ng_suite_B"))
    ng = overlay_gram(ng, [f"ng_suite_gfull_G{i}" for i in range(1, 7)])
    # copula change-point test with subsample ranks (Buecher et al. 2014), run_cvm_subsample.py
    sub = load("cvm_subsample")
    ng.update({k: v for k, v in sub.items() if k[0].startswith("G") and k[2] == "CvM-sub"})
    LV = {"G1": [8.0, 4.0, 2.0], "G2": [0.2, 0.35, 0.5], "G3": [0.2, 0.35, 0.5],
          "G4": [0.5, 0.7, 0.9], "G5": [0.5, 0.7, 0.9], "G6": [0.3, 0.5, 0.8]}
    PAR = {"G1": r"\nu", "G2": r"\tau", "G3": r"\tau", "G4": "a", "G5": "a", "G6": "b"}
    NAME = {"G1": "t copula", "G2": "Gumbel", "G3": "Clayton", "G4": "U-shape", "G5": "periodic",
            "G6": "volatility"}
    gcols = [(s, l, "power") for s in LV for l in LV[s]]
    M6 = [("DOMI", "DOMI"), ("matmi-a1", "Gram"), ("HSIC-diff", "HSIC"), ("dCor-diff", "dCor"),
          ("Spearman-diff", "Spear."), ("CvM-sub", "CvM")]
    modes = {m: pick_mode(ng, m, gcols) for m, _ in M6 + [("matmi-r2", ""), ("CopulaCvM", "")]}
    modes["DOMI"] = modes["matmi-a1"] = "gs"     # fixed in advance by Algorithm 1 (Section 5), not chosen from the data
    print("G-suite modes:", modes)
    lines = []
    wins = defaultdict(int)
    detectable = 0
    summary = []
    for s in LV:
        for l in LV[s]:
            vals = [cell(ng, s, l, m, modes[m], "power", "T6") for m, _ in M6]
            top = max(vals)
            lv = f"{l:g}"
            cells = [(r"\textbf{" + fmt(v) + "}") if fmt(v) == fmt(top) else fmt(v) for v in vals]
            lines.append(f"{s}, ${PAR[s]}{{=}}{lv}$ & " + " & ".join(cells) + r" \\")
            if top >= 0.10:
                detectable += 1
                for (m, lab), v in zip(M6, vals):
                    if fmt(v) == fmt(top):
                        wins[lab] += 1
            summary.append(dict(scen=s, level=l, **{lab: v for (m, lab), v in zip(M6, vals)}))
    open(os.path.join(OUT, "t6_nongauss.tex"), "w").write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "t6_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"Table 1 lower block: {detectable} cells with top power >= 0.10; top (ties counted for each):", dict(wins))

    # ---------------------------------------------------------------- Table B.4: full G comparison
    MB = [("DOMI", "DOMI"), ("matmi-a1", r"Gram $\alpha{=}1$"), ("matmi-r2", r"Gram $\alpha{=}2$"),
          ("HSIC-diff", "HSIC"), ("dCor-diff", "dCor"), ("Spearman-diff", "Spearman"),
          ("CvM-sub", "CvM"), ("CopulaCvM", "CvM, global")]
    lines = []
    for s in LV:
        for l in LV[s]:
            vals = []
            for m, _ in MB:
                g = cell(ng, s, l, m, "g", "power", "B4")
                gs = cell(ng, s, l, m, "gs", "power", "B4")
                vals.append(f"{fmt(g)}/{fmt(gs)}")
            lines.append(f"{s}, ${PAR[s]}={l:g}$ & " + " & ".join(vals) + r" \\")
    open(os.path.join(OUT, "tb11_nongauss.tex"), "w").write("\n".join(lines) + "\n")

    # ---------------------------------------------------------------- Table B.3: forms and D
    d16 = load("matmi_baseline_D16")
    if not os.path.exists(os.path.join(EXP, "matmi_baseline_matched_D16", "results.csv")):
        sys.exit("missing matmi_baseline_matched_D16/results.csv: run\n  python 00_SRC/run_matmi_baseline.py "
                 "--scenarios D3,D4 --D 16 --no-matmi --matched --tag matched_D16")
    md16 = load("matmi_baseline_matched_D16")
    BCOLS = [("D1", 0.35, "power", base, None), ("D2", 0.7, "power", base, d16), ("D2", 0.9, "power", base, d16),
             ("D3", 0.5, "power", matched, md16), ("D4", 0.7, "power", matched, md16),
             ("D4", 0.85, "power", matched, md16), ("M1", 2.0, "detect_rate", base, None)]
    rows = []
    rows.append(("DOMI, random features, $D{=}8$",
                 [fmt(cell(src, s, l, "DOMI", "gs", rate, "B3")) for s, l, rate, src, _ in BCOLS]))
    r16 = []
    for s, l, rate, src, s16 in BCOLS:
        if s16 is None:
            r16.append("--")
        else:
            r16.append(fmt(cell(s16, s, l, "DOMI", "gs", rate, "B3")))
    rows.append(("DOMI, random features, $D{=}16$", r16))
    for m, lab in [("matmi-a1", r"DOMI, Gram form ($\alpha{=}1$)"), ("matmi-r2", r"Gram form, $\alpha{=}2$")]:
        rows.append((lab, [fmt(cell(src, s, l, m, "gs", rate, "B3")) for s, l, rate, src, _ in BCOLS]))
    open(os.path.join(OUT, "tb10_gram.tex"), "w").write(
        "\n".join(lab + " & " + " & ".join(v) + r" \\" for lab, v in rows) + "\n")
    for lab, v in rows:
        print("B.3", lab, v)

    build_t5()
    build_t1_combined()
    build_seg()
    build_bmctc()
    build_detect()
    build_permcmp()
    build_offcentre()
    build_databased()
    build_mctire()
    build_cost()
    build_signflip()
    build_bmctc_factorial()
    build_finance_seeds()
    build_us_panel()
    build_us_yearly()
    build_asym()
    build_blockperm()
    build_mv()
    build_bandwidth()
    build_e3perm()
    with open(os.path.join(OUT, "provenance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(PROV[0]))
        w.writeheader()
        w.writerows(PROV)
    print(f"provenance: {len(PROV)} cells -> {os.path.relpath(OUT, ROOT)}/provenance.csv")


def build_t5():
    # ---------------------------------------------------------------- Table 2: timing
    tp = os.path.join(EXP, "scaling", "scaling.csv")
    if os.path.exists(tp):
        T = defaultdict(dict)
        for r in csv.DictReader(open(tp)):
            T[int(r["n"])][r["form"]] = r
        FORMS = [("primal-D8", "random features, $D{=}8$"), ("primal-D16", "random features, $D{=}16$"),
                 ("dual-full", "Gram form")]
        lines = []
        for f, lab in FORMS:
            cells = []
            for n in sorted(T):
                r = T[n].get(f)
                if r is None:
                    cells.append("--")
                    continue
                v = float(r["scan_sec"])
                s = f"{v:.1f}" if v < 100 else f"{v:,.0f}"
                if r["scan_measured"] != "True":
                    s += r"$^{*}$"
                cells.append(s)
                PROV.append(dict(table="T5", file=os.path.relpath(tp, ROOT), scen="D2", level=0.7, method=f,
                                 mode=f"n={n}", rate="scan_sec", value=v, printed=s))
            lines.append(lab + " & " + " & ".join(cells) + r" \\")
        open(os.path.join(OUT, "t5_timing.tex"), "w").write(
            "% n = " + ", ".join(str(n) for n in sorted(T)) + "\n" + "\n".join(lines) + "\n")
        print("Table 2 written for n =", sorted(T))


def build_t1_combined():
    """Combined Table 1 (single-break comparison, one column): an upper block with the Gaussian-margin
    designs S1-S4 and the marginal-only control M1 (Table 1 values stored in e1_matched_null/
    table1.csv, plus the full-segment Gram form), and a lower block with the non-Gaussian
    suite G1-G6. Rows = settings, columns = DOMI, Gram form, HSIC, dCor, Spearman, CvM. The three
    joint-distribution references go to Supplementary Table B.6 (tb_jointref.tex).
    Bold marks the largest power in a row (not in the M1 row, where the entry is a false-alarm rate)."""
    t1 = {r["row"]: r for r in csv.DictReader(open(os.path.join(EXP, "e1_matched_null", "table1.csv")))}
    base = overlay_gram(load("matmi_baseline"), ["matmi_baseline_full_T1a"])
    matched = overlay_gram(load("matmi_baseline_matched_D8"), ["matmi_baseline_full_T1b"])
    src_file = "04_DAOU/EXPERIMENT/e1_matched_null/table1.csv"
    COLS = [("S1 r=.35", "S1, $r{=}0.35$", "D1", 0.35, base, "power"),
            ("S2 a=.7", "S2, $a{=}0.7$", "D2", 0.7, base, "power"),
            ("S2 a=.9", "S2, $a{=}0.9$", "D2", 0.9, base, "power"),
            ("S3 τ=.5", "S3, $\\tau{=}0.5$", "D3", 0.5, matched, "power"),
            ("S4 r=.7", "S4, $r{=}0.7$", "D4", 0.7, matched, "power"),
            ("S4 r=.85", "S4, $r{=}0.85$", "D4", 0.85, matched, "power"),
            ("M1 (FA)", "M1, $s{=}2$ (FA)", "M1", 2.0, base, "detect_rate")]
    ROWS = [("DOMI (studentised)", None), (None, "matmi-a1"), ("HSIC difference", None),
            ("Distance correlation", None), ("Spearman difference", None), (None, "CvM-sub")]
    sub = load("cvm_subsample")
    cvm_mode = max(("g", "gs"), key=lambda md: sum(get(sub, s, l, "CvM-sub", md)[0]
                                                  for _, _, s, l, _, rate in COLS if rate == "power"))
    lines = []
    for head, lab, s, l, src, rate in COLS:
        vals = []
        for printed_row, key in ROWS:
            if key == "CvM-sub":
                v = cell(sub, s, l, key, cvm_mode, rate, "T1c")
            elif key:
                v = cell(src, s, l, key, "gs", rate, "T1c")
            else:
                v = float(t1[printed_row][head])
                PROV.append(dict(table="T1c", file=src_file, scen=s, level=l, method=printed_row,
                                 mode="as printed", rate=rate, value=v, printed=fmt(v)))
            vals.append(v)
        top = max(vals)
        cells = [(r"\textbf{" + fmt(v) + "}") if (rate == "power" and fmt(v) == fmt(top)) else fmt(v) for v in vals]
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    upper = "\n".join(lines)
    lower = open(os.path.join(OUT, "t6_nongauss.tex")).read().strip()
    open(os.path.join(OUT, "t1_combined.tex"), "w").write(upper + "\n\\midrule\n" + lower + "\n")
    # joint-distribution references (Supplementary Table B.6)
    ref = []
    for printed_row, lab in [("Joint-state Holevo", "Joint-state Holevo (Section 4.2)"),
                             ("Empirical-copula CvM", "Copula CvM, global pseudo-observations"),
                             ("MMD (median bandwidth)", "MMD (median bandwidth)"),
                             ("Gaussian likelihood ratio", "Gaussian likelihood ratio")]:
        vals = []
        for head, _, s, l, _, rate in COLS:
            v = float(t1[printed_row][head])
            PROV.append(dict(table="B-jointref", file=src_file, scen=s, level=l, method=printed_row,
                             mode="as printed", rate=rate, value=v, printed=fmt(v)))
            vals.append(fmt(v))
        ref.append(lab + " & " + " & ".join(vals) + r" \\")
    open(os.path.join(OUT, "tb_jointref.tex"), "w").write("\n".join(ref) + "\n")
    print("combined Table 1 and joint-reference table written")


def _rows(path):
    return list(csv.DictReader(open(path)))


def _loc(method):
    """Share of MB replicates (a=0.9) with all three breaks within 90 of the truth (mb_location/)."""
    L = _rows(os.path.join(EXP, "mb_location", "results.csv"))
    return float(next(r["all3_within_90"] for r in L if r["method"] == method and r["level"] == "0.9"))


def build_seg():
    """Supplementary Table B.7: multiple-break recovery with the KCP and e.divisive baselines and the
    stage-two re-test (seg_baselines/, run_seg_baselines.py)."""
    src = os.path.join(EXP, "seg_baselines", "results.csv")
    rel = os.path.relpath(src, ROOT)
    R = _rows(src)
    get = lambda d, lv, m, k: float(next(r[k] for r in R if r["design"] == d and r["level"] == lv and r["method"] == m))
    names = [("Holevo-partition", "Holevo partitioning"), ("KCP-raw", "KCP, raw values"), ("KCP-rank", "KCP, ranks"),
             ("edivisive-raw", "e.divisive, raw values"), ("edivisive-rank", "e.divisive, ranks"),
             ("Holevo-partition+DOMI", "Holevo partitioning + DOMI re-test"), ("KCP-rank+DOMI", "KCP, ranks + DOMI re-test"),
             ("edivisive-rank+DOMI", "e.divisive, ranks + DOMI re-test")]
    lines = []
    for m, lab in names:
        vals = []
        loc = _loc(m)
        PROV.append(dict(table="B7-seg", file=os.path.relpath(os.path.join(EXP, "mb_location", "results.csv"), ROOT),
                         scen="MB", level="0.9", method=m, mode="", rate="all3_within_90", value=loc, printed=f"{loc:.3f}"))
        for d, lv, k in [("null", "", "any_detection"), ("MB", "0.9", "k_correct"), ("MB", "0.9", "hausdorff_median"),
                         ("MB", "0.9", "ari_mean"), ("MB", "0.8", "k_correct"), ("M1", "1.6", "any_detection")]:
            v = get(d, lv, m, k)
            PROV.append(dict(table="B7-seg", file=rel, scen=d, level=lv, method=m, mode="", rate=k, value=v,
                             printed=f"{v:.0f}" if k == "hausdorff_median" else (f"{v:.3f}" if k != "ari_mean" else fmt(v))))
            vals.append(PROV[-1]["printed"])
        vals.insert(2, f"{loc:.3f}")
        lines.append(lab + " & " + " & ".join(vals) + r" \\")
    open(os.path.join(OUT, "tb7_seg.tex"), "w").write("\n".join(lines) + "\n")
    print("B.7 segmentation table written")


def build_bmctc():
    """Supplementary Table B.9: BMCTC (our reimplementation of [4]) under the Table 1 calibration, and
    with the Bernaola-Galvan rule of [4] (bmctc/, run_bmctc.py)."""
    src = os.path.join(EXP, "bmctc", "results.csv")
    rel = os.path.relpath(src, ROOT)
    R = [r for r in _rows(src) if r["part"] == "single" and r["null"] != "(null set)"]
    cells = [("S1_r0.35", "S1 $r{=}.35$"), ("S2_a0.7", "S2 $a{=}.7$"), ("S2_a0.9", "S2 $a{=}.9$"),
             ("S3_tau0.5", r"S3 $\tau{=}.5$"), ("S4_r0.7", "S4 $r{=}.7$"), ("S4_r0.85", "S4 $r{=}.85$"),
             ("G1_nu2", r"G1 $\nu{=}2$"), ("G2_tau0.2", r"G2 $\tau{=}.2$"), ("G4_a0.5", "G4 $a{=}.5$"),
             ("G6_b0.3", "G6 $b{=}.3$"), ("G6_b0.5", "G6 $b{=}.5$"), ("M1_s2", "M1 $s{=}2$ (FA)")]
    have = {r["cell"] for r in R}
    lines = []
    for c, lab in cells:
        if c not in have:
            raise SystemExit(f"bmctc cell missing: {c}")
        vals = []
        for stat, var, rate in [("BMCTC-a1.01", "raw", "power_localised"), ("BMCTC-a1.01", "studentised", "power_localised"),
                                ("BMCTC-a2", "raw", "power_localised"), ("BMCTC-a1.01", "BG-rule", "detect_rate")]:
            if c == "M1_s2" and var != "BG-rule":
                rate = "detect_rate"
            v = float(next(r[rate] for r in R if r["cell"] == c and r["stat"] == stat and r["variant"] == var))
            PROV.append(dict(table="B-bmctc", file=rel, scen=c, level="", method=stat, mode=var, rate=rate,
                             value=v, printed=fmt(v)))
            vals.append(fmt(v))
        lines.append(lab + " & " + " & ".join(vals) + r" \\")
    open(os.path.join(OUT, "tb_bmctc.tex"), "w").write("\n".join(lines) + "\n")
    print("BMCTC table written")


def build_detect():
    """Supplementary Table B.10: detection rates without the localization requirement, same replicates,
    variants and rows as Table 1 (detect_rates/table1_detect.csv from build_detect_rates.py; the copula
    column from cvm_subsample/, raw variant as in Table 1)."""
    src = os.path.join(EXP, "detect_rates", "table1_detect.csv")
    csrc = os.path.join(EXP, "cvm_subsample", "results.csv")
    D = _rows(src)
    C = _rows(csrc)
    t1 = open(os.path.join(OUT, "t1_combined.tex")).read()
    lines = []
    for line in t1.splitlines():
        if line.startswith(r"\midrule"):
            lines.append(line)
            continue
        lab = line.split(" & ")[0]
        m = re.match(r"([SGM])(\d), \$([a-z\\]+)\{=\}([0-9.]+)\$", lab)
        scen = {"S": "D", "G": "G", "M": "M"}[m.group(1)] + m.group(2)
        level = float(m.group(4))
        vals = []
        for meth in ("DOMI", "Gram", "HSIC", "dCor", "Spearman"):
            r = next(r for r in D if r["scen"] == scen and abs(float(r["level"]) - level) < 1e-9 and r["method"] == meth)
            v = float(r["detect_rate"])
            PROV.append(dict(table="B10-detect", file=os.path.relpath(src, ROOT), scen=scen, level=level, method=meth,
                             mode=r["variant"], rate="detect_rate", value=v, printed=fmt(v)))
            vals.append(v)
        r = next(r for r in C if r["scen"] == scen and abs(float(r["level"]) - level) < 1e-9 and r["key"] == "CvM-sub|g")
        v = float(r["detect_rate"])
        PROV.append(dict(table="B10-detect", file=os.path.relpath(csrc, ROOT), scen=scen, level=level, method="CvM-sub",
                         mode="g", rate="detect_rate", value=v, printed=fmt(v)))
        vals.append(v)
        top = max(vals)
        cells = [(r"\textbf{" + fmt(v) + "}") if (scen != "M1" and fmt(v) == fmt(top)) else fmt(v) for v in vals]
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    open(os.path.join(OUT, "tb_detect.tex"), "w").write("\n".join(lines) + "\n")
    print("B.10 detection table written")


def _pair(v1, v2):
    return fmt(v1) + "/" + fmt(v2)


def build_permcmp():
    """Supplementary Table B.11: rejection rate / localized power under the pair-permutation calibration the
    procedure uses (K=99, 200 replicates; perm_compare/, run_perm_compare.py). Studentized kernel and
    distance statistics, raw Spearman and copula statistics, as in Table 1."""
    src = os.path.join(EXP, "perm_compare", "results.csv")
    rel = os.path.relpath(src, ROOT)
    R = _rows(src)
    cols = [("DOMI-diff", "stud"), ("Gram-a1", "stud"), ("HSIC-diff", "stud"), ("dCor-diff", "stud"),
            ("Spearman-diff", "raw"), ("CvM-sub", "raw"), ("CopulaCvM", "raw")]
    cells = [("null_S2", r"no change, S2 null"), ("null_S3gauss", r"no change, Gaussian copula $\tau{=}0.5$"),
             ("null_S4mix", r"no change, sign-mixed $r{=}0.85$"),
             ("M1_s2.0", "M1 $s{=}2$ (margins only)"), ("M2_a0.9", "M2 $a{=}0.9$ (margins only)"),
             ("S1_r0.35", "S1 $r{=}0.35$"), ("S2_a0.7", "S2 $a{=}0.7$"), ("S3_tau0.5", r"S3 $\tau{=}0.5$"),
             ("S4_r0.85", "S4 $r{=}0.85$"), ("G6_b0.5", "G6 $b{=}0.5$")]
    lines = []
    for c, lab in cells:
        vals = []
        for st, var in cols:
            r = next(r for r in R if r["cell"] == c and r["stat"] == st and r["variant"] == var)
            if int(r["n_reps"]) < 200:
                vals.append("--")
                continue
            rej = float(r["reject_rate"])
            PROV.append(dict(table="B11-perm", file=rel, scen=c, level="", method=st, mode=var, rate="reject_rate",
                             value=rej, printed=f"{rej:.3f}"))
            if c.startswith("null_"):
                vals.append(f"{rej:.3f}")
            else:
                loc = float(r["power_localised"])
                PROV.append(dict(table="B11-perm", file=rel, scen=c, level="", method=st, mode=var,
                                 rate="power_localised", value=loc, printed=f"{loc:.3f}"))
                vals.append(f"{rej:.3f}/{loc:.3f}")
        lines.append(lab + " & " + " & ".join(vals) + r" \\")
    open(os.path.join(OUT, "tb_permcmp.tex"), "w").write("\n".join(lines) + "\n")
    print("B.11 permutation-comparison table written")


def build_databased():
    """Supplementary data-based simulation table: rejection rate / localized power on empirical weather
    margins (databased_sim/, run_databased_sim.py; K=99 pair permutations, 200 replicates)."""
    src = os.path.join(EXP, "databased_sim", "results.csv")
    rel = os.path.relpath(src, ROOT)
    R = _rows(src)
    stats = ["DOMI", "HSIC-diff", "dCor-diff", "Spearman-diff", "CvM-sub", "CopulaCvM"]
    conds = [("C0", "no change"), ("C1", "margins, winter to summer"),
             ("C2a", "dependence ends"), ("C2b", "dependence begins"),
             ("C3a", "ends, winter to summer"), ("C3b", "begins, winter to summer"),
             ("C4", "margins, December to February"), ("C5a", "ends, December to February"),
             ("C5b", "begins, December to February")]
    lines = []
    for variant, vlab in (("interp", "continuous"), ("rounded", "rounded")):
        first = True
        for c, lab in conds:
            rr = [r for r in R if r["variant"] == variant and r["cond"] == c]
            if not rr:
                continue
            vals = []
            for st in stats:
                r = next(r for r in rr if r["stat"] == st)
                if int(r["reps"]) != 200:
                    sys.exit(f"databased_sim {variant} {c} {st}: {r['reps']} replicates, expected 200")
                rej, loc = float(r["reject_rate"]), float(r["localised_power"])
                for rate, v in (("reject_rate", rej), ("localised_power", loc)):
                    PROV.append(dict(table="B-databased", file=rel, scen=f"{variant}|{c}", level="", method=st,
                                     mode=r["form"], rate=rate, value=v, printed=f"{v:.3f}"))
                vals.append(f"{rej:.3f}" if c == "C0" else f"{rej:.3f}/{loc:.3f}")
            lines.append((vlab if first else "") + f" & {c} {lab} & " + " & ".join(vals) + r" \\")
            first = False
        lines.append(r"\midrule")
    open(os.path.join(OUT, "tb_databased.tex"), "w").write("\n".join(lines[:-1]) + "\n")
    print("data-based simulation table written")


def build_mctire():
    """Supplementary MC-TIRE table: detection rate / localized power of the two MC-TIRE curves under the
    Table 1 calibration (mctire/, run_mctire.py; 500 null and 500 alternative replicates)."""
    src = os.path.join(EXP, "mctire", "results.csv")
    rel = os.path.relpath(src, ROOT)
    R = _rows(src)
    cells = [("S1_r0.35", "S1 $r{=}.35$"), ("S2_a0.7", "S2 $a{=}.7$"), ("S2_a0.9", "S2 $a{=}.9$"),
             ("S3_tau0.5", r"S3 $\tau{=}.5$"), ("S4_r0.7", "S4 $r{=}.7$"), ("S4_r0.85", "S4 $r{=}.85$"),
             ("G1_nu2", r"G1 $\nu{=}2$"), ("G2_tau0.2", r"G2 $\tau{=}.2$"), ("G4_a0.5", "G4 $a{=}.5$"),
             ("G6_b0.3", "G6 $b{=}.3$"), ("G6_b0.5", "G6 $b{=}.5$"), ("M1_s2", "M1 $s{=}2$ (FA)")]
    cols = [("MCTIRE-AS", "raw"), ("MCTIRE-AS", "studentised"), ("MCTIRE-sum", "raw"), ("MCTIRE-sum", "studentised")]
    lines = []
    for c, lab in cells:
        vals = []
        for st, var in cols:
            r = next(r for r in R if r["cell"] == c and r["stat"] == st and r["variant"] == var)
            if int(r["n_reps"]) != 500:
                sys.exit(f"mctire {c} {st} {var}: {r['n_reps']} replicates, expected 500")
            d_, l_ = float(r["detect_rate"]), float(r["power_localised"])
            for rate, v in (("detect_rate", d_), ("power_localised", l_)):
                PROV.append(dict(table="B-mctire", file=rel, scen=c, level="", method=st, mode=var, rate=rate,
                                 value=v, printed=f"{v:.3f}"))
            vals.append(f"{d_:.3f}" if c == "M1_s2" else f"{d_:.3f}/{l_:.3f}")
        lines.append(lab + " & " + " & ".join(vals) + r" \\")
    open(os.path.join(OUT, "tb_mctire.tex"), "w").write("\n".join(lines) + "\n")
    print("MC-TIRE table written")


def _sig3(v):
    """Three significant figures keeping trailing zeros (1.80, not 1.8)."""
    return f"{v:#.3g}".rstrip(".")


def build_cost():
    """Supplementary cost table: test-computation time of each statistic (data and permutation-index
    generation not timed) (cost_compare/, run_cost_compare.py;
    K = 99 pair permutations, S2 at a = 0.7, 10 data sets, one thread per test, 30-minute and 16 GB limits).
    Entry: median seconds over the 10 data sets; ">30 min" timeout, "mem" over the memory limit, "--" not
    run after a failure at a smaller n."""
    src = os.path.join(EXP, "cost_compare", "records.csv")
    if not os.path.exists(src):
        print("cost table skipped: no records"); return
    rel = os.path.relpath(src, ROOT)
    R = [r for r in _rows(src) if r["device"] == "cpu"]
    methods = [("DOMI", "DOMI"), ("Gram", "Gram"), ("HSIC-diff", "HSIC"), ("dCor-diff", "dCor"),
               ("Spearman-diff", "Spearman"), ("CvM-sub", "CvM"), ("CopulaCvM", "CvM, global")]
    cols = [("dense", 600), ("dense", 2400), ("dense", 9600), ("fixed", 600), ("fixed", 2400), ("fixed", 9600),
            ("fixed", 38400), ("fixed", 76800)]
    lines = []
    for m, lab in methods:
        vals = []
        for g, n in cols:
            rr = [r for r in R if r["grid"] == g and r["method"] == m and int(r["n"]) == n]
            st = {r["status"] for r in rr}
            if len(rr) != 10:
                sys.exit(f"cost_compare {g} {m} {n}: {len(rr)} records, expected 10")
            if st == {"ok"}:
                v = float(np.median([float(r["seconds"]) for r in rr]))
                PROV.append(dict(table="B-cost", file=rel, scen=g, level=n, method=m, mode="cpu",
                                 rate="median_seconds", value=v, printed=_sig3(v)))
                vals.append(_sig3(v) if v < 1000 else f"{v:.0f}")
            elif "timeout" in st:
                vals.append(r"$>$30 min")
            elif "memory" in st:
                vals.append("mem")
            elif st == {"not run"}:
                vals.append("--")
            else:
                sys.exit(f"cost_compare {g} {m} {n}: statuses {st}")
        lines.append(lab + " & " + " & ".join(vals) + r" \\")
    open(os.path.join(OUT, "tb_cost.tex"), "w").write("\n".join(lines) + "\n")
    print("cost table written")


def build_signflip():
    """Supplementary sign-reversal table: detection rate / localized power for a change of a Gaussian
    correlation from +r to -r (signflip/, run_signflip.py; Table 1 calibration, 500 null and 500 alternative
    replicates). Kernel and distance statistics studentized, rank, copula and CUSUM statistics raw, as in
    Table B.11; the combined statistic is studentized by construction."""
    src = os.path.join(EXP, "signflip", "results.csv")
    if not os.path.exists(src):
        print("sign-flip table skipped: no results"); return
    rel = os.path.relpath(src, ROOT)
    R = _rows(src)
    rows = [("DOMI-diff", "gs", "DOMI"), ("matmi-a1", "gs", "DOMI, Gram form"), ("HSIC-diff", "gs", "HSIC"),
            ("dCor-diff", "gs", "dCor"), ("Spearman-diff", "g", "Spearman"), ("CvM-sub", "g", "CvM"),
            ("CopulaCvM", "g", "CvM, global"), ("CorrCUSUM", "g", "correlation CUSUM"),
            ("combined", "gs", "combined (Section 4.5)")]
    lines = []
    for st, form, lab in rows:
        vals = []
        for lv in ("0.35", "0.7"):
            r = next(x for x in R if x["stat"] == st and x["form"] == form and x["r"] == lv)
            if int(r["reps"]) != 500:
                sys.exit(f"signflip {st} {form} r={lv}: {r['reps']} replicates, expected 500")
            d_, l_ = float(r["detect_rate"]), float(r["power_localised"])
            for rate, v in (("detect_rate", d_), ("power_localised", l_)):
                PROV.append(dict(table="B-signflip", file=rel, scen=f"r={lv}", level=lv, method=st, mode=form,
                                 rate=rate, value=v, printed=f"{v:.3f}"))
            vals.append(f"{d_:.3f}/{l_:.3f}")
        lines.append(lab + " & " + " & ".join(vals) + r" \\")
    open(os.path.join(OUT, "tb_signflip.tex"), "w").write("\n".join(lines) + "\n")
    print("sign-flip table written")


def build_bmctc_factorial():
    """Supplementary BMCTC 2 x 2 table: localized power on S4 and detection rate under M1 for each input (raw values
    or whole-window ranks) and kernel rule (BMCTC's window standardization with Silverman's width, or DOMI's
    median-heuristic width), Renyi order and form (bmctc_factorial/, run_bmctc_factorial.py; Table B.9 calibration)."""
    src = os.path.join(EXP, "bmctc_factorial", "results.csv")
    if not os.path.exists(src):
        print("BMCTC factorial table skipped: no results"); return
    rel = os.path.relpath(src, ROOT)
    R = _rows(src)
    lab_in = {"raw": "raw", "rank": "ranks"}
    lab_k = {"bmctc": "BMCTC", "domi": "DOMI"}
    lines = []
    for v in "ABCD":
        for stat, alab in (("BMCTC-a1.01", r"1.01"), ("BMCTC-a2", "2")):
            vals = []
            for cell, rate in (("S4_r0.7", "power_localised"), ("S4_r0.85", "power_localised"), ("M1_s2", "detect_rate")):
                for form in ("raw", "studentised"):
                    r = next(x for x in R if x["variant"] == v and x["cell"] == cell and x["stat"] == stat and x["form"] == form)
                    if int(r["n_reps"]) != 500:
                        sys.exit(f"bmctc_factorial {v} {cell} {stat} {form}: {r['n_reps']} replicates")
                    val = float(r[rate])
                    PROV.append(dict(table="B-bmctc-factorial", file=rel, scen=cell, level=v, method=stat, mode=form,
                                     rate=rate, value=val, printed=f"{val:.2f}"))
                    vals.append(f"{val:.2f}")
            r0 = next(x for x in R if x["variant"] == v)
            lines.append(f"{lab_in[r0['input']]} & {lab_k[r0['kernel']]} & {alab} & " + " & ".join(vals) + r" \\")
    open(os.path.join(OUT, "tb_bmctc_factorial.tex"), "w").write("\n".join(lines) + "\n")
    print("BMCTC factorial table written")


def _p3(x):
    """p-values lie on the grid 1/(K+1) = 0.001; a median of 20 draws may fall halfway, so print it exactly."""
    t = f"{x:.4f}"
    return t[:-1] if t.endswith("0") else t


def build_finance_seeds():
    """Supplementary table: single-break p-values on the daily 2021-2022 stock-bond window, median (range) over 20
    independent permutation draws of K = 999 (combined_block_seeds20/, run_combined_block_seeds.py)."""
    src = os.path.join(EXP, "combined_block_seeds20", "summary.json")
    if not os.path.exists(src):
        print("finance seed table skipped: no results"); return
    rel = os.path.relpath(src, ROOT)
    S = json.load(open(src))
    lines = []
    for m, lab in (("DOMI-diff", "DOMI"), ("combined", "combined (Section 4.5)"), ("Spearman-diff", "Spearman"),
                   ("HSIC-diff", "HSIC"), ("CopulaCvM", "CvM, global")):
        cells = []
        for b in ("pair", "20", "50"):
            v = S[b][m]
            if v["n_draws"] != 20:
                sys.exit(f"finance seeds {b} {m}: {v['n_draws']} draws")
            for k in ("median", "min", "max"):
                PROV.append(dict(table="B-finance-seeds", file=rel, scen=b, level="", method=m, mode=k,
                                 rate="p", value=v[k], printed=_p3(v[k])))
            cells.append(f"{_p3(v['median'])} ({_p3(v['min'])}--{_p3(v['max'])})")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    open(os.path.join(OUT, "tb_finance_seeds.tex"), "w").write("\n".join(lines) + "\n")
    print("finance seed table written")


US_PAIRS = (("SPX-GLD", "S\\&P 500--gold"), ("SPX-TNX", "S\\&P 500--10-year yield"), ("SPX-DXY", "S\\&P 500--dollar index"))


def _ymd(d):
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def build_us_panel():
    """Supplementary table: the US financial panel of Section 6.8 (us_panel/, run_us_finance_panel.py and
    run_us_finance_stage2.py). Upper block: whole-record single-break tests, K = 9999; lower block: stage two."""
    src, src2 = os.path.join(EXP, "us_panel", "summary.json"), os.path.join(EXP, "us_panel", "stage2.json")
    if not (os.path.exists(src) and os.path.exists(src2)):
        print("US panel table skipped: no results"); return
    S, T = json.load(open(src)), json.load(open(src2))
    rel, rel2 = os.path.relpath(src, ROOT), os.path.relpath(src2, ROOT)
    def rec(f, pair, key, v, printed):
        PROV.append(dict(table="B-us-panel", file=f, scen=pair, level="", method=key, mode="", rate="", value=v,
                         printed=printed))
        return printed
    top, bot = [], []
    for pair, lab in US_PAIRS:
        a = S[pair]; prim = a["primary_statistic"]; sen = a["sensitivity"][prim]
        if sen["n_draws"] != 20:
            sys.exit(f"US panel {pair}: {sen['n_draws']} sensitivity draws")
        cells = [lab, "DOMI" if prim == "DOMI-diff" else "combined",
                 rec(rel, pair, "block", a["diag"]["block"], f"{a['diag']['block']} ({a['diag']['n_blocks']})"),
                 rec(rel, pair, "primary_p", a["primary_p"], f"{a['primary_p']:.4f}"),
                 rec(rel, pair, "sens", (sen["min"], sen["max"]), f"{sen['min']:.3f}--{sen['max']:.3f}"),
                 rec(rel, pair, "bh_q", a["bh_q"], f"{a['bh_q']:.4f}"),
                 rec(rel, pair, "p_DOMI", a["p"]["DOMI-diff"], f"{a['p']['DOMI-diff']:.4f}"),
                 rec(rel, pair, "p_Spearman", a["p"]["Spearman-diff"], f"{a['p']['Spearman-diff']:.4f}"),
                 rec(rel, pair, "p_HSIC", a["p"]["HSIC-diff"], f"{a['p']['HSIC-diff']:.4f}"),
                 rec(rel, pair, "tau", a["tau_date"][prim], _ymd(a["tau_date"][prim]))]
        top.append(" & ".join(cells) + r" \\")
        b = T["results"][pair]; fd = b["feature_draws"]; w0, w1 = b["window"].split("-")
        cells = [lab, f"{_ymd(w0)} to {_ymd(w1)}",
                 rec(rel2, pair, "n_blocks", b["n_blocks"], str(b["n_blocks"])),
                 rec(rel2, pair, "p_dep", b["p_dep"], f"{b['p_dep']:.3f}"),
                 rec(rel2, pair, "q_dep", b["bh_q_dep"], f"{b['bh_q_dep']:.3f}"),
                 rec(rel2, pair, "p_marg_x", b["p_marg_x"], f"{b['p_marg_x']:.3f}"),
                 rec(rel2, pair, "p_marg_y", b["p_marg_y"], f"{b['p_marg_y']:.3f}"),
                 rec(rel2, pair, "fd", (fd["min"], fd["max"]), f"{fd['min']:.3f}--{fd['max']:.3f}"),
                 rec(rel2, pair, "fd_le05", fd["frac_le_05"], f"{round(20 * fd['frac_le_05'])} of 20"),
                 {"coupling change": "dependence", "change in both": "both", "marginal-driven": "scale",
                  "undetermined": "undetermined"}[b["cls"]]]
        bot.append(" & ".join(cells) + r" \\")
    open(os.path.join(OUT, "tb_us_panel.tex"), "w").write("\n".join(top) + "\n%%\n" + "\n".join(bot) + "\n")
    print("US panel table written")


def build_us_yearly():
    """Supplementary table: calendar-year Spearman correlation and bias-corrected DOMI of each US pair
    (us_panel/describe_yearly.json, describe_us_panel_breaks.py yearly)."""
    src = os.path.join(EXP, "us_panel", "describe_yearly.json")
    if not os.path.exists(src):
        print("US yearly table skipped: no results"); return
    Y = json.load(open(src)); rel = os.path.relpath(src, ROOT)
    years = sorted(Y["SPX-GLD"]["years"])
    lines = []
    for yr in years:
        cells = [yr]
        for pair, _ in US_PAIRS:
            v = Y[pair]["years"][yr]
            for k, f in (("spearman", "{:.2f}"), ("domi_bc", "{:.3f}")):
                t = f.format(v[k]).replace("-", "$-$")
                PROV.append(dict(table="B-us-yearly", file=rel, scen=pair, level=yr, method=k, mode="", rate="",
                                 value=v[k], printed=t))
                cells.append(t)
        lines.append(" & ".join(cells) + r" \\")
    open(os.path.join(OUT, "tb_us_yearly.tex"), "w").write("\n".join(lines) + "\n")
    print("US yearly table written")


def build_offcentre():
    """Supplementary Table B.12: detection rate / localized power with the break at tau = 150, 300, 450
    (offcenter/, run_offcentre.py; Table 1 calibration, 500 replicates; studentized forms)."""
    src = os.path.join(EXP, "offcentre", "results.csv")
    rel = os.path.relpath(src, ROOT)
    R = _rows(src)
    stats = [("DOMI-diff", "DOMI"), ("matmi-a1", "Gram"), ("HSIC-diff", "HSIC"), ("dCor-diff", "dCor")]
    cells = [("S2_a0.9", "S2 $a{=}0.9$"), ("S4_r0.85", "S4 $r{=}0.85$"), ("G4_a0.5", "G4 $a{=}0.5$"),
             ("G6_b0.5", "G6 $b{=}0.5$")]
    lines = []
    for c, lab in cells:
        for tau in ("150", "300", "450"):
            vals = []
            for st, _ in stats:
                r = next(r for r in R if r["cell"] == c and r["stat"] == st and r["variant"] == "studentised"
                         and r["tau"] == tau)
                d_, l_ = float(r["detect_rate"]), float(r["power_localised"])
                for rate, v in (("detect_rate", d_), ("power_localised", l_)):
                    PROV.append(dict(table="B12-offc", file=rel, scen=c, level=tau, method=st, mode="gs",
                                     rate=rate, value=v, printed=fmt(v)))
                vals.append(_pair(d_, l_))
            lines.append((lab if tau == "150" else "") + f" & {tau} & " + " & ".join(vals) + r" \\")
    open(os.path.join(OUT, "tb_offcentre.tex"), "w").write("\n".join(lines) + "\n")
    print("B.12 off-center table written")


def build_asym():
    """Supplementary Table B.13: permutation-free calibration at the independence boundary
    (asymptotic_threshold/results.json, run_asymptotic_threshold.py)."""
    src = os.path.join(EXP, "asymptotic_threshold", "results.json")
    rel = os.path.relpath(src, ROOT)
    d = json.load(open(src))
    lines = []
    for n in ("600", "2400", "9600", "38400"):
        e = d["exactmc"][n]
        row = [f"{int(n):,}".replace(",", "{,}"),
               f"{e['q95']:.2f} [{e['q95_ci'][0]:.2f}, {e['q95_ci'][1]:.2f}]", f"{e['asym_crit_median']:.2f}"]
        for k, v in (("exactmc_q95", e["q95"]), ("asym_crit_median", e["asym_crit_median"])):
            PROV.append(dict(table="B13-asym", file=rel, scen="H0", level=n, method=k, mode="", rate="value",
                             value=v, printed=f"{v:.2f}"))
        for null in ("H0", "M1"):
            for m in ("asym_raw", "asym_std"):
                v = d["level"][f"{null}_{n}"][m]["rate"]
                PROV.append(dict(table="B13-asym", file=rel, scen=null, level=n, method=m, mode="", rate="level",
                                 value=v, printed=f"{v:.3f}"))
                row.append(f"{v:.3f}")
        t = d["timing"][n]
        for k in ("wall_asym", "wall_perm_K99"):
            PROV.append(dict(table="B13-asym", file=rel, scen="", level=n, method=k, mode="", rate="seconds",
                             value=t[k], printed=f"{t[k]:.2f}"))
        row += [f"{t['wall_asym']:.2f}", f"{t['wall_perm_K99']:.2f}"]
        lines.append(" & ".join(row) + r" \\")
    open(os.path.join(OUT, "tb_asym.tex"), "w").write("\n".join(lines) + "\n")
    print("B.13 asymptotic table written")


def build_blockperm():
    """Supplementary Table B.5: level under serial dependence, pair and block permutation, symmetric
    studentization over all K+1 curves (block_perm/results.csv, run_block_perm_check.py)."""
    import csv as _csv
    src = os.path.join(EXP, "block_perm", "results.csv")
    rel = os.path.relpath(src, ROOT)
    rows = {(r["null"], r["scheme"]): float(r["fpr"]) for r in _csv.DictReader(open(src))}
    labels = {"iid": "serially independent", "ar1": "AR(1) margins, $\\phi = 0.6$, constant Gaussian copula",
              "garch": "GARCH(1,1) volatility clustering"}
    lines = []
    for null in ("iid", "ar1", "garch"):
        cells = []
        for sch in ("pair", "block20", "block50"):
            v = rows[(null, sch)]
            PROV.append(dict(table="B5-blockperm", file=rel, scen=null, level="", method=sch, mode="symmetric",
                             rate="fpr", value=v, printed=f"{v:.3f}"))
            cells.append(f"{v:.3f}")
        lines.append(labels[null] + " & " + " & ".join(cells) + r" \\")
    open(os.path.join(OUT, "tb_blockperm.tex"), "w").write("\n".join(lines) + "\n")
    print("B.5 block-permutation table written")


def _rows_csv(path):
    import csv as _csv
    return list(_csv.DictReader(open(path, encoding="utf-8")))


def build_mv():
    """Supplementary Table B.1: vector-valued blocks (mv_blocks/results.csv, run_mv_blocks.py)."""
    src = os.path.join(EXP, "mv_blocks", "results.csv")
    rel = os.path.relpath(src, ROOT)
    v = {(r["scen"], r["d"], r["method"]): float(r["power"]) for r in _rows_csv(src)}
    # the margins-only control MV-M1 is reported as a rejection rate (any detection), from the per-replicate records;
    # results.csv keeps only the localized rate there, which is not a false-alarm rate
    rec = os.path.join(EXP, "mv_blocks", "records_results.csv")
    acc = defaultdict(list)
    for r in _rows_csv(rec):
        if r["scen"] == "MV-M1" and r["side"] == "alt":
            acc[(r["d"], r["method"])].append(int(r["detect"]))
    rej = {k: sum(x) / len(x) for k, x in acc.items()}
    spec = [("MV-S2, DOMI on the blocks", "MV-S2", "DOMI"),
            ("MV-S2, DOMI pair by pair, Bonferroni", "MV-S2", "DOMI-pairwise/Bonf"),
            ("MV-S2, HSIC on the blocks", "MV-S2", "HSIC-diff"),
            ("MV-M1, DOMI on the blocks: rejection rate (nominal 0.05)", "MV-M1", "DOMI"),
            ("MV-M1, DOMI pair by pair, Bonferroni: rejection rate (nominal 0.05)", "MV-M1", "DOMI-pairwise/Bonf")]
    lines = []
    for lab, sc, m in spec:
        cells = []
        for d in ("1", "2", "3", "5"):
            x = rej[(d, m)] if sc == "MV-M1" else v[(sc, d, m)]
            PROV.append(dict(table="B1-mv", file=os.path.relpath(rec, ROOT) if sc == "MV-M1" else rel, scen=sc, level=d,
                             method=m, mode="", rate="reject_rate" if sc == "MV-M1" else "power",
                             value=x, printed=f"{x:.2f}"))
            cells.append(f"{x:.2f}")
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    open(os.path.join(OUT, "tb_mv.tex"), "w").write("\n".join(lines) + "\n")
    print("B.1 vector-block table written")


def build_bandwidth():
    """Supplementary Table B.2: kernel bandwidth (bandwidth_matched_null/tableB2.csv, written by
    rebuild_tableB2.py from the sweep aggregates)."""
    src = os.path.join(EXP, "bandwidth_matched_null", "tableB2.csv")
    rel = os.path.relpath(src, ROOT)
    rows = _rows_csv(src)
    cols = list(rows[0].keys())[1:]
    lab = {"DOMI, ×1 (deployed)": "DOMI, $\\times 1$ (deployed)", "DOMI, ×1/2": "DOMI, $\\times 1/2$",
           "DOMI, ×1/4": "DOMI, $\\times 1/4$", "HSIC, full Gram, ×1": "HSIC, full Gram, $\\times 1$",
           "HSIC, full Gram, ×1/4": "HSIC, full Gram, $\\times 1/4$"}
    lines = []
    for r in rows:
        cells = []
        for c in cols:
            x = float(r[c])
            PROV.append(dict(table="B2-bandwidth", file=rel, scen=c, level="", method=r["row"], mode="",
                             rate="power", value=x, printed=f"{x:.2f}"))
            cells.append(f"{x:.2f}")
        lines.append(lab[r["row"]] + " & " + " & ".join(cells) + r" \\")
    open(os.path.join(OUT, "tb_bandwidth.tex"), "w").write("\n".join(lines) + "\n")
    print("B.2 bandwidth table written")


def build_e3perm():
    """Supplementary Table B.8: DOMI under pair-permutation calibration (e3/results.csv, run_experiment e3)."""
    src = os.path.join(EXP, "e3", "results.csv")
    rel = os.path.relpath(src, ROOT)
    name = {"D1": ("S1", "r"), "D2": ("S2", "a"), "D4": ("S4", "r"), "M1": ("M1", "s")}   # article labels, as in Table 1
    lines = []
    for r in _rows_csv(src):
        if r["method"] != "DOMI-diff":
            continue
        f, p = float(r["fpr_perm"]), float(r["power_perm"])
        for k, x in (("fpr_perm", f), ("power_perm", p)):
            PROV.append(dict(table="B8-e3", file=rel, scen=r["scen"], level=r["level"], method="DOMI-diff",
                             mode="", rate=k, value=x, printed=f"{x:.2f}"))
        sc, sym = name[r["scen"]]
        lines.append(f"{sc}, ${sym}{{=}}{r['level']}$ & {f:.2f} & {p:.2f} " + r"\\")
    open(os.path.join(OUT, "tb_e3perm.tex"), "w").write("\n".join(lines) + "\n")
    print("B.8 permutation table written")

if __name__ == "__main__":
    main()
