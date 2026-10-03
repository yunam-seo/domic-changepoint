"""Main-text Tables 3 (three-break benchmark MB) and 4 (US financial panel), rendered from stored outputs.

Reads   04_DAOU/EXPERIMENT/{seg_baselines,e2,mb_location,bmctc}/results.csv, domi_binseg/results.json,
        us_panel/{summary,stage2}.json
Writes  04_DAOU/EXPERIMENT/tables/table3_mb.tex, table4_finance.tex
Every printed entry is read from those files; nothing is typed by hand."""
import csv, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = os.path.join(ROOT, "04_DAOU", "EXPERIMENT")
OUT = os.path.join(E, "tables")


def rows(path):
    return list(csv.DictReader(open(os.path.join(E, path))))


def pick(rs, **kw):
    hit = [r for r in rs if all(r[k] == v for k, v in kw.items())]
    assert len(hit) == 1, (kw, len(hit))
    return hit[0]


seg, e2, loc, bm = rows("seg_baselines/results.csv"), rows("e2/results.csv"), rows("mb_location/results.csv"), \
    rows("bmctc/results.csv")
qb = json.load(open(os.path.join(E, "domi_binseg", "results.json")))
f3 = lambda x: f"{float(x):.3f}"
f2 = lambda x: f"{float(x):.2f}"

# ---- Table 3: MB, columns null rate | P(K=3) at a=0.9 | all three within 90 | ARI | P(K=3) at a=0.8
T3 = []
def seg_row(label, m, locsrc, locm):
    nul = pick(seg, design="null", method=m)["any_detection"]
    a9, a8 = pick(seg, design="MB", level="0.9", method=m), pick(seg, design="MB", level="0.8", method=m)
    w = pick(loc, source=locsrc, method=locm, level="0.9")["all3_within_90"]
    T3.append((label, f3(nul), f3(a9["k_correct"]), f3(w), f2(a9["ari_mean"]), f3(a8["k_correct"])))
seg_row("Holevo partitioning", "Holevo-partition", "e2", "Holevo-partition")
for m, lab in (("PELT-Gauss", "Rank-Gaussian PELT"), ("BinSeg-MMD", "Kernel binary seg.")):
    a9, a8 = pick(e2, scen="P1", level="0.9", method=m), pick(e2, scen="P1", level="0.8", method=m)
    w = pick(loc, source="e2", method=m, level="0.9")["all3_within_90"]
    T3.append((lab, f3(a9["null_fa"]), f3(a9["k_correct"]), f3(w), f2(a9["ari_mean"]), f3(a8["k_correct"])))
seg_row("KCP, raw values", "KCP-raw", "seg_baselines", "KCP-raw")
seg_row("KCP, ranks", "KCP-rank", "seg_baselines", "KCP-rank")
seg_row("e.divisive, raw values", "edivisive-raw", "seg_baselines", "edivisive-raw")
seg_row("e.divisive, ranks", "edivisive-rank", "seg_baselines", "edivisive-rank")
b9 = pick(bm, part="multi", cell="MB_a0.9", stat="BMCTC-a1.01", variant="calibrated")
b8 = pick(bm, part="multi", cell="MB_a0.8", stat="BMCTC-a1.01", variant="calibrated")
w = pick(loc, source="bmctc", method="BMCTC-a1.01", level="0.9")["all3_within_90"]
T3.append(("BMCTC, $\\alpha{=}1.01$", f3(b9["null_fa"]), f3(b9["k_correct"]), f3(w), f2(b9["ari_mean"]),
           f3(b8["k_correct"])))
mb = qb["multibreak"]
T3.append(("DOMI binary seg.", f3(mb["null_any_break"]), f3(mb["exactly_three"]), "--", "--", "--"))

t3 = ["\\begin{table}[!t]", "\\centering",
      "\\caption{Three-break benchmark MB (200 replicates). Null: achieved null detection rate after calibration to a "
      "nominal 5\\%. At $a=0.9$: probability of exactly three breaks, of all three within 90 time points of the true "
      "breaks, and mean adjusted Rand index (ARI); last column: exactly three at $a=0.8$. --: not computed. More in "
      "Supplementary Table B.7.}",
      "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{l ccccc}", "\\toprule",
      "Method & Null & $P(\\hat K{=}3)$ & All three & ARI & $a{=}0.8$ \\\\", "\\midrule"]
t3 += [" & ".join(r) + " \\\\" for r in T3]
t3 += ["\\bottomrule", "\\end{tabular}", "\\label{tab:mb}", "\\end{table}", ""]

# ---- Table 4: US panel
s = json.load(open(os.path.join(E, "us_panel", "summary.json")))
st = json.load(open(os.path.join(E, "us_panel", "stage2.json")))
nd = st["config"]["n_draws"]
LAB = {"SPX-GLD": "Stocks--gold", "SPX-TNX": "Stocks--Treasury", "SPX-DXY": "Stocks--dollar"}
T4 = []
for k in ("SPX-GLD", "SPX-TNX", "SPX-DXY"):
    d, r = s[k], st["results"][k]
    T4.append((LAB[k], str(d["diag"]["block"]), f3(d["primary_p"]), f3(d["bh_q"]),
               f"{r['date'][:4]}-{r['date'][4:6]}", str(r["n_blocks"]), f3(r["bh_q_dep"]),
               f"{round(r['feature_draws']['frac_le_05'] * nd)}/{nd}"))
t4 = ["\\begin{table}[!t]", "\\centering",
      "\\caption{Pre-specified US financial panel (Section 6.8; daily, 2011-08-23 to 2026-08-20). Stage one: "
      "whole-record single-break test (DOMI for gold, the combined statistic otherwise) under block permutation at block "
      "length $\\hat b$ (days), Benjamini--Hochberg $q$ over the three pairs. Stage two: dependence re-test on the "
      "largest centered window of whole blocks (Blocks), Benjamini--Hochberg $q$ over the three pairs; Draws: feature "
      "draws with $p\\le0.05$.}",
      "\\footnotesize", "\\setlength{\\tabcolsep}{2pt}", "\\begin{tabular}{l ccc cccc}", "\\toprule",
      " & \\multicolumn{4}{c}{Stage one} & \\multicolumn{3}{c}{Stage two} \\\\",
      "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}",
      "Pair & $\\hat b$ & $p$ & $q$ & Break & Blocks & $q$ & Draws \\\\", "\\midrule"]
t4 += [" & ".join(r) + " \\\\" for r in T4]
t4 += ["\\bottomrule", "\\end{tabular}", "\\label{tab:fin}", "\\end{table}", ""]

os.makedirs(OUT, exist_ok=True)
for name, body in (("table3_mb.tex", t3), ("table4_finance.tex", t4)):
    p = os.path.join(OUT, name)
    if os.path.exists(p):
        old = open(p).read()
        assert old == "\n".join(body), f"{p} exists with different content; write a new version instead"
    open(p, "w").write("\n".join(body))
    print("\n".join(body))
