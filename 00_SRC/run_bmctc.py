#!/usr/bin/env python
"""BMCTC (Qian et al. 2024, reference [4]) under the article's calibration protocol.

What
    The closest prior method: a moving-window matrix-based Renyi total correlation (TC) sequence
    segmented by the Bernaola-Galvan (BG) algorithm. Our reimplementation is in dots/bmctc.py;
    its docstring lists every choice the description in [4] leaves open (alpha = 1.01 primary
    and 2 secondary, Gaussian kernel on window-z-scored raw values with Silverman's width
    1.06 L^(-1/5), window L = 60, step 1, BG delta = 0.40, eta = 4.19 ln N - 11.54, P0 = 0.95,
    minimum BG segment l0 = 30 TC points). BMCTC is deterministic given the data.

Part 1 -- single break on the designs of Table 1 (Supplementary Section B.13, Table B.9),
    n = 600, tau = 300, 500 null + 500 alternative replicates
    Cells: S1 r=0.35, S2 a=0.7/0.9, S3 tau=0.5, S4 r=0.7/0.85, M1 s=2 (false alarm), G1 nu=2,
    G2 tau=0.2, G4 a=0.5, G6 b=0.3/0.5 (scenario codes D1-D4 = S1-S4). Same generators and base
    seed (20260825) as run_offcentre.py / run_cvm_subsample.py; S3 and S4 against the matched null
    (the pre-change regime at the alternative's own level held throughout), every other cell
    against its scenario's level-0 null.
    Scan curve: the BG statistic T(k) between the left and right means of the TC sequence (TC
    values with window center < k vs >= k) at every candidate k = 60..540.
      raw "g":          detect if max_k T(k) > 95% quantile of the null max (all 500 nulls)
      studentized "gs": per-candidate null mean/sd from nulls 0..249, threshold from 250..499
      localized power:  detect and |tau_hat - tau| <= 30, tau_hat = argmax of the curve used
    Rule of [4] (uncalibrated): the top-level BG test, P(T_max) >= 0.95, i.e. "the BG
    algorithm places at least one break"; its rate is reported on every null set (false alarm),
    on M1 (false alarm) and on the alternatives.

Part 2 -- multi-break scenario MB (code P1: n = 1800, dependence breaks at 450/900/1350)
    Levels a = 0.8, 0.9, 200 replicates each and 200 null replicates, base seed 20260825, exactly
    the draws of run_experiment.py --part e2. Recursive BG on the TC sequence:
      calibrated: the split rule q < q*, q = -log P_BG(T_max), q* = 5% quantile of the top-level q
                  over the null replicates (null any-break rate <= 5%, as e2 calibrates its
                  penalties); the same q* is applied at every recursion level.
      rule of [4]: P0 = 0.95.
    Scored as e2: exactly-three-breaks rate, mean |K_hat - 3|, median Hausdorff distance, mean
    adjusted Rand index; plus true breaks recovered within 60 (as run_domi_binseg.py).

Run
    python 00_SRC/run_bmctc.py --procs 16            # compute (TC sequences) and evaluate
    python 00_SRC/run_bmctc.py --evaluate            # re-evaluate from the stored TC sequences
Outputs  04_DAOU/EXPERIMENT/bmctc/
    tc/<set>.npz            TC sequences, one row per replicate, per alpha (float64)
    records_null.csv.gz     per null replicate and alpha: max of raw curve, tau_hat, BG q, BG verdict
    records_alt.csv.gz      per alternative replicate and statistic/variant: max, tau_hat, detected,
                            localized hit, |tau_hat - tau|
    records_mb.csv.gz       per MB replicate, alpha and rule: breaks found, K, exact3, hits,
                            Hausdorff, ARI
    results.csv             one row per (part, cell, statistic, variant)
    summary.json            configuration, thresholds, q*, underflow counts, rows, runtime
    run.log
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from dots.synth import SCENARIOS, generate  # noqa: E402
from dots.synth_ng import SCENARIOS_NG, generate_ng  # noqa: E402
from dots.extras import studentize  # noqa: E402
from dots.evaluate import summarize_alt, aggregate  # noqa: E402
from dots.persist import save_records  # noqa: E402
from dots.pelt import hausdorff, rand_index_adj  # noqa: E402
from dots import bmctc as B  # noqa: E402

OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "bmctc")
TC = os.path.join(OUT, "tc")
CFG = dict(n=600, tau=300, w=60, tol=30, fpr=0.05, base_seed=20260825, reps=500,
           mb_reps=200, mb_taus=[450, 900, 1350], mb_tol=60,
           L=B.L_WIN, alphas=list(B.ALPHAS), l0=B.L0, P0=0.95)
# cell -> (scenario code, level, matched null)
CELLS = {
    "S1_r0.35": ("D1", 0.35, False), "S2_a0.7": ("D2", 0.7, False), "S2_a0.9": ("D2", 0.9, False),
    "S3_tau0.5": ("D3", 0.5, True), "S4_r0.7": ("D4", 0.7, True), "S4_r0.85": ("D4", 0.85, True),
    "M1_s2": ("M1", 2.0, False), "G1_nu2": ("G1", 2.0, False), "G2_tau0.2": ("G2", 0.2, False),
    "G4_a0.5": ("G4", 0.5, False), "G6_b0.3": ("G6", 0.3, False), "G6_b0.5": ("G6", 0.5, False),
}
STAT = {1.01: "BMCTC-a1.01", 2.0: "BMCTC-a2"}


def _levels(scen):
    return (SCENARIOS_NG if scen.startswith("G") else SCENARIOS)[scen]["levels"]


def null_set(cell):
    scen, level, matched = CELLS[cell]
    li = _levels(scen).index(level) if matched else 0
    return f"null_{scen}_L{li}"


def set_list():
    sets = []
    for c in CELLS:
        if null_set(c) not in sets:
            sets.append(null_set(c))
    sets += [f"alt_{c}" for c in CELLS]
    sets += ["mb_null"] + [f"mb_alt_L{li}" for li in range(len(SCENARIOS["P1"]["levels"]))]
    return sets


def draw(setname, r):
    """(x, y) of replicate r of a set, from the article's generators and seeds."""
    bs = CFG["base_seed"]
    if setname.startswith("mb_"):
        li = 0 if setname == "mb_null" else int(setname.split("_L")[1])
        smp = generate("P1", SCENARIOS["P1"]["levels"][li], r, null=setname == "mb_null",
                       base_seed=bs, level_idx=li)
    elif setname.startswith("null_"):
        _, scen, lis = setname.split("_")
        li = int(lis[1:])
        gen = generate_ng if scen.startswith("G") else generate
        smp = gen(scen, _levels(scen)[li], r, n=CFG["n"], null=True, base_seed=bs, level_idx=li)
    else:
        scen, level, _ = CELLS[setname[4:]]
        gen = generate_ng if scen.startswith("G") else generate
        smp = gen(scen, level, r, n=CFG["n"], tau=CFG["tau"], null=False, base_seed=bs,
                  level_idx=_levels(scen).index(level))
    Z = smp["Z"]
    return Z[:, smp["blocks"][0][0]], Z[:, smp["blocks"][1][0]]


def _job(a):
    setname, r = a
    x, y = draw(setname, r)
    return B.tc_series(x, y, CFG["L"], tuple(CFG["alphas"]))[1]


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(os.path.join(OUT, "run.log"), "a") as f:
        f.write(line + "\n")


def compute(procs):
    with Pool(procs) as pool:
        for s in set_list():
            p = os.path.join(TC, f"{s}.npz")
            if os.path.exists(p):
                continue
            t0 = time.time()
            reps = CFG["mb_reps"] if s.startswith("mb_") else CFG["reps"]
            res = pool.map(_job, [(s, r) for r in range(reps)], chunksize=5)
            arrs = {f"a{a}": np.array([d[a] for d in res]) for a in CFG["alphas"]}
            tmp = os.path.join(TC, f".{s}.tmp.npz")
            np.savez_compressed(tmp, **arrs)
            os.replace(tmp, p)
            say(f"{s}: {reps} reps {time.time()-t0:.0f}s")


def load(s):
    z = np.load(os.path.join(TC, f"{s}.npz"))
    return {a: z[f"a{a}"] for a in CFG["alphas"]}


def curves(tcs):
    """(reps, N) TC sequences -> (reps, G) BG statistic on the candidate grid 60..540.

    Split j (left count) <-> time j + L/2. With N = 541 TC values, BG's admissible splits are
    j = 30..511; the article's grid 60..540 is j = 30..510, so the last split is dropped."""
    return np.array([B.bg_tstat(s, CFG["l0"])[1][:-1] for s in tcs])


def _se(p, n):
    return round(float(np.sqrt(p * (1 - p) / n)), 4)


# ---------------------------------------------------------------- part 1
def evaluate_single(rows, info):
    n, w, tol, fpr, tau = CFG["n"], CFG["w"], CFG["tol"], CFG["fpr"], CFG["tau"]
    grid = np.arange(w, n - w + 1)
    half = CFG["reps"] // 2
    NTC = n - CFG["L"] + 1
    assert len(grid) == NTC - 2 * CFG["l0"] and grid[0] == CFG["l0"] + CFG["L"] // 2
    nulls = {}
    for c in CELLS:
        s = null_set(c)
        if s in nulls:
            continue
        tcs = load(s)
        cur = {a: curves(tcs[a]) for a in CFG["alphas"]}
        mu0 = {a: cur[a][:half].mean(0) for a in cur}
        sd0 = {a: cur[a][:half].std(0) for a in cur}
        thr = {}
        for a in cur:
            thr[f"{STAT[a]}|g"] = float(np.quantile(cur[a].max(1), 1 - fpr))
            thr[f"{STAT[a]}|gs"] = float(np.quantile(
                [np.max(studentize(v, mu0[a], sd0[a])) for v in cur[a][half:]], 1 - fpr))
        nulls[s] = (cur, mu0, sd0, thr)
        recs = []
        for r in range(CFG["reps"]):
            rec = {}
            for a in cur:
                tm = float(cur[a][r].max())
                q = B.bg_q(tm, NTC)
                rec.update({f"max|{STAT[a]}|g": tm, f"tau_hat|{STAT[a]}": int(grid[np.argmax(cur[a][r])]),
                            f"bgq|{STAT[a]}": q, f"bg_detect|{STAT[a]}": int(q < B.Q_PUBLISHED)})
            recs.append(rec)
        save_records(OUT, "records_null.csv", recs, {"set": s})
        for a in cur:
            fa = float(np.mean([rr[f"bg_detect|{STAT[a]}"] for rr in recs]))
            rows.append(dict(part="single", cell=s, scen=s.split("_")[1], level="", null="(null set)",
                             stat=STAT[a], variant="BG-rule", detect_rate=round(fa, 4),
                             detect_se=_se(fa, CFG["reps"]), threshold=f"P0={CFG['P0']}",
                             n_reps=CFG["reps"]))
        info["thresholds"][s] = thr
        info["null_q_zero"][s] = {STAT[a]: int(sum(rr[f"bgq|{STAT[a]}"] == 0 for rr in recs)) for a in cur}
    for c, (scen, level, matched) in CELLS.items():
        cur0, mu0, sd0, thr = nulls[null_set(c)]
        tcs = load(f"alt_{c}")
        cur = {a: curves(tcs[a]) for a in CFG["alphas"]}
        recs = []
        for r in range(CFG["reps"]):
            st = {}
            for a in cur:
                st[f"{STAT[a]}|g"] = cur[a][r]
                st[f"{STAT[a]}|gs"] = studentize(cur[a][r], mu0[a], sd0[a])
            rec = summarize_alt(st, grid, tau, w, n, thr, tol)
            for a in cur:   # BG verdict of [4] on the same replicate
                k = f"{STAT[a]}|bg"
                tm = float(cur[a][r].max())
                q = B.bg_q(tm, NTC)
                th = int(grid[np.argmax(cur[a][r])])
                det = q < B.Q_PUBLISHED
                rec[k] = dict(max=tm, tau_hat=th, detected=bool(det), power_hit=bool(det and abs(th - tau) <= tol),
                              loc_err=abs(th - tau) if det else np.nan, bgq=q)
            for k in rec:
                rec[k]["abs_err"] = abs(rec[k]["tau_hat"] - tau)
            recs.append(rec)
        save_records(OUT, "records_alt.csv", recs, {"cell": c, "scen": scen, "level": level,
                                                    "matched_null": int(matched)})
        agg = aggregate(recs, thr)
        R = len(recs)
        for k, v in agg.items():
            m, mode = k.split("|")
            err_det = [x[k]["abs_err"] for x in recs if x[k]["detected"]]
            rows.append(dict(
                part="single", cell=c, scen=scen, level=level, null="matched" if matched else "level-0",
                stat=m, variant=dict(g="raw", gs="studentised", bg="BG-rule")[mode],
                detect_rate=round(v["detect_rate"], 4), detect_se=_se(v["detect_rate"], R),
                power_localised=round(v["power"], 4), power_se=_se(v["power"], R),
                median_abs_err_all=float(np.median([x[k]["abs_err"] for x in recs])),
                median_abs_err_detected=float(np.median(err_det)) if err_det else "",
                threshold=v["threshold"] if mode != "bg" else f"P0={CFG['P0']}", n_reps=R))
        say(f"evaluated {c}")


# ---------------------------------------------------------------- part 2
def _score(cps, taus, n):
    return dict(k=len(cps), exact3=int(len(cps) == 3),
                hits=int(sum(any(abs(c - t) <= CFG["mb_tol"] for c in cps) for t in taus)),
                hausdorff=hausdorff(taus, cps, n), ari=float(rand_index_adj(taus, cps, n)),
                cps=" ".join(map(str, cps)))


def evaluate_mb(rows, info):
    half_L = CFG["L"] // 2
    n = 1800
    taus = CFG["mb_taus"]
    nul = load("mb_null")
    qstar = {}
    for a in CFG["alphas"]:
        qtop = np.array([B.bg_top(s, CFG["l0"])[2] for s in nul[a]])
        qstar[a] = float(np.quantile(qtop, CFG["fpr"]))
        info["mb_null_q_zero"][STAT[a]] = int((qtop == 0).sum())
    info["mb_qstar"] = {STAT[a]: qstar[a] for a in qstar}
    recs = []
    sets = [("null", "mb_null")] + [(lv, f"mb_alt_L{li}") for li, lv in enumerate(SCENARIOS["P1"]["levels"])]
    for lv, s in sets:
        tcs = nul if s == "mb_null" else load(s)
        for a in CFG["alphas"]:
            for rule, q0 in (("calibrated", qstar[a]), ("published", B.Q_PUBLISHED)):
                res = []
                for r, seq in enumerate(tcs[a]):
                    cps = [j + half_L for j in B.bg_segment(seq, q0, CFG["l0"])]
                    d = _score(cps, [] if s == "mb_null" else taus, n)
                    res.append(d)
                    recs.append(dict(level=lv, rep=r, stat=STAT[a], rule=rule, q0=q0, **d))
                if s == "mb_null":
                    info.setdefault("mb_null_fa", {})[f"{STAT[a]}|{rule}"] = float(np.mean([d["k"] > 0 for d in res]))
                    continue
                R = len(res)
                rows.append(dict(
                    part="multi", cell=f"MB_a{lv}", scen="P1", level=lv, null="MB null (level 0)",
                    stat=STAT[a], variant=rule, threshold=q0 if rule == "calibrated" else f"P0={CFG['P0']}",
                    null_fa=info["mb_null_fa"][f"{STAT[a]}|{rule}"],
                    k_correct=float(np.mean([d["exact3"] for d in res])),
                    k_err_mean=float(np.mean([abs(d["k"] - 3) for d in res])),
                    mean_k=float(np.mean([d["k"] for d in res])),
                    hausdorff_median=float(np.median([d["hausdorff"] for d in res])),
                    ari_mean=float(np.mean([d["ari"] for d in res])),
                    mean_hits=float(np.mean([d["hits"] for d in res])), n_reps=R))
    save_records(OUT, "records_mb.csv", recs, {})
    say("evaluated MB")


def evaluate(runtime_compute=None):
    for p in ("records_null.csv.gz", "records_alt.csv.gz", "records_mb.csv.gz"):
        if os.path.exists(os.path.join(OUT, p)):
            os.remove(os.path.join(OUT, p))           # rebuilt from the TC sequences every time
    t0 = time.time()
    rows, info = [], dict(thresholds={}, null_q_zero={}, mb_null_q_zero={})
    evaluate_single(rows, info)
    evaluate_mb(rows, info)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys, restval="")
        wr.writeheader()
        wr.writerows(rows)
    old = {}
    sp = os.path.join(OUT, "summary.json")
    if os.path.exists(sp):
        old = json.load(open(sp)).get("runtime_sec", {})
    rt = dict(old, evaluate=round(time.time() - t0, 1))
    if runtime_compute is not None:
        rt["compute"] = runtime_compute
    json.dump(dict(config=CFG, cells={c: dict(scen=s, level=l, matched_null=m) for c, (s, l, m) in CELLS.items()},
                   implementation="dots/bmctc.py (our reimplementation; choices listed in its docstring)",
                   note=("Table 1 protocol; raw threshold from all 500 nulls, studentised moments from nulls "
                         "0..249 and threshold from 250..499; power = detect and |tau_hat - tau| <= 30; "
                         "BG-rule = top-level BG test at P0 = 0.95 (uncalibrated); MB calibrated rule "
                         "q < q*, q* = 5% quantile of the null top-level q; scenario codes D1-D4 = S1-S4, "
                         "P1 = MB"),
                   runtime_sec=rt, **info, rows=rows),
              open(sp, "w"), indent=1, default=float)
    say(f"evaluated: {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--evaluate", action="store_true", help="evaluate only (TC sequences already stored)")
    a = ap.parse_args()
    os.makedirs(TC, exist_ok=True)
    rtc = None
    if not a.evaluate:
        t0 = time.time()
        compute(a.procs)
        rtc = round(time.time() - t0, 1)
        say(f"compute done {rtc}s")
    evaluate(rtc)


if __name__ == "__main__":
    main()
