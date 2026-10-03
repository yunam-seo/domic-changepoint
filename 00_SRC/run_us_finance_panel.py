#!/usr/bin/env python
"""US financial panel under a fixed protocol (Section 6.8; Supplementary Section B.22).

Pairs: SPX-GLD (primary statistic DOMI), SPX-TNX and SPX-DXY (primary statistic: the combined statistic of
Section 4.5). Daily closes, returns on each series' own trading days (log-returns; first differences for the
yield), aligned on the dates common to both, 2011-08-22 .. 2026-08-20.

Per pair, one whole-record single-break test (candidates n/10 .. 9n/10), Algorithm 1 with symmetric
studentization over all K + 1 curves (the test of run_e6_fullscan.scan_test), calibrated by block permutation
at the block length recommended by the exchangeability diagnostic of Section 4.3 (L = 20, K = 199, seed 0), or
by pair permutation if the diagnostic does not reject.
  main          K = 9999 from permutation seed 20260930; the K replicas are drawn in chunks whose generators
                are numpy SeedSequence(20260930).spawn(n_chunks), a fixed function of the seed
  sensitivity   20 further draws of K = 999, seeds 1..20 (reported, not used for decisions)
Benjamini-Hochberg at q = 0.10 over the three primary p-values.

Run:    python 00_SRC/run_us_finance_panel.py --part diag
        python 00_SRC/run_us_finance_panel.py --part main --procs 24
        python 00_SRC/run_us_finance_panel.py --part sens --procs 24
        python 00_SRC/run_us_finance_panel.py --part summary
Writes 04_DAOU/EXPERIMENT/us_panel/{config.json, diag.json, main_<pair>.json, records_T_main_<pair>.csv.gz,
       sens.jsonl, summary.json}
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from dots.perm import ge  # noqa: E402
from dots.segtests import _perm_index  # noqa: E402
import run_e6_fullscan as FS  # noqa: E402

# set US_PANEL_OUT to write a re-run elsewhere; config.json records the SHA-256 of the inputs, so a re-run
# on differently revised downloads stops rather than mixing with the stored outputs
OUT = os.environ.get("US_PANEL_OUT", os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "us_panel"))
FILES = {"SPX": (os.path.join(ROOT, "01_ORG", "FINANCE", "yahoo_gspc.csv"), "logret"),
         "TNX": (os.path.join(ROOT, "01_ORG", "FINANCE", "yahoo_tnx.csv"), "diff"),
         "GLD": (os.path.join(ROOT, "01_ORG", "FINANCE", "yahoo_gld.csv"), "logret"),
         "DXY": (os.path.join(ROOT, "01_ORG", "FINANCE", "yahoo_dxy.csv"), "logret")}
PAIRS = {"SPX-GLD": ("SPX", "GLD", "DOMI-diff"), "SPX-TNX": ("SPX", "TNX", "combined"),
         "SPX-DXY": ("SPX", "DXY", "combined")}
CFG = dict(first="20110822", last="20260820", D=8, w_frac=0.1, diag_L=20, diag_K=199, diag_seed=0,
           K_main=9999, seed_main=20260930, n_chunks=48, K_sens=999, seeds_sens=list(range(1, 21)),
           bh_q=0.10, pairs={k: dict(x=v[0], y=v[1], primary=v[2]) for k, v in PAIRS.items()})
STATS = FS.STATS


def load(code):
    path, kind = FILES[code]
    s = {r["date"]: float(r["close"]) for r in csv.DictReader(open(path)) if float(r["close"]) > 0}
    s = {d: v for d, v in s.items() if CFG["first"] <= d <= CFG["last"]}
    return FS.returns(s, kind)


def pair_data(name):
    a, b, _ = PAIRS[name]
    (da, ra), (db, rb) = load(a), load(b)
    ia, ib = {d: i for i, d in enumerate(da)}, {d: i for i, d in enumerate(db)}
    common = [d for d in da if d in ib]
    return common, ra[[ia[d] for d in common]], rb[[ib[d] for d in common]]


def _chunk(a):
    name, block, seed_state, k = a
    _, x, y = pair_data(name)
    w = int(CFG["w_frac"] * len(x))
    rng = np.random.default_rng(seed_state)
    out = {m: np.empty((k, len(x) - 2 * w + 1)) for m in STATS}
    for i in range(k):
        idx = _perm_index(len(x), rng, block)
        _, c = FS.curves(x[idx], y[idx], w)
        for m in STATS:
            out[m][i] = c[m]
    return out


def studentised_test(obs, reps, grid):
    Tall, tau = {}, {}
    for m in STATS:
        A = np.vstack([obs[m][None, :], reps[m]])
        S = (A - A.mean(0)) / (A.std(0) + 1e-12)
        Tall[m] = S.max(1)
        tau[m] = int(grid[int(np.argmax(S[0]))])
    Tall["combined"] = np.maximum(Tall["DOMI-diff"], Tall["Spearman-diff"])
    tau["combined"] = tau["DOMI-diff"] if Tall["DOMI-diff"][0] >= Tall["Spearman-diff"][0] else tau["Spearman-diff"]
    K = len(Tall["DOMI-diff"]) - 1
    p = {m: (1 + int(sum(ge(t, T[0]) for t in T[1:]))) / (K + 1) for m, T in Tall.items()}
    return p, tau, Tall


def part_diag():
    from run_exch_diag import exch_diag
    out = {}
    for name in PAIRS:
        dates, x, y = pair_data(name)
        d = exch_diag(x, y, L=CFG["diag_L"], K=CFG["diag_K"], seed=CFG["diag_seed"])
        b = None if d["decision"] == "pair" else int(d["b_hat"])
        out[name] = dict(n=len(x), first=dates[0], last=dates[-1], decision=d["decision"], block=b,
                         n_blocks=(len(x) // b) if b else len(x), pvalues=d["pvalues"])
        print(name, out[name], flush=True)
    json.dump(out, open(os.path.join(OUT, "diag.json"), "w"), indent=1)


def part_main(procs):
    diag = json.load(open(os.path.join(OUT, "diag.json")))
    ss = np.random.SeedSequence(CFG["seed_main"]).spawn(CFG["n_chunks"])
    sizes = [CFG["K_main"] // CFG["n_chunks"] + (1 if i < CFG["K_main"] % CFG["n_chunks"] else 0)
             for i in range(CFG["n_chunks"])]
    for name in PAIRS:
        path = os.path.join(OUT, f"main_{name}.json")
        if os.path.exists(path):
            continue
        t0 = time.time()
        dates, x, y = pair_data(name)
        w = int(CFG["w_frac"] * len(x))
        grid, obs = FS.curves(x, y, w)
        with Pool(procs) as pool:
            parts = pool.map(_chunk, [(name, diag[name]["block"], s, k) for s, k in zip(ss, sizes)])
        reps = {m: np.vstack([p_[m] for p_ in parts]) for m in STATS}
        p, tau, Tall = studentised_test(obs, reps, grid)
        res = dict(pair=name, n=len(x), w=w, block=diag[name]["block"], K=CFG["K_main"], p=p,
                   tau_date={m: dates[t] for m, t in tau.items()}, seconds=round(time.time() - t0, 1))
        with gzip.open(os.path.join(OUT, f"records_T_main_{name}.csv.gz"), "wt", newline="") as f:
            wr = csv.writer(f, lineterminator="\n")
            wr.writerow(["stat", "replica", "T"])
            for m, T in Tall.items():
                wr.writerows((m, i, repr(float(t))) for i, t in enumerate(T))
        json.dump(res, open(path, "w"), indent=1)
        print(name, {m: round(v, 4) for m, v in p.items()}, res["tau_date"], f"{res['seconds']:.0f}s", flush=True)


def _sens_job(a):
    name, block, seed = a
    dates, x, y = pair_data(name)
    w = int(CFG["w_frac"] * len(x))
    p, tau, _ = FS.scan_test(x, y, w, CFG["K_sens"], block, seed)
    return dict(pair=name, seed=seed, p=p, tau_date={m: dates[t] for m, t in tau.items()})


def part_sens(procs):
    diag = json.load(open(os.path.join(OUT, "diag.json")))
    path = os.path.join(OUT, "sens.jsonl")
    done = {(r["pair"], r["seed"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    jobs = [(n, diag[n]["block"], s) for n in PAIRS for s in CFG["seeds_sens"] if (n, s) not in done]
    with Pool(procs) as pool:
        for rec in pool.imap_unordered(_sens_job, jobs):
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec["pair"], rec["seed"], {m: round(v, 3) for m, v in rec["p"].items()}, flush=True)


def part_summary():
    diag = json.load(open(os.path.join(OUT, "diag.json")))
    main = {n: json.load(open(os.path.join(OUT, f"main_{n}.json"))) for n in PAIRS}
    prim = {n: main[n]["p"][PAIRS[n][2]] for n in PAIRS}
    order = sorted(PAIRS, key=lambda n: prim[n])
    m, q, run = len(order), {}, 1.0
    for i in range(m, 0, -1):
        run = min(run, prim[order[i - 1]] * m / i)
        q[order[i - 1]] = run
    sens = [json.loads(l) for l in open(os.path.join(OUT, "sens.jsonl"))] if os.path.exists(os.path.join(OUT, "sens.jsonl")) else []
    out = {}
    for n in PAIRS:
        rr = [r for r in sens if r["pair"] == n]
        out[n] = dict(diag=diag[n], primary_statistic=PAIRS[n][2], p=main[n]["p"], tau_date=main[n]["tau_date"],
                      primary_p=prim[n], bh_q=q[n], survives_bh=q[n] <= CFG["bh_q"],
                      sensitivity={s: dict(median=float(np.median([r["p"][s] for r in rr])),
                                           min=float(min(r["p"][s] for r in rr)), max=float(max(r["p"][s] for r in rr)),
                                           n_draws=len(rr)) for s in rr[0]["p"]} if rr else None)
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["diag", "main", "sens", "summary"])
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = dict(CFG, data_sha256={k: hashlib.sha256(open(v[0], "rb").read()).hexdigest() for k, v in FILES.items()})
    cp = os.path.join(OUT, "config.json")
    if not os.path.exists(cp):
        json.dump(cfg, open(cp, "w"), indent=1)
    elif json.load(open(cp)) != json.loads(json.dumps(cfg)):
        sys.exit("config.json differs from the current configuration or inputs: use a new output folder")
    {"diag": part_diag, "main": lambda: part_main(a.procs), "sens": lambda: part_sens(a.procs),
     "summary": part_summary}[a.part]()


if __name__ == "__main__":
    main()
