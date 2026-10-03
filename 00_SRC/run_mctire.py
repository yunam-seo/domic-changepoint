#!/usr/bin/env python
"""MC-TIRE (Cao et al.) under the article's single-break calibration protocol.

What
    A representation-learning change-point method for multi-channel series: parallel autoencoders
    whose shared features separate cross-channel ("A&S") from channel-specific ("B") information,
    in the time and the frequency domain. The authors' code is used unchanged through
    mctire_adapter.py (settings of their multi-channel experiments; see its docstring). The method
    has no calibration of its own; its peak-merging heuristic returns a set of change points with no
    false-alarm control. Here its dissimilarity curves are calibrated exactly as BMCTC is in
    run_bmctc.py, so that its localized power is comparable with Table 1.

Protocol (identical to run_bmctc.py part 1)
    Cells S1 r=0.35, S2 a=0.7/0.9, S3 tau=0.5, S4 r=0.7/0.85, M1 s=2, G1 nu=2, G2 tau=0.2, G4 a=0.5,
    G6 b=0.3/0.5; n = 600, tau = 300; 500 null and 500 alternative replicates per cell from the
    article's generators and base seed (run_bmctc.draw); S3 and S4 against the matched null.
    Channels: the X and Y components. Candidate grid 60..540 (curve index i <-> time i + 40).
    Statistics: MCTIRE-AS, the cross-channel dissimilarity curve (the branch the method assigns to
    cross-channel changes), and MCTIRE-sum, the sum of the A&S and B curves.
      raw "g":          detect if max over the grid > 95% quantile of the null maxima (all 500 nulls)
      studentized "gs": per-candidate null mean/sd from nulls 0..249, threshold from 250..499
      localized power:  detect and |tau_hat - tau| <= 30
    Seed of the network for replicate r of set s: SeedSequence([20260927, crc32(s), r]).

Run
    python 00_SRC/run_mctire.py --prepare                    # article environment: write the inputs
    <tf-env>/bin/python 00_SRC/run_mctire.py --run --procs 20 # TensorFlow environment: curves
    python 00_SRC/run_mctire.py --evaluate                   # article environment: calibration
Outputs  04_DAOU/EXPERIMENT/mctire/
    inputs/<set>.npz        the series, (reps, n, 2) float64, and inputs_sha256.json
    shards/<set>/<r>.npz    curves of one replicate (A&S, B)
    curves/<set>.npz        all curves of a set, (reps, 521) each
    records_null.csv.gz, records_alt.csv.gz, results.csv, summary.json, run.log
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
OUT = os.path.join(ROOT, "04_DAOU", "EXPERIMENT", "mctire")
CFG = dict(n=600, tau=300, w=60, tol=30, fpr=0.05, reps=500, ws=40, seed=20260927)
CELLS = ["S1_r0.35", "S2_a0.7", "S2_a0.9", "S3_tau0.5", "S4_r0.7", "S4_r0.85", "M1_s2", "G1_nu2",
         "G2_tau0.2", "G4_a0.5", "G6_b0.3", "G6_b0.5"]
STATS = ["MCTIRE-AS", "MCTIRE-sum"]


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(os.path.join(OUT, "run.log"), "a") as f:
        f.write(line + "\n")


def sets():
    import run_bmctc as RB
    out = []
    for c in CELLS:
        if RB.null_set(c) not in out:
            out.append(RB.null_set(c))
    return out + [f"alt_{c}" for c in CELLS]


def prepare():
    import run_bmctc as RB
    os.makedirs(os.path.join(OUT, "inputs"), exist_ok=True)
    digests = {}
    for s in sets():
        Z = np.stack([np.column_stack(RB.draw(s, r)) for r in range(CFG["reps"])])
        p = os.path.join(OUT, "inputs", f"{s}.npz")
        np.savez_compressed(p, Z=Z)
        digests[s] = hashlib.sha256(Z.tobytes()).hexdigest()
        say(f"prepared {s} {Z.shape}")
    json.dump(dict(config=CFG, sets=sets(), sha256_of_Z=digests),
              open(os.path.join(OUT, "inputs_sha256.json"), "w"), indent=1)


def seed_of(s, r):
    return int(np.random.SeedSequence([CFG["seed"], zlib.crc32(s.encode()), r]).generate_state(1)[0] % (2 ** 31))


_Z = {}


def _job(a):
    s, r = a
    p = os.path.join(OUT, "shards", s, f"{r:03d}.npz")
    if os.path.exists(p):
        return
    import mctire_adapter as M
    if s not in _Z:
        _Z[s] = np.load(os.path.join(OUT, "inputs", f"{s}.npz"))["Z"]
    t0 = time.time()
    a_s, b = M.curves(_Z[s][r], seed_of(s, r))
    tmp = p[:-4] + ".tmp.npz"
    np.savez(tmp, AS=a_s, B=b)
    os.replace(tmp, p)
    return time.time() - t0


def run(procs):
    S = json.load(open(os.path.join(OUT, "inputs_sha256.json")))["sets"]
    for s in S:
        os.makedirs(os.path.join(OUT, "shards", s), exist_ok=True)
    jobs = [(s, r) for r in range(CFG["reps"]) for s in S]
    say(f"run: {len(jobs)} series, {procs} processes")
    done = 0
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(procs, maxtasksperchild=50) as pool:
        for _ in pool.imap_unordered(_job, jobs, chunksize=1):
            done += 1
            if done % 200 == 0:
                say(f"{done}/{len(jobs)}")
    collect(S)


def collect(S):
    os.makedirs(os.path.join(OUT, "curves"), exist_ok=True)
    for s in S:
        fs = [os.path.join(OUT, "shards", s, f"{r:03d}.npz") for r in range(CFG["reps"])]
        if not all(os.path.exists(f) for f in fs):
            say(f"{s}: incomplete, not collected")
            continue
        A = [np.load(f) for f in fs]
        np.savez_compressed(os.path.join(OUT, "curves", f"{s}.npz"),
                            AS=np.stack([a["AS"] for a in A]), B=np.stack([a["B"] for a in A]))
    say("collected")


def _curves(s):
    z = np.load(os.path.join(OUT, "curves", f"{s}.npz"))
    lo = CFG["w"] - CFG["ws"]
    hi = CFG["n"] - CFG["w"] - CFG["ws"] + 1
    return {"MCTIRE-AS": z["AS"][:, lo:hi], "MCTIRE-sum": (z["AS"] + z["B"])[:, lo:hi]}


def evaluate():
    import csv
    import run_bmctc as RB
    from dots.extras import studentize
    from dots.evaluate import summarize_alt, aggregate
    from dots.persist import save_records
    n, w, tol, fpr, tau = CFG["n"], CFG["w"], CFG["tol"], CFG["fpr"], CFG["tau"]
    grid = np.arange(w, n - w + 1)
    half = CFG["reps"] // 2
    rows, thr_all, nulls = [], {}, {}
    for c in CELLS:
        s = RB.null_set(c)
        if s in nulls:
            continue
        cur = _curves(s)
        assert all(v.shape == (CFG["reps"], len(grid)) for v in cur.values())
        mu0 = {k: v[:half].mean(0) for k, v in cur.items()}
        sd0 = {k: v[:half].std(0) for k, v in cur.items()}
        thr = {}
        for k, v in cur.items():
            thr[f"{k}|g"] = float(np.quantile(v.max(1), 1 - fpr))
            thr[f"{k}|gs"] = float(np.quantile([np.max(studentize(x, mu0[k], sd0[k])) for x in v[half:]], 1 - fpr))
        nulls[s] = (mu0, sd0, thr)
        thr_all[s] = thr
        save_records(OUT, "records_null.csv",
                     [{f"max|{k}": float(v[r].max()) for k, v in cur.items()} |
                      {f"tau_hat|{k}": int(grid[np.argmax(v[r])]) for k, v in cur.items()}
                      for r in range(CFG["reps"])], {"set": s})
    for c in CELLS:
        mu0, sd0, thr = nulls[RB.null_set(c)]
        cur = _curves(f"alt_{c}")
        recs = []
        for r in range(CFG["reps"]):
            st = {}
            for k, v in cur.items():
                st[f"{k}|g"] = v[r]
                st[f"{k}|gs"] = studentize(v[r], mu0[k], sd0[k])
            rec = summarize_alt(st, grid, tau, w, n, thr, tol)
            for k in rec:
                rec[k]["abs_err"] = abs(rec[k]["tau_hat"] - tau)
            recs.append(rec)
        scen, level, matched = RB.CELLS[c]
        save_records(OUT, "records_alt.csv", recs, {"cell": c, "scen": scen, "level": level,
                                                    "matched_null": int(matched)})
        R = len(recs)
        for k, v in aggregate(recs, thr).items():
            m, mode = k.split("|")
            err_det = [x[k]["abs_err"] for x in recs if x[k]["detected"]]
            rows.append(dict(cell=c, scen=scen, level=level, null="matched" if matched else "level-0", stat=m,
                             variant=dict(g="raw", gs="studentised")[mode],
                             detect_rate=round(v["detect_rate"], 4),
                             detect_se=round(float(np.sqrt(v["detect_rate"] * (1 - v["detect_rate"]) / R)), 4),
                             power_localised=round(v["power"], 4),
                             power_se=round(float(np.sqrt(v["power"] * (1 - v["power"]) / R)), 4),
                             median_abs_err_all=float(np.median([x[k]["abs_err"] for x in recs])),
                             median_abs_err_detected=float(np.median(err_det)) if err_det else "",
                             threshold=v["threshold"], n_reps=R))
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    json.dump(dict(config=CFG, cells={c: dict(zip(("scen", "level", "matched_null"), RB.CELLS[c])) for c in CELLS},
                   thresholds=thr_all, method_settings="mctire_adapter.py", repository_commit="f538a86"),
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    for r in rows:
        print(f"{r['cell']:10s} {r['stat']:11s} {r['variant']:12s} detect {r['detect_rate']:.3f}  "
              f"localized {r['power_localised']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.prepare:
        prepare()
    if a.run:
        run(a.procs)
    if a.collect:
        collect(json.load(open(os.path.join(OUT, "inputs_sha256.json")))["sets"])
    if a.evaluate:
        evaluate()


if __name__ == "__main__":
    main()
