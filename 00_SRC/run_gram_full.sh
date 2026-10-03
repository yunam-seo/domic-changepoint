#!/usr/bin/env bash
# Full-segment Gram form of DOMI (every row of each segment, --msub 0): the Gram-form entries of
# Table 1, upper block (T1a, T1b), the further levels of Figure 2 and Section 6.1 (T1c-T1e),
# Table 1, lower block (G1-G6), and Supplementary Table B.3.
#
# Run:  bash 00_SRC/run_gram_full.sh            (PROCS=<n> to set the worker count)
# Writes 04_DAOU/EXPERIMENT/matmi_baseline_full_T1{a,b,c,d,e}/ and ng_suite_gfull_G{1..6}/.
set -eu
cd "$(dirname "$0")/.."
PROCS=${PROCS:-$(nproc)}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

COMMON="--reps 500 --procs $PROCS --D 8 --msub 0"

# Table 1, upper block (T1a, T1b) and the further levels of Figure 2 and Section 6.1 (T1c-T1e)
python 00_SRC/run_matmi_baseline.py --scenarios D1,D2,M1 $COMMON \
    --levels "D1:0.35;D2:0.7,0.9;M1:2.0" --tag full_T1a
python 00_SRC/run_matmi_baseline.py --scenarios D3,D4 --matched $COMMON \
    --levels "D3:0.5;D4:0.7,0.85" --tag full_T1b
python 00_SRC/run_matmi_baseline.py --scenarios D1 $COMMON \
    --levels "D1:0.2,0.5" --tag full_T1c
python 00_SRC/run_matmi_baseline.py --scenarios D2 $COMMON \
    --levels "D2:0.5" --tag full_T1d
python 00_SRC/run_matmi_baseline.py --scenarios D3,D4 --matched $COMMON \
    --levels "D3:0.3,0.7;D4:0.5" --tag full_T1e

# Table 1, lower block (non-Gaussian suite)
for k in 1 2 3 4 5 6; do
    python 00_SRC/run_ng_suite.py --scenarios G$k --reps 500 --procs "$PROCS" --no-deps \
        --msub 0 --tag gfull_G$k
done
