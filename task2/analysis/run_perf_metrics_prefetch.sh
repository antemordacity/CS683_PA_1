#!/usr/bin/env bash
# run_perf_metrics_prefetch.sh  Collect L1D / LLC misses + instructions/cycles via
# `perf stat` for matmul_prefetch_param across prefetch distances, at a fixed
# (large-enough-to-matter) matrix size. This is the data behind "identify the optimal
# prefetch distance / cache fill level" -- the distance where LLC/L1D misses bottom
# out is your answer, and it should line up with the speedup peak from
# run_prefetch_distance_sweep.sh.
#
# REQUIRES: `perf` installed and perf_event access allowed (see
# run_perf_metrics_size.sh for the permission note).
#
# Usage:
#   ./run_perf_metrics_prefetch.sh                       # default distances, size 1024
#   ./run_perf_metrics_prefetch.sh "1 2 4 8 16 32 64" 2048 5
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/perf_raw_prefetch

if ! command -v perf >/dev/null 2>&1; then
    echo "ERROR: 'perf' not found. Install with:"
    echo "  sudo apt-get install linux-tools-common linux-tools-\$(uname -r) linux-tools-generic"
    exit 1
fi

DISTANCES=(${1:-1 2 4 8 16 32 64 128})
SIZE=${2:-1024}
ITERS=${3:-5}

EVENTS="instructions,cycles,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses"
if perf list 2>/dev/null | grep -q "l2_rqsts.miss"; then
    EVENTS="$EVENTS,l2_rqsts.miss,l2_rqsts.references"
fi

for D in "${DISTANCES[@]}"; do
    g++ -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -I../include \
        -DPARAM_PREFETCH_DISTANCE="$D" \
        bench_perf_workload.cpp matmul_prefetch_param.cpp \
        ../src/matmul_naive.cpp ../src/matmul_simd.cpp ../src/matmul_prefetch.cpp \
        ../src/matmul_optimized.cpp \
        -o /tmp/bench_perf_d

    RAWFILE="results/perf_raw_prefetch/dist${D}_${SIZE}.csv"
    perf stat -x, -e "$EVENTS" -o "$RAWFILE" -- \
        /tmp/bench_perf_d prefetch_param "$SIZE" "$SIZE" "$SIZE" "$ITERS"
    echo "distance=$D done"
done

python3 parse_perf_csv.py results/perf_raw_prefetch results/perf_metrics_prefetch.csv
