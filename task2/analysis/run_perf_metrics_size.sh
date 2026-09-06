#!/usr/bin/env bash
# run_perf_metrics_size.sh  Collect L1D / LLC misses + instructions/cycles via
# `perf stat` for each stage at each matrix size (Task: "Analyze CPU Metrics" /
# "Analyze Instruction Count", the matrix-size half).
#
# REQUIRES: `perf` installed and perf_event access allowed. If you see "Permission
# denied", either run this script with sudo, or:
#   sudo sysctl -w kernel.perf_event_paranoid=-1
#
# Usage:
#   ./run_perf_metrics_size.sh                      # default sizes, default iters
#   ./run_perf_metrics_size.sh "256 512 1024 2048" 5
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/perf_raw_size

if ! command -v perf >/dev/null 2>&1; then
    echo "ERROR: 'perf' not found. Install with:"
    echo "  sudo apt-get install linux-tools-common linux-tools-\$(uname -r) linux-tools-generic"
    exit 1
fi

SIZES=(${1:-256 512 1024 2048})
ITERS=${2:-5}
STAGES=(naive simd prefetch optimized)

EVENTS="instructions,cycles,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses"
if perf list 2>/dev/null | grep -q "l2_rqsts.miss"; then
    EVENTS="$EVENTS,l2_rqsts.miss,l2_rqsts.references"
    echo "using raw event l2_rqsts.miss for L2 misses"
else
    echo "note: raw event 'l2_rqsts.miss' not available on this CPU/perf build;"
    echo "      l2_misses/l2_refs will be blank in the aggregated CSV."
fi

g++ -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -I../include \
    bench_perf_workload.cpp matmul_prefetch_param.cpp \
    ../src/matmul_naive.cpp ../src/matmul_simd.cpp ../src/matmul_prefetch.cpp \
    ../src/matmul_optimized.cpp \
    -o /tmp/bench_perf

for n in "${SIZES[@]}"; do
    for stage in "${STAGES[@]}"; do
        RAWFILE="results/perf_raw_size/${stage}_${n}.csv"
        perf stat -x, -e "$EVENTS" -o "$RAWFILE" -- \
            /tmp/bench_perf "$stage" "$n" "$n" "$n" "$ITERS"
        echo "stage=$stage size=$n done"
    done
done

python3 parse_perf_csv.py results/perf_raw_size results/perf_metrics_size.csv
