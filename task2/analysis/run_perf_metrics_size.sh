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

# Pin to one logical CPU so a single-threaded run doesn't get bounced between
# core types on a hybrid (P-core/E-core) chip mid-measurement. Override with
# PIN_CPU=<n> if you want a specific P-core; check `lscpu -e` to see which
# logical CPU numbers map to which core type on your machine.
PIN_CPU=${PIN_CPU:-0}

EVENTS="cpu_core/instructions/,cpu_core/cycles/,cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/LLC-loads/,cpu_core/LLC-load-misses/,cpu_core/l2_request.all/,cpu_core/l2_request.miss/"

echo "using cpu_core L2 events"

g++ -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -I../include \
    bench_perf_workload.cpp matmul_prefetch_param.cpp \
    ../src/matmul_naive.cpp ../src/matmul_simd.cpp ../src/matmul_prefetch.cpp \
    ../src/matmul_optimized.cpp \
    -o /tmp/bench_perf

for n in "${SIZES[@]}"; do
    for stage in "${STAGES[@]}"; do
        RAWFILE="results/perf_raw_size/${stage}_${n}.csv"
        perf stat -x, -C "$PIN_CPU" -e "$EVENTS" -o "$RAWFILE" -- \
            /tmp/bench_perf "$stage" "$n" "$n" "$n" "$ITERS"
        echo "stage=$stage size=$n done"
    done
done

python3 parse_perf_csv.py results/perf_raw_size results/perf_metrics_size.csv