#!/usr/bin/env bash
# run_perf_metrics_prefetch.sh
# Collect instructions/cycles + L1D/L2/LLC cache metrics across
# prefetch distances on the Intel hybrid CPU.
#
# Usage:
#   sudo ./run_perf_metrics_prefetch.sh
#   sudo ./run_perf_metrics_prefetch.sh "1 2 4 8 16 32 64 128" 1024 5

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

# CPU 0 is a P-core on this machine.
PIN_CPU=${PIN_CPU:-0}

# Explicitly use the P-core PMU for all events.
EVENTS="cpu_core/instructions/,cpu_core/cycles/,cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/LLC-loads/,cpu_core/LLC-load-misses/,cpu_core/l2_request.all/,cpu_core/l2_request.miss/"

echo "Using CPU: $PIN_CPU"
echo "Using cpu_core PMU"
echo "Collecting L1D + L2 + LLC + instructions + cycles"

for D in "${DISTANCES[@]}"; do

    echo "----------------------------------------"
    echo "Prefetch distance: $D"

    g++ -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -I../include \
        -DPARAM_PREFETCH_DISTANCE="$D" \
        bench_perf_workload.cpp matmul_prefetch_param.cpp \
        ../src/matmul_naive.cpp ../src/matmul_simd.cpp \
        ../src/matmul_prefetch.cpp ../src/matmul_optimized.cpp \
        -o /tmp/bench_perf_d

    RAWFILE="results/perf_raw_prefetch/dist${D}_${SIZE}.csv"

    sudo perf stat \
        -x, \
        -C "$PIN_CPU" \
        -e "$EVENTS" \
        -o "$RAWFILE" \
        -- \
        /tmp/bench_perf_d prefetch_param "$SIZE" "$SIZE" "$SIZE" "$ITERS"

    echo "distance=$D done"

done

python3 parse_perf_csv.py \
    results/perf_raw_prefetch \
    results/perf_metrics_prefetch.csv

echo "----------------------------------------"
echo "wrote results/perf_metrics_prefetch.csv"