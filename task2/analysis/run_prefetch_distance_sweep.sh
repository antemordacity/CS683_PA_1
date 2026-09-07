#!/usr/bin/env bash
# run_prefetch_distance_sweep.sh  ANALYSIS TASK 2: speedup vs. prefetch distance.
#
# Recompiles matmul_prefetch_param.cpp once per distance (-DPARAM_PREFETCH_DISTANCE=<d>)
# and times it at each of a few sizes, so you get a (distance x size) grid.
#
# Usage:
#   ./run_prefetch_distance_sweep.sh                  # default distances, default sizes
#   ./run_prefetch_distance_sweep.sh "1 2 4 8 16 32 64 128" "512 1024 2048"
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

DISTANCES=(${1:-1 2 4 8 16 32 64 128})
SIZES=${2:-512 1024 2048}

OUT=results/prefetch_distance.csv
echo "distance,M,N,K,time_ms,gflops,speedup,rel_err" > "$OUT"

for D in "${DISTANCES[@]}"; do
    g++ -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -I../include \
        -DPARAM_PREFETCH_DISTANCE="$D" \
        matmul_prefetch_param.cpp bench_prefetch_distance.cpp ../src/matmul_naive.cpp \
        -o /tmp/bench_pd
    /tmp/bench_pd "$D" $SIZES | tail -n +2 >> "$OUT"
    echo "distance=$D done"
done
echo "wrote $OUT"
