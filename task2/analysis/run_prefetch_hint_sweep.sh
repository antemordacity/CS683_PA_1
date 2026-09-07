#!/usr/bin/env bash
# run_prefetch_hint_sweep.sh  ANALYSIS TASK 3: speedup vs. prefetch locality hint.
#
# Recompiles matmul_prefetch_param.cpp once per hint (-DPARAM_PREFETCH_HINT=...) at a
# fixed distance -- pass the distance your run_prefetch_distance_sweep.sh run found
# best, so the two analyses agree with each other.
#
# Usage:
#   ./run_prefetch_hint_sweep.sh <best_distance> ["512 1024 2048"]
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <best_distance_from_run_prefetch_distance_sweep> [sizes]"
    exit 1
fi
DISTANCE=$1
SIZES=${2:-512 1024 2048}

HINTS=(_MM_HINT_T0 _MM_HINT_T1 _MM_HINT_T2 _MM_HINT_NTA)

OUT=results/prefetch_hint.csv
echo "hint,distance,M,N,K,time_ms,gflops,speedup,rel_err" > "$OUT"

for H in "${HINTS[@]}"; do
    g++ -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -I../include \
        -DPARAM_PREFETCH_DISTANCE="$DISTANCE" -DPARAM_PREFETCH_HINT="$H" \
        matmul_prefetch_param.cpp bench_prefetch_hint.cpp ../src/matmul_naive.cpp \
        -o /tmp/bench_ph
    /tmp/bench_ph "$H" "$DISTANCE" $SIZES | tail -n +2 >> "$OUT"
    echo "hint=$H done"
done
echo "wrote $OUT"
