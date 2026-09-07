#!/usr/bin/env bash
# run_size_sweep.sh  ANALYSIS TASK 1: speedup of simd/prefetch/optimized vs. naive
# across matrix sizes small enough to fit in cache through well beyond it.
#
# Usage:
#   ./run_size_sweep.sh                    # default sizes: 128 .. 4096
#   ./run_size_sweep.sh 128 256 512 1024   # custom size list
#
# NOTE: naive is O(n^3) scalar, so the largest sizes take a while (a single 4096^3
# call is on the order of a minute or more depending on the machine); this is
# expected, not a bug. Shrink the size list if you want a faster first pass.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

g++ -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -I../include \
    bench_size.cpp ../src/matmul_naive.cpp ../src/matmul_simd.cpp \
    ../src/matmul_prefetch.cpp ../src/matmul_optimized.cpp \
    -o /tmp/bench_size

OUT=results/size_sweep.csv
/tmp/bench_size "$@" | tee "$OUT"
echo "wrote $OUT"
