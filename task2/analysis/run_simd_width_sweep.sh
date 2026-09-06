#!/usr/bin/env bash
# run_simd_width_sweep.sh  ANALYSIS TASK 4: speedup vs. SIMD width (128/256/512-bit).
#
# Compiles matmul_simd_sse.cpp (-msse4.1, no AVX/FMA), ../src/matmul_simd.cpp
# (-mavx2 -mfma, your real graded AVX2 kernel), and -- if this CPU supports it --
# matmul_simd_avx512.cpp (-mavx512f -mfma) as SEPARATE object files with their own
# flags, then links them into one binary so all three widths are measured in the same
# process on the same random data.
#
# Usage:
#   ./run_simd_width_sweep.sh                    # default sizes
#   ./run_simd_width_sweep.sh 128 256 512 1024   # custom sizes
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

BASEFLAGS="-std=c++17 -O2 -fno-tree-vectorize -I../include"

g++ $BASEFLAGS -msse4.1 -c matmul_simd_sse.cpp -o /tmp/sse.o
g++ $BASEFLAGS -mavx2 -mfma -c ../src/matmul_simd.cpp -o /tmp/avx2.o
g++ $BASEFLAGS -c ../src/matmul_naive.cpp -o /tmp/naive.o

if grep -qi avx512f /proc/cpuinfo; then
    echo "AVX-512F supported on this CPU -- including the 512-bit variant."
    g++ $BASEFLAGS -mavx512f -mfma -c matmul_simd_avx512.cpp -o /tmp/avx512.o
    g++ $BASEFLAGS -mavx2 -DHAVE_AVX512 -c bench_simd_width.cpp -o /tmp/bench.o
    g++ /tmp/sse.o /tmp/avx2.o /tmp/avx512.o /tmp/naive.o /tmp/bench.o -o /tmp/bench_simd_width
else
    echo "AVX-512F NOT supported on this CPU -- skipping the 512-bit variant."
    echo "(results/simd_width.csv will only have 128/256-bit rows; note this in your report.)"
    g++ $BASEFLAGS -mavx2 -c bench_simd_width.cpp -o /tmp/bench.o
    g++ /tmp/sse.o /tmp/avx2.o /tmp/naive.o /tmp/bench.o -o /tmp/bench_simd_width
fi

OUT=results/simd_width.csv
/tmp/bench_simd_width "$@" | tee "$OUT"
echo "wrote $OUT"
