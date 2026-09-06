// bench_common.h  ANALYSIS-ONLY helper shared by the bench_*.cpp drivers in this
// directory. Not part of the graded submission; not included by src/*.cpp.
//
// Wraps one call of a MatMulFn on a fresh random M,N,K workload: times it with the
// same pa1::time_median_ms median-of-reps approach main.cpp uses, and checks
// correctness against a reference implementation with the same relative-tolerance
// metric main.cpp uses (pa1::max_abs / pa1::max_abs_diff), so numbers here are
// directly comparable with what `./bin/matmul` reports.
#pragma once

#include "matmul.h"
#include "timer.h"
#include "utils.h"

namespace bench {

struct Result {
    double ms;       // median wall-clock time for one MxNxK call
    double gflops;
    double rel_err;  // vs the supplied reference implementation
};

inline Result run_and_time(MatMulFn fn, MatMulFn ref_fn, int M, int N, int K,
                            unsigned seed, int warmup, int reps) {
    float* A = pa1::alloc_floats(static_cast<std::size_t>(M) * K);
    float* B = pa1::alloc_floats(static_cast<std::size_t>(N) * K);
    float* C = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
    float* ref = pa1::alloc_floats(static_cast<std::size_t>(M) * N);

    pa1::fill_random(A, static_cast<std::size_t>(M) * K, seed);
    pa1::fill_random(B, static_cast<std::size_t>(N) * K, seed + 1u);

    const int lda = K, ldb = K, ldc = N;
    ref_fn(A, B, ref, M, N, K, lda, ldb, ldc);

    auto run = [&]() { fn(A, B, C, M, N, K, lda, ldb, ldc); };
    run();  // one untimed call to produce the result we check

    const float ref_mag = pa1::max_abs(ref, static_cast<std::size_t>(M) * N);
    const float abs_err = pa1::max_abs_diff(C, ref, static_cast<std::size_t>(M) * N);
    const float rel_err = abs_err / (ref_mag + 1e-30f);

    const double ms = pa1::time_median_ms(run, warmup, reps);
    const double gflops = pa1::matmul_flops(M, N, K) / (ms * 1e6);

    pa1::free_floats(A);
    pa1::free_floats(B);
    pa1::free_floats(C);
    pa1::free_floats(ref);

    return {ms, gflops, static_cast<double>(rel_err)};
}

// Fewer timed reps for bigger (slower) workloads so the size sweep finishes in a
// reasonable time; naive is O(n^3) scalar so this matters most at the top end.
inline int reps_for_size(int n) {
    if (n <= 512) return 5;
    if (n <= 1024) return 3;
    return 1;
}

}  // namespace bench
