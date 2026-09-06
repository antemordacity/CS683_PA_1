// bench_perf_workload.cpp  ANALYSIS-ONLY. Minimal harness meant to be wrapped by
// `perf stat` / `perf record`, not run directly for timing (use bench_size.cpp /
// main.cpp for that -- this has no internal timer so perf attaches to exactly the
// compute you want characterized, nothing else).
//
// Allocates buffers and fills them ONCE, then calls the chosen stage `iters` times
// back-to-back with no printing/timing in between, so perf's counted region is pure
// matmul work (plus loop overhead, negligible at any iters >= a handful).
//
// Usage: ./bench_perf_workload <stage> <M> <N> <K> <iters> [seed]
//   stage: naive | simd | prefetch | prefetch_param | optimized
//   (prefetch_param resolves to matmul_prefetch_param.cpp, linked in for the
//    perf-vs-prefetch-parameter sweep; requires it to be compiled into this binary)
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "matmul.h"
#include "utils.h"

extern void matmul_prefetch_param(const float*, const float*, float*, int, int, int,
                                   int, int, int);

static MatMulFn resolve(const char* name) {
    if (!std::strcmp(name, "naive")) return matmul_naive;
    if (!std::strcmp(name, "simd")) return matmul_simd;
    if (!std::strcmp(name, "prefetch")) return matmul_prefetch;
    if (!std::strcmp(name, "prefetch_param")) return matmul_prefetch_param;
    if (!std::strcmp(name, "optimized")) return matmul_optimized;
    return nullptr;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s <stage> <M> <N> <K> <iters> [seed]\n",
                      argv[0]);
        return 1;
    }
    MatMulFn fn = resolve(argv[1]);
    if (!fn) {
        std::fprintf(stderr, "unknown stage '%s'\n", argv[1]);
        return 1;
    }
    const int M = std::atoi(argv[2]), N = std::atoi(argv[3]), K = std::atoi(argv[4]);
    const int iters = std::atoi(argv[5]);
    const unsigned seed =
        argc > 6 ? static_cast<unsigned>(std::strtoul(argv[6], nullptr, 10)) : 1234u;

    float* A = pa1::alloc_floats(static_cast<std::size_t>(M) * K);
    float* B = pa1::alloc_floats(static_cast<std::size_t>(N) * K);
    float* C = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
    pa1::fill_random(A, static_cast<std::size_t>(M) * K, seed);
    pa1::fill_random(B, static_cast<std::size_t>(N) * K, seed + 1u);
    const int lda = K, ldb = K, ldc = N;

    for (int it = 0; it < iters; ++it) {
        fn(A, B, C, M, N, K, lda, ldb, ldc);
    }

    // Touch C so the whole loop can't be dead-code-eliminated with -O2.
    volatile float sink = C[0];
    (void)sink;

    pa1::free_floats(A);
    pa1::free_floats(B);
    pa1::free_floats(C);
    return 0;
}
