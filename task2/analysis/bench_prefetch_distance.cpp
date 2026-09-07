// bench_prefetch_distance.cpp  ANALYSIS TASK 2 driver: speedup vs. prefetch distance.
//
// Links against matmul_prefetch_param.cpp, which is recompiled once per distance value
// by run_prefetch_distance_sweep.sh (-DPARAM_PREFETCH_DISTANCE=<d>). This binary just
// times the resulting kernel at a few sizes and labels the CSV rows with the distance
// it was told to expect (argv[1]) -- the sweep script is responsible for keeping that
// label in sync with the -D it actually compiled with.
//
// Usage: ./bench_prefetch_distance <distance_label> [sizes...]
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench_common.h"

extern void matmul_prefetch_param(const float* A, const float* B, float* C, int M,
                                   int N, int K, int lda, int ldb, int ldc);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <distance_label> [sizes...]\n", argv[0]);
        return 1;
    }
    const int distance_label = std::atoi(argv[1]);

    std::vector<int> sizes;
    for (int i = 2; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {512, 1024, 2048};

    const unsigned seed = 1234u;

    std::printf("distance,M,N,K,time_ms,gflops,speedup,rel_err\n");
    for (int n : sizes) {
        const int reps = bench::reps_for_size(n);
        auto ref = bench::run_and_time(matmul_naive, matmul_naive, n, n, n, seed, 1,
                                        reps);
        auto r = bench::run_and_time(matmul_prefetch_param, matmul_naive, n, n, n,
                                      seed, 1, reps);
        const double speedup = (r.ms > 0.0) ? ref.ms / r.ms : 0.0;
        std::printf("%d,%d,%d,%d,%.4f,%.3f,%.4f,%.3e\n", distance_label, n, n, n,
                    r.ms, r.gflops, speedup, r.rel_err);
        std::fflush(stdout);
    }
    return 0;
}
