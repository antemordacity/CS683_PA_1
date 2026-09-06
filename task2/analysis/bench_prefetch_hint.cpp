// bench_prefetch_hint.cpp  ANALYSIS TASK 3 driver: speedup vs. prefetch locality hint.
//
// Links against matmul_prefetch_param.cpp, recompiled once per hint by
// run_prefetch_hint_sweep.sh (-DPARAM_PREFETCH_HINT=_MM_HINT_T0|T1|T2|NTA), at the
// distance found best by the distance sweep (pass it via argv so the two stay
// consistent instead of hardcoding a number here).
//
// Usage: ./bench_prefetch_hint <hint_label> <distance_used> [sizes...]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "bench_common.h"

extern void matmul_prefetch_param(const float* A, const float* B, float* C, int M,
                                   int N, int K, int lda, int ldb, int ldc);

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <hint_label> <distance_used> [sizes...]\n",
                      argv[0]);
        return 1;
    }
    const char* hint_label = argv[1];
    const int distance_used = std::atoi(argv[2]);

    std::vector<int> sizes;
    for (int i = 3; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {512, 1024, 2048};

    const unsigned seed = 1234u;

    std::printf("hint,distance,M,N,K,time_ms,gflops,speedup,rel_err\n");
    for (int n : sizes) {
        const int reps = bench::reps_for_size(n);
        auto ref = bench::run_and_time(matmul_naive, matmul_naive, n, n, n, seed, 1,
                                        reps);
        auto r = bench::run_and_time(matmul_prefetch_param, matmul_naive, n, n, n,
                                      seed, 1, reps);
        const double speedup = (r.ms > 0.0) ? ref.ms / r.ms : 0.0;
        std::printf("%s,%d,%d,%d,%d,%.4f,%.3f,%.4f,%.3e\n", hint_label,
                    distance_used, n, n, n, r.ms, r.gflops, speedup, r.rel_err);
        std::fflush(stdout);
    }
    return 0;
}
