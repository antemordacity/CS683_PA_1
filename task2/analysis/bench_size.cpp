// bench_size.cpp  ANALYSIS TASK 1 driver: speedup vs. matrix size.
//
// Runs matmul_naive, matmul_simd, matmul_prefetch, matmul_optimized (the exact
// functions in ../src, compiled with the pinned flags) at each of a list of square
// M=N=K sizes and prints one CSV row per (stage, size) to stdout.
//
// Usage:
//   ./bench_size                       # default size list (128 .. 4096)
//   ./bench_size 128 256 512 1024       # custom size list
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench_common.h"

int main(int argc, char** argv) {
    std::vector<int> sizes;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    } else {
        sizes = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096};
    }
    const unsigned seed = 1234u;
    const int warmup = 1;

    std::printf("stage,M,N,K,time_ms,gflops,speedup,rel_err\n");
    for (int n : sizes) {
        const int reps = bench::reps_for_size(n);

        auto ref = bench::run_and_time(matmul_naive, matmul_naive, n, n, n, seed,
                                        warmup, reps);
        std::printf("naive,%d,%d,%d,%.4f,%.3f,%.4f,%.3e\n", n, n, n, ref.ms,
                    ref.gflops, 1.0, ref.rel_err);
        std::fflush(stdout);

        struct { const char* name; MatMulFn fn; } stages[] = {
            {"simd", matmul_simd},
            {"prefetch", matmul_prefetch},
            {"optimized", matmul_optimized},
        };
        for (auto& s : stages) {
            auto r = bench::run_and_time(s.fn, matmul_naive, n, n, n, seed, warmup,
                                          reps);
            const double speedup = (r.ms > 0.0) ? ref.ms / r.ms : 0.0;
            std::printf("%s,%d,%d,%d,%.4f,%.3f,%.4f,%.3e\n", s.name, n, n, n, r.ms,
                        r.gflops, speedup, r.rel_err);
            std::fflush(stdout);
        }
    }
    return 0;
}
