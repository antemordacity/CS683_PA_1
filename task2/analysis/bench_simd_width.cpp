// bench_simd_width.cpp  ANALYSIS TASK 4 driver: speedup vs. SIMD width.
//
// matmul_simd_sse, matmul_simd (AVX2, the real ../src file), and matmul_simd_avx512
// are compiled as SEPARATE OBJECT FILES with different -m flags by
// run_simd_width_sweep.sh, then linked together into this one binary -- so all three
// widths are measured in the same process on the same random data.
//
// Usage: ./bench_simd_width [--no-avx512] [sizes...]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "bench_common.h"

extern void matmul_simd_sse(const float*, const float*, float*, int, int, int, int,
                             int, int);
#ifdef HAVE_AVX512
extern void matmul_simd_avx512(const float*, const float*, float*, int, int, int, int,
                                int, int);
#endif

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-avx512") == 0) continue;  // handled at compile time
        sizes.push_back(std::atoi(argv[i]));
    }
    if (sizes.empty()) sizes = {128, 256, 512, 1024, 2048};

    const unsigned seed = 1234u;

    std::printf("width,M,N,K,time_ms,gflops,speedup,rel_err\n");
    for (int n : sizes) {
        const int reps = bench::reps_for_size(n);
        auto ref = bench::run_and_time(matmul_naive, matmul_naive, n, n, n, seed, 1,
                                        reps);
        std::printf("naive,%d,%d,%d,%.4f,%.3f,%.4f,%.3e\n", n, n, n, ref.ms,
                    ref.gflops, 1.0, ref.rel_err);

        auto r128 = bench::run_and_time(matmul_simd_sse, matmul_naive, n, n, n, seed,
                                         1, reps);
        std::printf("128,%d,%d,%d,%.4f,%.3f,%.4f,%.3e\n", n, n, n, r128.ms,
                    r128.gflops, ref.ms / r128.ms, r128.rel_err);

        auto r256 = bench::run_and_time(matmul_simd, matmul_naive, n, n, n, seed, 1,
                                         reps);
        std::printf("256,%d,%d,%d,%.4f,%.3f,%.4f,%.3e\n", n, n, n, r256.ms,
                    r256.gflops, ref.ms / r256.ms, r256.rel_err);

#ifdef HAVE_AVX512
        auto r512 = bench::run_and_time(matmul_simd_avx512, matmul_naive, n, n, n,
                                         seed, 1, reps);
        std::printf("512,%d,%d,%d,%.4f,%.3f,%.4f,%.3e\n", n, n, n, r512.ms,
                    r512.gflops, ref.ms / r512.ms, r512.rel_err);
#endif
        std::fflush(stdout);
    }
    return 0;
}
