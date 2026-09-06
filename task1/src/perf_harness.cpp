// perf_harness.cpp  -- profiling-only harness, NOT used for grading.
//
// main.cpp (the provided grading harness) always runs conv_naive as its
// correctness/speedup baseline, even in "single stage" mode -- which means
// every perf record/stat against it captures naive's work too. This file is
// a separate, minimal binary that runs ONLY the stage you name. Nothing else
// executes in the timed loop, so perf counters (instructions, L1D hits/
// misses, MPKI, etc.) reflect that one stage and nothing more.
//
// This file does NOT replace main.cpp and is NOT part of what gets graded --
// it's a local dev tool for isolating perf numbers per stage.
//
// Usage:
//   ./perf_harness <stage> [H W K] [seed] [--check] [--iters=N]
//
//   stage    : naive | reorder | unroll | tile | simd | optimized
//   H W K    : workload size (default 2048 2048 3)
//   seed     : RNG seed (default 1234)
//   --check  : ALSO run conv_naive once and print max abs diff, to sanity
//              check correctness. Do NOT pass --check while perf is
//              attached -- it adds naive's own work to whatever gets
//              measured, defeating the point of this harness. Run --check
//              standalone first, then re-run without it under perf.
//   --iters=N: how many times to call the stage function inside this one
//              process (default 10). Hardware counters (instructions, L1D
//              hits/misses, etc.) come from perf attaching to the whole
//              process, not from anything this program can read itself --
//              so "average perf stats over N runs" means running the stage
//              N times in one process and dividing perf's totals by N. The
//              harness prints iters=N in its output specifically so that
//              division is easy to automate (see the Makefile's mpki/
//              perf-stat targets, which do exactly this).
//
// Examples:
//   ./perf_harness simd 2048 2048 3 --check        # verify correctness once
//   perf stat -e instructions,L1-dcache-loads,L1-dcache-load-misses \
//       ./perf_harness simd 2048 2048 3             # 10 runs, raw (summed) counters
//   perf record -g -o perf.data -- ./perf_harness tile 1024 1024 3
//   perf report -i perf.data --stdio                # everything here IS conv_tile

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "convolution.h"
#include "timer.h"
#include "utils.h"

// Default number of in-process repetitions of the target stage. All of them
// run back-to-back with no separate "warmup" phase: since perf counts the
// whole process regardless, a warmup phase would just be uncounted-by-us but
// still-counted-by-perf work, which defeats clean averaging. If you want
// caches pre-warmed, just raise --iters and drop the first columns of your
// own timing analysis; perf's totals will still divide cleanly by N.
static constexpr int kDefaultIters = 10;

struct Stage {
    const char* key;
    ConvFn fn;
};

static const Stage kStages[] = {
    {"naive", conv_naive},         {"reorder", conv_reorder},
    {"unroll", conv_unroll},       {"tile", conv_tile},
    {"simd", conv_simd},           {"optimized", conv_optimized},
};
static constexpr int kNumStages = sizeof(kStages) / sizeof(kStages[0]);

static ConvFn find_stage(const char* key) {
    for (int i = 0; i < kNumStages; ++i)
        if (std::strcmp(kStages[i].key, key) == 0) return kStages[i].fn;
    return nullptr;
}

static void usage(const char* prog) {
    std::printf("Usage: %s <stage> [H W K] [seed] [--check] [--iters=N]\n", prog);
    std::printf("  stage: naive | reorder | unroll | tile | simd | optimized\n");
    std::printf("  H W K default to 2048 2048 3; seed defaults to 1234.\n");
    std::printf("  --check   : also run conv_naive once and report max abs diff.\n");
    std::printf("              Do NOT use --check while perf is attached.\n");
    std::printf("  --iters=N : run the stage N times in this one process\n");
    std::printf("              (default %d). Divide perf's totals by N for\n", kDefaultIters);
    std::printf("              per-run averages -- see Makefile's mpki target.\n");
}

int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "help") == 0 ||
        std::strcmp(argv[1], "-h") == 0) {
        usage(argv[0]);
        return (argc < 2) ? 1 : 0;
    }

    const ConvFn fn = find_stage(argv[1]);
    if (!fn) {
        std::printf("error: unknown stage '%s'\n\n", argv[1]);
        usage(argv[0]);
        return 1;
    }

    int H = 2048, W = 2048, K = 3;
    unsigned seed = 1234u;
    bool check = false;
    int iters = kDefaultIters;

    auto is_flag = [](const char* s) { return std::strncmp(s, "--", 2) == 0; };

    int pos = 2;
    if (argc >= 5 && !is_flag(argv[2])) {
        H = std::atoi(argv[2]);
        W = std::atoi(argv[3]);
        K = std::atoi(argv[4]);
        pos = 5;
    }
    if (argc > pos && !is_flag(argv[pos])) {
        seed = static_cast<unsigned>(std::strtoul(argv[pos], nullptr, 10));
        ++pos;
    }
    for (int i = pos; i < argc; ++i) {
        if (std::strcmp(argv[i], "--check") == 0) {
            check = true;
        } else if (std::strncmp(argv[i], "--iters=", 8) == 0) {
            iters = std::atoi(argv[i] + 8);
        }
    }
    if (iters <= 0) {
        std::printf("error: --iters must be positive.\n");
        return 1;
    }

    if (H <= 0 || W <= 0 || K <= 0) {
        std::printf("error: H, W, K must be positive.\n");
        return 1;
    }
    if (W % 8 != 0) {
        std::printf("error: W (%d) must be a multiple of 8.\n", W);
        return 1;
    }
    if (K % 2 == 0) {
        std::printf("error: K (%d) must be odd.\n", K);
        return 1;
    }

    float* img = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
    float* ker = pa1::alloc_floats(static_cast<std::size_t>(K) * K);
    float* out = pa1::alloc_floats(static_cast<std::size_t>(H) * W);

    pa1::fill_random(img, static_cast<std::size_t>(H) * W, seed);
    pa1::fill_random(ker, static_cast<std::size_t>(K) * K, seed + 1u);
    float* in = pa1::make_padded(img, H, W, K);

    // ---- optional correctness check -- never run this while perf is attached
    if (check) {
        float* ref = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
        conv_naive(in, ref, ker, H, W, K);
        fn(in, out, ker, H, W, K);
        const float diff = pa1::max_abs_diff(out, ref, H, W);
        std::printf("[check] stage=%s max_abs_diff=%.6g (%s)\n", argv[1], diff,
                    diff <= 1e-3f ? "OK" : "FAIL");
        pa1::free_floats(ref);
    }

    // ---- the ONLY thing that runs repeatedly / gets profiled: this stage ----
    // No separate warmup phase: every one of the `iters` calls is counted by
    // perf regardless of what we tell time_median_ms, so folding warmup in as
    // just more iterations keeps "process total / iters" an honest average.
    auto run = [&]() { fn(in, out, ker, H, W, K); };
    const double ms = pa1::time_median_ms(run, /*warmup=*/0, /*reps=*/iters);
    const double flops = pa1::conv_flops(H, W, K);
    const double gflops = flops / (ms * 1e6);

    std::printf("stage=%s H=%d W=%d K=%d seed=%u iters=%d  time(median)=%.3fms  GFLOP/s=%.2f\n",
                argv[1], H, W, K, seed, iters, ms, gflops);

    pa1::free_floats(in);
    pa1::free_floats(img);
    pa1::free_floats(ker);
    pa1::free_floats(out);
    return 0;
}
