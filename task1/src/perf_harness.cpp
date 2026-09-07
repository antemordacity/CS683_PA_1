// perf_harness.cpp  -- profiling-only harness, NOT used for grading.
//
// main.cpp (the provided grading harness) always runs conv_naive as its
// correctness/speedup baseline. This file is a separate, minimal binary
// that runs ONLY the stage you name.
//
// Usage:
//   ./perf_harness <stage> [H W K] [seed] [--check] [--iters=N]
//
//   stage    : naive | reorder | unroll | tile | simd | simd_tiled | optimized
//   H W K    : workload size (default 2048 2048 3)
//   seed     : RNG seed (default 1234)
//   --check  : ALSO run conv_naive once and print max abs diff. Do NOT pass
//              --check while perf is attached.
//   --iters=N: how many times to call the stage function inside this one
//              process (default 10).

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "convolution.h"
#include "timer.h"
#include "utils.h"

// Default number of in-process repetitions of the target stage.
static constexpr int kDefaultIters = 10;

struct Stage {
    const char* key;
    ConvFn fn;
};

static const Stage kStages[] = {
    {"naive",      conv_naive},
    {"reorder",    conv_reorder},
    {"unroll",     conv_unroll},
    {"tile",       conv_tile},
    {"simd",       conv_simd},
    {"simd_tiled", conv_simd_tiled},
    {"optimized",  conv_optimized},
};

static constexpr int kNumStages =
    sizeof(kStages) / sizeof(kStages[0]);

static ConvFn find_stage(const char* key) {
    for (int i = 0; i < kNumStages; ++i) {
        if (std::strcmp(kStages[i].key, key) == 0) {
            return kStages[i].fn;
        }
    }

    return nullptr;
}

static void usage(const char* prog) {
    std::printf(
        "Usage: %s <stage> [H W K] [seed] [--check] [--iters=N]\n",
        prog
    );

    std::printf(
        "  stage: naive | reorder | unroll | tile | simd | "
        "simd_tiled | optimized\n"
    );

    std::printf(
        "  H W K default to 2048 2048 3; seed defaults to 1234.\n"
    );

    std::printf(
        "  --check   : also run conv_naive once and report max abs diff.\n"
    );

    std::printf(
        "              Do NOT use --check while perf is attached.\n"
    );

    std::printf(
        "  --iters=N : run the stage N times in this one process\n"
    );

    std::printf(
        "              (default %d). Divide perf's totals by N for\n",
        kDefaultIters
    );

    std::printf(
        "              per-run averages.\n"
    );
}

int main(int argc, char** argv) {
    if (argc < 2 ||
        std::strcmp(argv[1], "help") == 0 ||
        std::strcmp(argv[1], "-h") == 0) {

        usage(argv[0]);
        return (argc < 2) ? 1 : 0;
    }

    // Find requested stage.
    const ConvFn fn = find_stage(argv[1]);

    if (!fn) {
        std::printf(
            "error: unknown stage '%s'\n\n",
            argv[1]
        );

        usage(argv[0]);
        return 1;
    }

    // Defaults.
    int H = 2048;
    int W = 2048;
    int K = 3;

    unsigned seed = 1234u;

    bool check = false;

    int iters = kDefaultIters;

    auto is_flag = [](const char* s) {
        return std::strncmp(s, "--", 2) == 0;
    };

    // Parse positional arguments.
    int pos = 2;

    if (argc >= 5 && !is_flag(argv[2])) {
        H = std::atoi(argv[2]);
        W = std::atoi(argv[3]);
        K = std::atoi(argv[4]);

        pos = 5;
    }

    // Optional seed.
    if (argc > pos && !is_flag(argv[pos])) {
        seed = static_cast<unsigned>(
            std::strtoul(argv[pos], nullptr, 10)
        );

        ++pos;
    }

    // Parse flags.
    for (int i = pos; i < argc; ++i) {
        if (std::strcmp(argv[i], "--check") == 0) {
            check = true;
        }
        else if (std::strncmp(argv[i], "--iters=", 8) == 0) {
            iters = std::atoi(argv[i] + 8);
        }
    }

    // Validate iterations.
    if (iters <= 0) {
        std::printf(
            "error: --iters must be positive.\n"
        );

        return 1;
    }

    // Validate dimensions.
    if (H <= 0 || W <= 0 || K <= 0) {
        std::printf(
            "error: H, W, K must be positive.\n"
        );

        return 1;
    }

    // The SIMD stages operate on 8 floats at a time.
    if (W % 8 != 0) {
        std::printf(
            "error: W (%d) must be a multiple of 8.\n",
            W
        );

        return 1;
    }

    // Convolution uses symmetric padding, so K must be odd.
    if (K % 2 == 0) {
        std::printf(
            "error: K (%d) must be odd.\n",
            K
        );

        return 1;
    }

    // Allocate input, kernel, and output.
    float* img =
        pa1::alloc_floats(
            static_cast<std::size_t>(H) * W
        );

    float* ker =
        pa1::alloc_floats(
            static_cast<std::size_t>(K) * K
        );

    float* out =
        pa1::alloc_floats(
            static_cast<std::size_t>(H) * W
        );

    // Generate deterministic input.
    pa1::fill_random(
        img,
        static_cast<std::size_t>(H) * W,
        seed
    );

    pa1::fill_random(
        ker,
        static_cast<std::size_t>(K) * K,
        seed + 1u
    );

    // Construct padded input.
    float* in =
        pa1::make_padded(
            img,
            H,
            W,
            K
        );

    // ------------------------------------------------------------
    // Optional correctness check.
    //
    // This deliberately runs outside the profiling path.
    // ------------------------------------------------------------
    if (check) {
        float* ref =
            pa1::alloc_floats(
                static_cast<std::size_t>(H) * W
            );

        // Reference result.
        conv_naive(
            in,
            ref,
            ker,
            H,
            W,
            K
        );

        // Candidate result.
        fn(
            in,
            out,
            ker,
            H,
            W,
            K
        );

        const float diff =
            pa1::max_abs_diff(
                out,
                ref,
                H,
                W
            );

        std::printf(
            "[check] stage=%s max_abs_diff=%.6g (%s)\n",
            argv[1],
            diff,
            diff <= 1e-3f ? "OK" : "FAIL"
        );

        pa1::free_floats(ref);
    }

    // ------------------------------------------------------------
    // Profiling/timing path.
    //
    // ONLY the requested stage is repeatedly executed here.
    // ------------------------------------------------------------
    auto run = [&]() {
        fn(
            in,
            out,
            ker,
            H,
            W,
            K
        );
    };

    const double ms =
        pa1::time_median_ms(
            run,
            /*warmup=*/0,
            /*reps=*/iters
        );

    const double flops =
        pa1::conv_flops(
            H,
            W,
            K
        );

    const double gflops =
        flops / (ms * 1e6);

    std::printf(
        "stage=%s H=%d W=%d K=%d seed=%u iters=%d "
        "time(median)=%.3fms GFLOP/s=%.2f\n",
        argv[1],
        H,
        W,
        K,
        seed,
        iters,
        ms,
        gflops
    );

    // Free allocations.
    pa1::free_floats(in);
    pa1::free_floats(img);
    pa1::free_floats(ker);
    pa1::free_floats(out);

    return 0;
}