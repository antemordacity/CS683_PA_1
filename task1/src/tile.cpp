#include <cstdio>
#include <cstdlib>
#include <vector>

#include "convolution.h"
#include "timer.h"
#include "utils.h"
void conv_tile_T(const float* in, float* out, const float* ker,
                 int H, int W, int K, int T) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // Initialize output
    for (int i = 0; i < H * W; ++i)
        out[i] = 0.0f;

    // Output tiles
    for (int oy0 = 0; oy0 < H; oy0 += T) {
        for (int ox0 = 0; ox0 < W; ox0 += T) {

            // Kernel
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {

                    const float kval = ker[ky * K + kx];

                    // Pixels inside tile
                    for (int oy = oy0;
                         oy < oy0 + T && oy < H;
                         ++oy) {

                        for (int ox = ox0;
                             ox < ox0 + T && ox < W;
                             ++ox) {

                            out[oy * W + ox] +=
                                in[(oy + ky) * in_stride + (ox + kx)]
                                * kval;
                        }
                    }
                }
            }
        }
    }
}


int main() {

    const int H = 2048;
    const int W = 2048;
    const int K = 32;

    const int tile_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    const int num_tiles =
        sizeof(tile_sizes) / sizeof(tile_sizes[0]);

    const int warmup = 2;
    const int reps = 7;

    // Allocate data
    float* img = pa1::alloc_floats(
        static_cast<std::size_t>(H) * W);

    float* ker = pa1::alloc_floats(
        static_cast<std::size_t>(K) * K);

    float* out = pa1::alloc_floats(
        static_cast<std::size_t>(H) * W);

    float* ref = pa1::alloc_floats(
        static_cast<std::size_t>(H) * W);

    // Same random data every time
    pa1::fill_random(
        img,
        static_cast<std::size_t>(H) * W,
        1234u);

    pa1::fill_random(
        ker,
        static_cast<std::size_t>(K) * K,
        1235u);

    float* in = pa1::make_padded(img, H, W, K);

    // Reference result
    conv_naive(in, ref, ker, H, W, K);

    printf("H=%d W=%d K=%d\n\n", H, W, K);

    printf("%-10s %-15s %-15s\n",
           "Tile", "Time (ms)", "Speedup");

    printf("------------------------------------------\n");

    // First get naive timing
    auto naive_run = [&]() {
        conv_naive(in, out, ker, H, W, K);
    };

    double naive_ms =
        pa1::time_median_ms(naive_run, warmup, reps);

    printf("%-10s %-15.3f %-15.3f\n",
           "naive",
           naive_ms,
           1.0);

    // Test every tile size
    for (int i = 0; i < num_tiles; ++i) {

        const int T = tile_sizes[i];

        auto tile_run = [&]() {
            conv_tile_T(in, out, ker, H, W, K, T);
        };

        // Correctness check
        tile_run();

        float error =
            pa1::max_abs_diff(out, ref, H, W);

        if (error > 1e-3f) {
            printf("T=%d INCORRECT error=%g\n",
                   T, error);
            continue;
        }

        double ms =
            pa1::time_median_ms(tile_run, warmup, reps);

        double speedup =
            naive_ms / ms;

        printf("%-10d %-15.3f %-15.3fx\n",
               T, ms, speedup);
    }

    pa1::free_floats(in);
    pa1::free_floats(img);
    pa1::free_floats(ker);
    pa1::free_floats(out);
    pa1::free_floats(ref);

    return 0;
}