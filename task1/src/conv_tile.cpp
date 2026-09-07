#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    constexpr int TILE_H = 128;
    constexpr int TILE_W = 128;

    for (int oy0 = 0; oy0 < H; oy0 += TILE_H) {
        for (int ox0 = 0; ox0 < W; ox0 += TILE_W) {

            const int oy_end = (oy0 + TILE_H < H)
                             ? oy0 + TILE_H
                             : H;

            const int ox_end = (ox0 + TILE_W < W)
                             ? ox0 + TILE_W
                             : W;

            // Compute this output tile.
            for (int oy = oy0; oy < oy_end; ++oy) {
                for (int ox = ox0; ox < ox_end; ++ox) {

                    float acc = 0.0f;

                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_row =
                            in + (oy + ky) * in_stride + ox;

                        const float* ker_row =
                            ker + ky * K;

                        for (int kx = 0; kx < K; ++kx) {
                            acc += in_row[kx] * ker_row[kx];
                        }
                    }

                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}
