// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"
void conv_tile(const float* in, float* out, const float* ker,
                int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    constexpr int TILE_H = 32;
    constexpr int TILE_W = 32;

    for (int oy0 = 0; oy0 < H; oy0 += TILE_H) {
        for (int ox0 = 0; ox0 < W; ox0 += TILE_W) {

            int oy_end = oy0 + TILE_H;
            int ox_end = ox0 + TILE_W;

            if (oy_end > H)
                oy_end = H;

            if (ox_end > W)
                ox_end = W;

            for (int oy = oy0; oy < oy_end; ++oy) {
                for (int ox = ox0; ox < ox_end; ++ox) {

                    float acc = 0.0f;

                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            acc +=
                                in[(oy + ky) * in_stride + (ox + kx)]
                                * ker[ky * K + kx];
                        }
                    }

                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}