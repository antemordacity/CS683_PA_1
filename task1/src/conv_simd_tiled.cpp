// conv_simd_tile.cpp  STAGE 5: CACHE TILING + AVX2 SIMD

#include <immintrin.h>
#include "convolution.h"

void conv_simd_tiled(const float* in, float* out, const float* ker,
                    int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    constexpr int TILE_H = 64;
    constexpr int TILE_W = 64;

    for (int oy0 = 0; oy0 < H; oy0 += TILE_H) {
        for (int ox0 = 0; ox0 < W; ox0 += TILE_W) {

            const int oy_end = (oy0 + TILE_H < H)
                             ? oy0 + TILE_H
                             : H;

            const int ox_end = (ox0 + TILE_W < W)
                             ? ox0 + TILE_W
                             : W;

            for (int oy = oy0; oy < oy_end; ++oy) {

                int ox = ox0;

                // SIMD: process 8 output pixels at a time
                for (; ox + 7 < ox_end; ox += 8) {

                    __m256 acc = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ++ky) {

                        const float* in_row =
                            in + (oy + ky) * in_stride + ox;

                        const float* ker_row =
                            ker + ky * K;

                        for (int kx = 0; kx < K; ++kx) {

                            __m256 vin =
                                _mm256_loadu_ps(in_row + kx);

                            __m256 k =
                                _mm256_set1_ps(ker_row[kx]);

                            acc = _mm256_fmadd_ps(vin, k, acc);
                        }
                    }

                    _mm256_storeu_ps(out + oy * W + ox, acc);
                }

                // Scalar remainder inside the tile
                for (; ox < ox_end; ++ox) {

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
