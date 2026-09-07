#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    if (K <= 3) {

        for (int oy = 0; oy < H; ++oy) {
            for (int ox = 0; ox < W; ++ox) {

                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;

                for (int ky = 0; ky < K; ++ky) {
                    const float* in_row =
                        in + (oy + ky) * in_stride + ox;
                    const float* ker_row =
                        ker + ky * K;

                    acc0 += in_row[0] * ker_row[0];
                    acc1 += in_row[1] * ker_row[1];
                    acc2 += in_row[2] * ker_row[2];
                }

                out[oy * W + ox] = acc0 + acc1 + acc2;
            }
        }

    } else if (K <= 5) {

        for (int oy = 0; oy < H; ++oy) {
            for (int ox = 0; ox < W; ++ox) {

                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                float acc4 = 0.0f;

                for (int ky = 0; ky < K; ++ky) {
                    const float* in_row =
                        in + (oy + ky) * in_stride + ox;
                    const float* ker_row =
                        ker + ky * K;

                    acc0 += in_row[0] * ker_row[0];
                    acc1 += in_row[1] * ker_row[1];
                    acc2 += in_row[2] * ker_row[2];
                    acc3 += in_row[3] * ker_row[3];
                    acc4 += in_row[4] * ker_row[4];
                }

                out[oy * W + ox] =
                    acc0 + acc1 + acc2 + acc3 + acc4;
            }
        }

    } else if (K <= 7) {

        for (int oy = 0; oy < H; ++oy) {
            for (int ox = 0; ox < W; ++ox) {

                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                float acc4 = 0.0f;
                float acc5 = 0.0f;
                float acc6 = 0.0f;

                for (int ky = 0; ky < K; ++ky) {
                    const float* in_row =
                        in + (oy + ky) * in_stride + ox;
                    const float* ker_row =
                        ker + ky * K;

                    acc0 += in_row[0] * ker_row[0];
                    acc1 += in_row[1] * ker_row[1];
                    acc2 += in_row[2] * ker_row[2];
                    acc3 += in_row[3] * ker_row[3];
                    acc4 += in_row[4] * ker_row[4];
                    acc5 += in_row[5] * ker_row[5];
                    acc6 += in_row[6] * ker_row[6];
                }

                out[oy * W + ox] =
                    acc0 + acc1 + acc2 + acc3 +
                    acc4 + acc5 + acc6;
            }
        }

    } else {

        for (int oy = 0; oy < H; ++oy) {
            for (int ox = 0; ox < W; ++ox) {

                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                float acc4 = 0.0f;
                float acc5 = 0.0f;
                float acc6 = 0.0f;
                float acc7 = 0.0f;
                float acc8 = 0.0f;

                for (int ky = 0; ky < K; ++ky) {
                    const float* in_row =
                        in + (oy + ky) * in_stride + ox;
                    const float* ker_row =
                        ker + ky * K;

                    int kx = 0;

                    for (; kx + 8 < K; kx += 9) {
                        acc0 += in_row[kx]     * ker_row[kx];
                        acc1 += in_row[kx + 1] * ker_row[kx + 1];
                        acc2 += in_row[kx + 2] * ker_row[kx + 2];
                        acc3 += in_row[kx + 3] * ker_row[kx + 3];
                        acc4 += in_row[kx + 4] * ker_row[kx + 4];
                        acc5 += in_row[kx + 5] * ker_row[kx + 5];
                        acc6 += in_row[kx + 6] * ker_row[kx + 6];
                        acc7 += in_row[kx + 7] * ker_row[kx + 7];
                        acc8 += in_row[kx + 8] * ker_row[kx + 8];
                    }

                    for (; kx < K; ++kx) {
                        acc0 += in_row[kx] * ker_row[kx];
                    }
                }

                out[oy * W + ox] =
                    acc0 + acc1 + acc2 + acc3 +
                    acc4 + acc5 + acc6 + acc7 + acc8;
            }
        }
    }
}