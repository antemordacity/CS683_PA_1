// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"
#include<iostream>
void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  
    int kx = 0;
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f, acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f, acc8 = 0.0f, acc9 = 0.0f, acc10 = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (kx = 0; kx < K - 10; kx+=11) {
                    acc0 += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    acc1 += in[(oy + ky) * in_stride + (ox + kx + 1)] * ker[ky * K + kx + 1];
                    acc2 += in[(oy + ky) * in_stride + (ox + kx + 2)] * ker[ky * K + kx + 2];
                    acc3 += in[(oy + ky) * in_stride + (ox + kx + 3)] * ker[ky * K + kx + 3];
                    acc4 += in[(oy + ky) * in_stride + (ox + kx + 4)] * ker[ky * K + kx + 4];
                    acc5 += in[(oy + ky) * in_stride + (ox + kx + 5)] * ker[ky * K + kx + 5];
                    acc6 += in[(oy + ky) * in_stride + (ox + kx + 6)] * ker[ky * K + kx + 6];
                    acc7 += in[(oy + ky) * in_stride + (ox + kx + 7)] * ker[ky * K + kx + 7];
                    acc8 += in[(oy + ky) * in_stride + (ox + kx + 8)] * ker[ky * K + kx + 8];
                    acc9 += in[(oy + ky) * in_stride + (ox + kx + 9)] * ker[ky * K + kx + 9];
                    acc10 += in[(oy + ky) * in_stride + (ox + kx + 10)] * ker[ky * K + kx + 10];
                }
                for(; kx < K; ++kx){
                    acc0 += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + acc10;
        }
    }
}
