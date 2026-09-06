// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    for(int i = 0; i < H*W; i++)
        out[i] = 0.0f;
    int ox = 0;
    for (int oy = 0; oy < H; ++oy) {
            for (int ky = 0; ky < K; ++ky) {
                int kx = 0;
                for (; kx < K-2; kx+=3) {
                __m256 k0 = _mm256_set1_ps(ker[ky * K + kx]);
                __m256 k1 = _mm256_set1_ps(ker[ky * K + kx + 1]);
                __m256 k2 = _mm256_set1_ps(ker[ky * K + kx + 2]);
                for (ox = 0; ox < W; ox+=8) {
                    __m256 vout = _mm256_loadu_ps(&out[oy * W + ox]);
                    __m256 vin0 = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    __m256 vin1 = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx + 1)]);
                    __m256 vin2 = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx + 2)]);
                    __m256 acc0 = _mm256_mul_ps(vin0, k0);
                    __m256 acc1 = _mm256_mul_ps(vin1, k1);
                    __m256 acc2 = _mm256_fmadd_ps(vin2, k2, vout);  
                    vout = _mm256_add_ps(_mm256_add_ps(acc0, acc1), acc2);
                    _mm256_storeu_ps(&out[oy * W + ox], vout);
                }
            }
                for (; kx < K; kx++) {
                    __m256 k = _mm256_set1_ps(ker[ky * K + kx]);
                    for (ox = 0; ox < W; ox+=8) {
                        __m256 vout = _mm256_loadu_ps(&out[oy * W + ox]);
                        __m256 vin = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                        _mm256_storeu_ps(&out[oy * W + ox], _mm256_fmadd_ps(vin, k, vout));
                    }   
                }
            }
        }
    }
