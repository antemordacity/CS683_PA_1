// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {

    for (int i = 0; i < M; ++i) {

        const float* a = A + static_cast<long>(i) * lda;

        int j = 0;

        // Register tile: compute 4 output elements at once.
        for (; j + 3 < N; j += 4) {

            const float* b0 = B + static_cast<long>(j) * ldb;
            const float* b1 = B + static_cast<long>(j + 1) * ldb;
            const float* b2 = B + static_cast<long>(j + 2) * ldb;
            const float* b3 = B + static_cast<long>(j + 3) * ldb;

            // Four independent AVX2 accumulators.
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            int p = 0;

            // Process 8 K-elements at a time.
            for (; p + 7 < K; p += 8) {

                __m256 va = _mm256_loadu_ps(a + p);

                __m256 vb0 = _mm256_loadu_ps(b0 + p);
                __m256 vb1 = _mm256_loadu_ps(b1 + p);
                __m256 vb2 = _mm256_loadu_ps(b2 + p);
                __m256 vb3 = _mm256_loadu_ps(b3 + p);

                acc0 = _mm256_fmadd_ps(va, vb0, acc0);
                acc1 = _mm256_fmadd_ps(va, vb1, acc1);
                acc2 = _mm256_fmadd_ps(va, vb2, acc2);
                acc3 = _mm256_fmadd_ps(va, vb3, acc3);
            }

            // Horizontal reduction of the four accumulators.
            float temp0[8];
            float temp1[8];
            float temp2[8];
            float temp3[8];

            _mm256_storeu_ps(temp0, acc0);
            _mm256_storeu_ps(temp1, acc1);
            _mm256_storeu_ps(temp2, acc2);
            _mm256_storeu_ps(temp3, acc3);

            float sum0 = temp0[0] + temp0[1] + temp0[2] + temp0[3]
                       + temp0[4] + temp0[5] + temp0[6] + temp0[7];

            float sum1 = temp1[0] + temp1[1] + temp1[2] + temp1[3]
                       + temp1[4] + temp1[5] + temp1[6] + temp1[7];

            float sum2 = temp2[0] + temp2[1] + temp2[2] + temp2[3]
                       + temp2[4] + temp2[5] + temp2[6] + temp2[7];

            float sum3 = temp3[0] + temp3[1] + temp3[2] + temp3[3]
                       + temp3[4] + temp3[5] + temp3[6] + temp3[7];

            // Scalar cleanup for K not divisible by 8.
            for (; p < K; ++p) {
                sum0 += a[p] * b0[p];
                sum1 += a[p] * b1[p];
                sum2 += a[p] * b2[p];
                sum3 += a[p] * b3[p];
            }

            C[static_cast<long>(i) * ldc + j]     = sum0;
            C[static_cast<long>(i) * ldc + j + 1] = sum1;
            C[static_cast<long>(i) * ldc + j + 2] = sum2;
            C[static_cast<long>(i) * ldc + j + 3] = sum3;
        }

        // Handle remaining columns when N is not divisible by 4.
        for (; j < N; ++j) {

            const float* b = B + static_cast<long>(j) * ldb;

            __m256 acc = _mm256_setzero_ps();

            int p = 0;

            for (; p + 7 < K; p += 8) {
                __m256 va = _mm256_loadu_ps(a + p);
                __m256 vb = _mm256_loadu_ps(b + p);

                acc = _mm256_fmadd_ps(va, vb, acc);
            }

            float temp[8];
            _mm256_storeu_ps(temp, acc);

            float sum = temp[0] + temp[1] + temp[2] + temp[3]
                      + temp[4] + temp[5] + temp[6] + temp[7];

            for (; p < K; ++p) {
                sum += a[p] * b[p];
            }

            C[static_cast<long>(i) * ldc + j] = sum;
        }
    }
}