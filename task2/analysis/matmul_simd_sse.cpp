// matmul_simd_sse.cpp  ANALYSIS-ONLY 128-bit (SSE) variant, for the SIMD-width sweep.
// NOT graded, NOT compiled with -mavx2/-mfma. Compile with e.g. -msse4.1 only -- no
// AVX, no FMA (FMA instructions require VEX encoding, i.e. at least AVX; a genuine
// "SSE" data point has to use separate mul+add, which is the honest 128-bit baseline).
//
// Structurally identical to ../src/matmul_simd.cpp (same 4-column register tile, same
// horizontal-reduction shape) except __m256/_mm256_* -> __m128/_mm_* (4 floats/vector
// instead of 8) and _mm256_fmadd_ps -> separate _mm_mul_ps + _mm_add_ps. Keeping the
// surrounding structure identical isolates SIMD width as the only variable.
#include <immintrin.h>

#include "matmul.h"

void matmul_simd_sse(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {

    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;

        int j = 0;
        for (; j + 3 < N; j += 4) {
            const float* b0 = B + static_cast<long>(j) * ldb;
            const float* b1 = B + static_cast<long>(j + 1) * ldb;
            const float* b2 = B + static_cast<long>(j + 2) * ldb;
            const float* b3 = B + static_cast<long>(j + 3) * ldb;

            __m128 acc0 = _mm_setzero_ps();
            __m128 acc1 = _mm_setzero_ps();
            __m128 acc2 = _mm_setzero_ps();
            __m128 acc3 = _mm_setzero_ps();

            int p = 0;
            for (; p + 3 < K; p += 4) {
                __m128 va = _mm_loadu_ps(a + p);

                __m128 vb0 = _mm_loadu_ps(b0 + p);
                __m128 vb1 = _mm_loadu_ps(b1 + p);
                __m128 vb2 = _mm_loadu_ps(b2 + p);
                __m128 vb3 = _mm_loadu_ps(b3 + p);

                acc0 = _mm_add_ps(acc0, _mm_mul_ps(va, vb0));
                acc1 = _mm_add_ps(acc1, _mm_mul_ps(va, vb1));
                acc2 = _mm_add_ps(acc2, _mm_mul_ps(va, vb2));
                acc3 = _mm_add_ps(acc3, _mm_mul_ps(va, vb3));
            }

            float temp0[4], temp1[4], temp2[4], temp3[4];
            _mm_storeu_ps(temp0, acc0);
            _mm_storeu_ps(temp1, acc1);
            _mm_storeu_ps(temp2, acc2);
            _mm_storeu_ps(temp3, acc3);

            float sum0 = temp0[0] + temp0[1] + temp0[2] + temp0[3];
            float sum1 = temp1[0] + temp1[1] + temp1[2] + temp1[3];
            float sum2 = temp2[0] + temp2[1] + temp2[2] + temp2[3];
            float sum3 = temp3[0] + temp3[1] + temp3[2] + temp3[3];

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

        for (; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m128 acc = _mm_setzero_ps();

            int p = 0;
            for (; p + 3 < K; p += 4) {
                __m128 va = _mm_loadu_ps(a + p);
                __m128 vb = _mm_loadu_ps(b + p);
                acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
            }

            float temp[4];
            _mm_storeu_ps(temp, acc);
            float sum = temp[0] + temp[1] + temp[2] + temp[3];

            for (; p < K; ++p) sum += a[p] * b[p];

            C[static_cast<long>(i) * ldc + j] = sum;
        }
    }
}
