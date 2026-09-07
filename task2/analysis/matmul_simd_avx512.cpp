// matmul_simd_avx512.cpp  ANALYSIS-ONLY 512-bit (AVX-512) variant, for the
// SIMD-width sweep. NOT graded. Compile with -mavx512f -mfma. Requires a CPU that
// supports AVX-512F -- check with: grep avx512f /proc/cpuinfo (run_simd_width_sweep.sh
// does this check for you and skips this variant with a note if unsupported).
//
// Structurally identical to ../src/matmul_simd.cpp (same 4-column register tile, same
// horizontal-reduction shape) except __m256/_mm256_* -> __m512/_mm512_* (16
// floats/vector instead of 8). Keeping the surrounding structure identical isolates
// SIMD width as the only variable.
#include <immintrin.h>

#include "matmul.h"

void matmul_simd_avx512(const float* A, const float* B, float* C,
                         int M, int N, int K, int lda, int ldb, int ldc) {

    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;

        int j = 0;
        for (; j + 3 < N; j += 4) {
            const float* b0 = B + static_cast<long>(j) * ldb;
            const float* b1 = B + static_cast<long>(j + 1) * ldb;
            const float* b2 = B + static_cast<long>(j + 2) * ldb;
            const float* b3 = B + static_cast<long>(j + 3) * ldb;

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            int p = 0;
            for (; p + 15 < K; p += 16) {
                __m512 va = _mm512_loadu_ps(a + p);

                __m512 vb0 = _mm512_loadu_ps(b0 + p);
                __m512 vb1 = _mm512_loadu_ps(b1 + p);
                __m512 vb2 = _mm512_loadu_ps(b2 + p);
                __m512 vb3 = _mm512_loadu_ps(b3 + p);

                acc0 = _mm512_fmadd_ps(va, vb0, acc0);
                acc1 = _mm512_fmadd_ps(va, vb1, acc1);
                acc2 = _mm512_fmadd_ps(va, vb2, acc2);
                acc3 = _mm512_fmadd_ps(va, vb3, acc3);
            }

            float sum0 = _mm512_reduce_add_ps(acc0);
            float sum1 = _mm512_reduce_add_ps(acc1);
            float sum2 = _mm512_reduce_add_ps(acc2);
            float sum3 = _mm512_reduce_add_ps(acc3);

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
            __m512 acc = _mm512_setzero_ps();

            int p = 0;
            for (; p + 15 < K; p += 16) {
                __m512 va = _mm512_loadu_ps(a + p);
                __m512 vb = _mm512_loadu_ps(b + p);
                acc = _mm512_fmadd_ps(va, vb, acc);
            }

            float sum = _mm512_reduce_add_ps(acc);

            for (; p < K; ++p) sum += a[p] * b[p];

            C[static_cast<long>(i) * ldc + j] = sum;
        }
    }
}
