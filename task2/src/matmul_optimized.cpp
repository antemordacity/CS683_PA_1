// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>

#include "matmul.h"

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {

    const int TILE = 32;
    const int PREFETCH_DISTANCE = 32;

    // C must start from zero because K is processed in tiles.
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[static_cast<long>(i) * ldc + j] = 0.0f;
        }
    }

    // Cache blocking.
    for (int ii = 0; ii < M; ii += TILE) {

        for (int jj = 0; jj < N; jj += TILE) {

            for (int kk = 0; kk < K; kk += TILE) {

                const int i_end = (ii + TILE < M) ? ii + TILE : M;
                const int j_end = (jj + TILE < N) ? jj + TILE : N;
                const int k_end = (kk + TILE < K) ? kk + TILE : K;

                for (int i = ii; i < i_end; ++i) {

                    const float* a =
                        A + static_cast<long>(i) * lda;

                    int j = jj;

                    // Register tile: four C values at once.
                    for (; j + 3 < j_end; j += 4) {

                        const float* b0 =
                            B + static_cast<long>(j) * ldb;

                        const float* b1 =
                            B + static_cast<long>(j + 1) * ldb;

                        const float* b2 =
                            B + static_cast<long>(j + 2) * ldb;

                        const float* b3 =
                            B + static_cast<long>(j + 3) * ldb;

                        // Load the partial sums accumulated by
                        // previous K tiles.
                        float sum0 =
                            C[static_cast<long>(i) * ldc + j];

                        float sum1 =
                            C[static_cast<long>(i) * ldc + j + 1];

                        float sum2 =
                            C[static_cast<long>(i) * ldc + j + 2];

                        float sum3 =
                            C[static_cast<long>(i) * ldc + j + 3];

                        __m256 acc0 = _mm256_setzero_ps();
                        __m256 acc1 = _mm256_setzero_ps();
                        __m256 acc2 = _mm256_setzero_ps();
                        __m256 acc3 = _mm256_setzero_ps();

                        int p = kk;

                        // SIMD over K tile.
                        for (; p + 7 < k_end; p += 8) {

                            int pf = p + PREFETCH_DISTANCE;

                            if (pf < k_end) {
                                _mm_prefetch(
                                    reinterpret_cast<const char*>(a + pf),
                                    _MM_HINT_T0
                                );

                                _mm_prefetch(
                                    reinterpret_cast<const char*>(b0 + pf),
                                    _MM_HINT_T0
                                );

                                _mm_prefetch(
                                    reinterpret_cast<const char*>(b1 + pf),
                                    _MM_HINT_T0
                                );

                                _mm_prefetch(
                                    reinterpret_cast<const char*>(b2 + pf),
                                    _MM_HINT_T0
                                );

                                _mm_prefetch(
                                    reinterpret_cast<const char*>(b3 + pf),
                                    _MM_HINT_T0
                                );
                            }

                            __m256 va =
                                _mm256_loadu_ps(a + p);

                            __m256 vb0 =
                                _mm256_loadu_ps(b0 + p);

                            __m256 vb1 =
                                _mm256_loadu_ps(b1 + p);

                            __m256 vb2 =
                                _mm256_loadu_ps(b2 + p);

                            __m256 vb3 =
                                _mm256_loadu_ps(b3 + p);

                            acc0 =
                                _mm256_fmadd_ps(va, vb0, acc0);

                            acc1 =
                                _mm256_fmadd_ps(va, vb1, acc1);

                            acc2 =
                                _mm256_fmadd_ps(va, vb2, acc2);

                            acc3 =
                                _mm256_fmadd_ps(va, vb3, acc3);
                        }

                        // Reduce the four AVX2 accumulators.
                        float temp0[8];
                        float temp1[8];
                        float temp2[8];
                        float temp3[8];

                        _mm256_storeu_ps(temp0, acc0);
                        _mm256_storeu_ps(temp1, acc1);
                        _mm256_storeu_ps(temp2, acc2);
                        _mm256_storeu_ps(temp3, acc3);

                        sum0 += temp0[0] + temp0[1]
                              + temp0[2] + temp0[3]
                              + temp0[4] + temp0[5]
                              + temp0[6] + temp0[7];

                        sum1 += temp1[0] + temp1[1]
                              + temp1[2] + temp1[3]
                              + temp1[4] + temp1[5]
                              + temp1[6] + temp1[7];

                        sum2 += temp2[0] + temp2[1]
                              + temp2[2] + temp2[3]
                              + temp2[4] + temp2[5]
                              + temp2[6] + temp2[7];

                        sum3 += temp3[0] + temp3[1]
                              + temp3[2] + temp3[3]
                              + temp3[4] + temp3[5]
                              + temp3[6] + temp3[7];

                        // K-tile tail.
                        for (; p < k_end; ++p) {
                            sum0 += a[p] * b0[p];
                            sum1 += a[p] * b1[p];
                            sum2 += a[p] * b2[p];
                            sum3 += a[p] * b3[p];
                        }

                        C[static_cast<long>(i) * ldc + j] =
                            sum0;

                        C[static_cast<long>(i) * ldc + j + 1] =
                            sum1;

                        C[static_cast<long>(i) * ldc + j + 2] =
                            sum2;

                        C[static_cast<long>(i) * ldc + j + 3] =
                            sum3;
                    }

                    // Remaining columns of the tile.
                    for (; j < j_end; ++j) {

                        const float* b =
                            B + static_cast<long>(j) * ldb;

                        float sum =
                            C[static_cast<long>(i) * ldc + j];

                        __m256 acc =
                            _mm256_setzero_ps();

                        int p = kk;

                        for (; p + 7 < k_end; p += 8) {

                            int pf = p + PREFETCH_DISTANCE;

                            if (pf < k_end) {
                                _mm_prefetch(
                                    reinterpret_cast<const char*>(a + pf),
                                    _MM_HINT_T0
                                );

                                _mm_prefetch(
                                    reinterpret_cast<const char*>(b + pf),
                                    _MM_HINT_T0
                                );
                            }

                            __m256 va =
                                _mm256_loadu_ps(a + p);

                            __m256 vb =
                                _mm256_loadu_ps(b + p);

                            acc =
                                _mm256_fmadd_ps(va, vb, acc);
                        }

                        float temp[8];

                        _mm256_storeu_ps(temp, acc);

                        sum += temp[0] + temp[1]
                             + temp[2] + temp[3]
                             + temp[4] + temp[5]
                             + temp[6] + temp[7];

                        for (; p < k_end; ++p) {
                            sum += a[p] * b[p];
                        }

                        C[static_cast<long>(i) * ldc + j] =
                            sum;
                    }
                }
            }
        }
    }
}
