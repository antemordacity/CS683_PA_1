// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {

    // Cache tile size.
    const int TILE = 32;

    // How far ahead to prefetch in the K dimension.
    const int PREFETCH_DISTANCE = 8;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[static_cast<long>(i) * ldc + j] = 0.0f;
        }
    }

    for (int ii = 0; ii < M; ii += TILE) {

        for (int jj = 0; jj < N; jj += TILE) {

            for (int kk = 0; kk < K; kk += TILE) {

                int i_end = (ii + TILE < M) ? ii + TILE : M;
                int j_end = (jj + TILE < N) ? jj + TILE : N;
                int k_end = (kk + TILE < K) ? kk + TILE : K;

                for (int i = ii; i < i_end; ++i) {

                    const float* a = A + static_cast<long>(i) * lda;

                    for (int j = jj; j < j_end; ++j) {

                        const float* b = B + static_cast<long>(j) * ldb;

                        // Load the existing C value because we process
                        // K in multiple tiles.
                        float sum = C[static_cast<long>(i) * ldc + j];

                        int p = kk;

                        // SIMD over the current K tile.
                        __m256 acc = _mm256_setzero_ps();

                        for (; p + 7 < k_end; p += 8) {

                            // Prefetch future A and B data.
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

                            __m256 va = _mm256_loadu_ps(a + p);
                            __m256 vb = _mm256_loadu_ps(b + p);

                            acc = _mm256_fmadd_ps(va, vb, acc);
                        }

                        // Reduce SIMD accumulator.
                        float temp[8];
                        _mm256_storeu_ps(temp, acc);

                        sum += temp[0] + temp[1] + temp[2] + temp[3]
                             + temp[4] + temp[5] + temp[6] + temp[7];

                        // Scalar cleanup for K-tile tail.
                        for (; p < k_end; ++p) {
                            sum += a[p] * b[p];
                        }

                        C[static_cast<long>(i) * ldc + j] = sum;
                    }
                }
            }
        }
    }
}
