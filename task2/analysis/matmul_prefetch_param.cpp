// matmul_prefetch_param.cpp  ANALYSIS-ONLY. NOT graded, NOT src/matmul_prefetch.cpp.
//
// Byte-for-byte the same kernel structure as ../src/matmul_prefetch.cpp, except
// PREFETCH_DISTANCE and the prefetch locality hint are compile-time configurable so
// bench_prefetch_distance.cpp / bench_prefetch_hint.cpp can sweep them without
// touching the graded file:
//
//   -DPARAM_PREFETCH_DISTANCE=<int>                          (default 8)
//   -DPARAM_PREFETCH_HINT=_MM_HINT_T0|T1|T2|NTA               (default _MM_HINT_T0)
//
// IMPORTANT: if you re-tune TILE or the loop structure in the real
// src/matmul_prefetch.cpp, mirror those changes here too, or this sweep will be
// characterizing a kernel shape you no longer submit.
#include <immintrin.h>

#include "matmul.h"

#ifndef PARAM_PREFETCH_DISTANCE
#define PARAM_PREFETCH_DISTANCE 8
#endif
#ifndef PARAM_PREFETCH_HINT
#define PARAM_PREFETCH_HINT _MM_HINT_T0
#endif

void matmul_prefetch_param(const float* A, const float* B, float* C,
                            int M, int N, int K, int lda, int ldb, int ldc) {

    const int TILE = 32;
    const int PREFETCH_DISTANCE = PARAM_PREFETCH_DISTANCE;

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

                        float sum = C[static_cast<long>(i) * ldc + j];

                        int p = kk;
                        __m256 acc = _mm256_setzero_ps();

                        for (; p + 7 < k_end; p += 8) {
                            int pf = p + PREFETCH_DISTANCE;

                            if (pf < k_end) {
                                _mm_prefetch(
                                    reinterpret_cast<const char*>(a + pf),
                                    PARAM_PREFETCH_HINT);
                                _mm_prefetch(
                                    reinterpret_cast<const char*>(b + pf),
                                    PARAM_PREFETCH_HINT);
                            }

                            __m256 va = _mm256_loadu_ps(a + p);
                            __m256 vb = _mm256_loadu_ps(b + p);

                            acc = _mm256_fmadd_ps(va, vb, acc);
                        }

                        float temp[8];
                        _mm256_storeu_ps(temp, acc);

                        sum += temp[0] + temp[1] + temp[2] + temp[3]
                             + temp[4] + temp[5] + temp[6] + temp[7];

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
