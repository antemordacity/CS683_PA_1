#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;


// ============================================================
// Naive convolution
// ============================================================

void conv_naive(const float* in, float* out, const float* ker,
                int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {

            float acc = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {

                    acc +=
                        in[(oy + ky) * in_stride + (ox + kx)]
                        * ker[ky * K + kx];
                }
            }

            out[oy * W + ox] = acc;
        }
    }
}


// ============================================================
// Regular 2D tiled convolution
// ============================================================

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K,
               int TILE_H, int TILE_W) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy0 = 0; oy0 < H; oy0 += TILE_H) {
        for (int ox0 = 0; ox0 < W; ox0 += TILE_W) {

            const int oy_end =
                std::min(oy0 + TILE_H, H);

            const int ox_end =
                std::min(ox0 + TILE_W, W);

            for (int oy = oy0; oy < oy_end; ++oy) {
                for (int ox = ox0; ox < ox_end; ++ox) {

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


// ============================================================
// Benchmark helper
// ============================================================

template <typename Func>
double benchmark(Func func, int runs) {

    std::vector<double> times;
    times.reserve(runs);

    // Warmup
    func();

    for (int i = 0; i < runs; ++i) {

        auto start = Clock::now();

        func();

        auto end = Clock::now();

        double elapsed =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        times.push_back(elapsed);
    }

    // Median
    std::sort(times.begin(), times.end());

    return times[times.size() / 2];
}


// ============================================================
// Main
// ============================================================

int main() {

    // Matrix sizes to test
    const std::vector<int> matrix_sizes = {
        128,
        256,
        512,
        1024,
        2048
    };

    // Kernel sizes to test
    const std::vector<int> kernel_sizes = {
        3,
        5,
        11,
        21,
    };

    // Tile sizes to test
    const std::vector<int> tile_sizes = {
        4,
        8,
        16,
        32,
        64,
        128
    };

    // Number of timing repetitions
    const int RUNS = 7;


    // --------------------------------------------------------
    // Random input
    // --------------------------------------------------------

    std::mt19937 rng(12345);

    std::uniform_real_distribution<float> dist(
        0.0f,
        1.0f
    );


    // --------------------------------------------------------
    // CSV file
    // --------------------------------------------------------

    std::ofstream csv("conv_results.csv");

    if (!csv) {
        std::cerr << "Could not open conv_results.csv\n";
        return 1;
    }

    csv << "matrix_size,K,tile_size,"
        << "naive_ms,tiled_ms,speedup,max_error\n";


    // --------------------------------------------------------
    // Header
    // --------------------------------------------------------

    std::cout
        << "=============================================\n"
        << "       CONVOLUTION TILING BENCHMARK\n"
        << "=============================================\n\n";


    // --------------------------------------------------------
    // Run experiments
    // --------------------------------------------------------

    for (int N : matrix_sizes) {

        int H = N;
        int W = N;

        std::cout
            << "\nMatrix: "
            << N << " x " << N
            << "\n";


        for (int K : kernel_sizes) {

            int p = K / 2;

            int padded_H = H + 2 * p;
            int padded_W = W + 2 * p;

            std::vector<float> in(
                padded_H * padded_W
            );

            std::vector<float> ker(
                K * K
            );

            std::vector<float> out_naive(
                H * W
            );

            std::vector<float> out_tile(
                H * W
            );


            // ------------------------------------------------
            // Initialize
            // ------------------------------------------------

            for (float& x : in)
                x = dist(rng);

            for (float& x : ker)
                x = dist(rng);


            // ------------------------------------------------
            // Benchmark naive
            // ------------------------------------------------

            double naive_ms = benchmark(
                [&]() {
                    conv_naive(
                        in.data(),
                        out_naive.data(),
                        ker.data(),
                        H,
                        W,
                        K
                    );
                },
                RUNS
            );


            std::cout
                << "\n  K = "
                << K
                << "   naive = "
                << std::fixed
                << std::setprecision(3)
                << naive_ms
                << " ms\n";


            // ------------------------------------------------
            // Test every tile size
            // ------------------------------------------------

            for (int TILE : tile_sizes) {

                double tiled_ms = benchmark(
                    [&]() {
                        conv_tile(
                            in.data(),
                            out_tile.data(),
                            ker.data(),
                            H,
                            W,
                            K,
                            TILE,
                            TILE
                        );
                    },
                    RUNS
                );


                // ------------------------------------------------
                // Correctness
                // ------------------------------------------------

                float max_error = 0.0f;

                for (size_t i = 0;
                     i < out_naive.size();
                     ++i) {

                    max_error = std::max(
                        max_error,
                        std::abs(
                            out_naive[i] - out_tile[i]
                        )
                    );
                }


                // ------------------------------------------------
                // Speedup
                // ------------------------------------------------

                double speedup =
                    naive_ms / tiled_ms;


                // ------------------------------------------------
                // Print
                // ------------------------------------------------

                std::cout
                    << "    tile "
                    << std::setw(3)
                    << TILE
                    << " : "
                    << std::setw(10)
                    << tiled_ms
                    << " ms"
                    << "   "
                    << speedup
                    << "x"
                    << "   error = "
                    << max_error
                    << "\n";


                // ------------------------------------------------
                // CSV
                // ------------------------------------------------

                csv
                    << N << ","
                    << K << ","
                    << TILE << ","
                    << naive_ms << ","
                    << tiled_ms << ","
                    << speedup << ","
                    << max_error
                    << "\n";
            }
        }
    }


    csv.close();


    std::cout
        << "\n=============================================\n"
        << "Benchmark complete.\n"
        << "Results saved to: conv_results.csv\n"
        << "=============================================\n";

    return 0;
}