#include "common.h"

constexpr int TILE = 32;
constexpr int LDS_ITERS = 2048;

template <int MODE>
__global__ void lds_bank_conflict_kernel(float* __restrict__ out) {
    // MODE 0: row-wise (no conflict)
    // MODE 1: col-wise (32-way bank conflict)
    // MODE 2: col-wise with +1 padding (conflict-free)
    // MODE 3: col-wise with XOR swizzle (conflict-free, zero memory overhead)
    __shared__ float tile_std[TILE][TILE];
    __shared__ float tile_pad[TILE][TILE + 1];

    int tx = threadIdx.x % TILE;
    int ty = threadIdx.x / TILE; // 0..7 for block=256, or 0..31 for block=1024

    if constexpr (MODE == 2) {
        tile_pad[ty][tx] = static_cast<float>(threadIdx.x) * 0.001f;
    } else {
        tile_std[ty][tx] = static_cast<float>(threadIdx.x) * 0.001f;
    }
    __syncthreads();

    float acc = 0.0f;
    // Dependent chain prevents compiler from hoisting LDS reads out of the loop
    for (int i = 0; i < LDS_ITERS; ++i) {
        int r = (ty + i) & (TILE - 1);
        float v = 0.0f;
        if constexpr (MODE == 0) {
            v = tile_std[r][tx];
            tile_std[r][tx] = v + 1e-6f;
        } else if constexpr (MODE == 1) {
            v = tile_std[tx][r];
            tile_std[tx][r] = v + 1e-6f;
        } else if constexpr (MODE == 2) {
            v = tile_pad[tx][r];
            tile_pad[tx][r] = v + 1e-6f;
        } else if constexpr (MODE == 3) {
            v = tile_std[tx][r ^ tx];
            tile_std[tx][r ^ tx] = v + 1e-6f;
        }
        acc += v;
    }

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    out[gid] = acc;
}

int main() {
    // Use 1024 threads per block (32x32 tile = 16 Wavefronts on Wave64)
    // Launch enough blocks to saturate all CUs
    constexpr int block_size = 1024;
    constexpr int grid_size = 256;
    constexpr int total_threads = block_size * grid_size;

    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, total_threads * sizeof(float)));

    std::printf("=== Exp 3: LDS Bank Conflict (32x32 Tile, %d LDS r/w iters) ===\n", LDS_ITERS);

    auto run_mode = [&](const char* label, auto kernel_fn) {
        float ms = benchmark_kernel([&]() {
            hipLaunchKernelGGL(kernel_fn, dim3(grid_size), dim3(block_size), 0, 0, d_out);
        });
        std::printf("  %-38s : %8.4f ms\n", label, ms);
        return ms;
    };

    float t0 = run_mode("0. Row-wise (No conflict)", lds_bank_conflict_kernel<0>);
    float t1 = run_mode("1. Col-wise (32-way Bank Conflict)", lds_bank_conflict_kernel<1>);
    float t2 = run_mode("2. Col-wise + Padding [32][33]", lds_bank_conflict_kernel<2>);
    float t3 = run_mode("3. Col-wise + XOR Swizzle [tx][r^tx]", lds_bank_conflict_kernel<3>);

    std::printf("  Slowdown (32-way vs No-conflict)      : %.2fx\n", t1 / t0);
    std::printf("  Speedup  (Padding vs Conflict)        : %.2fx\n", t1 / t2);
    std::printf("  Speedup  (Swizzle vs Conflict)        : %.2fx\n", t1 / t3);

    HIP_CHECK(hipFree(d_out));
    return 0;
}
