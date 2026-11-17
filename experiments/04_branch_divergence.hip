#include "common.h"

constexpr int BRANCH_ITERS = 4096;

template <int MODE>
__global__ void branch_divergence_kernel(const float* __restrict__ in,
                                         float* __restrict__ out,
                                         int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    float x = in[gid];
    float y = 1.0001f;

    for (int i = 0; i < BRANCH_ITERS; ++i) {
        if constexpr (MODE == 0) {
            // Baseline: no branch
            x = x * 1.00001f + y;
        } else if constexpr (MODE == 1) {
            // Wave-uniform branch: all lanes in a Wavefront take the same path (SCC scalar branch)
            if (((threadIdx.x / warpSize) & 1) == 0) {
                x = x * 1.00001f + y;
            } else {
                x = x * 0.99999f - y;
            }
        } else if constexpr (MODE == 2) {
            // 2-way Lane-divergent branch: half lanes active in each branch (EXEC masked)
            if ((threadIdx.x & 1) == 0) {
                x = x * 1.00001f + y;
            } else {
                x = x * 0.99999f - y;
            }
        } else if constexpr (MODE == 3) {
            // 4-way Lane-divergent branch: 1/4 lanes active per branch
            int lane_mod = threadIdx.x & 3;
            if (lane_mod == 0) {
                x = x * 1.00001f + y;
            } else if (lane_mod == 1) {
                x = x * 0.99999f - y;
            } else if (lane_mod == 2) {
                x = x * 1.00002f + y * 0.5f;
            } else {
                x = x * 0.99998f - y * 0.5f;
            }
        }
    }
    out[gid] = x;
}

int main() {
    constexpr int N = 1 << 20;
    constexpr int block_size = 256;
    constexpr int grid_size = (N + block_size - 1) / block_size;

    float *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(float)));
    HIP_CHECK(hipMemset(d_in, 0, N * sizeof(float)));

    std::printf("=== Exp 4: SIMT Branch Divergence & EXEC Mask (%d iters) ===\n", BRANCH_ITERS);

    auto run_mode = [&](const char* label, auto kernel_fn) {
        float ms = benchmark_kernel([&]() {
            hipLaunchKernelGGL(kernel_fn, dim3(grid_size), dim3(block_size), 0, 0,
                               d_in, d_out, N);
        });
        std::printf("  %-42s : %8.4f ms\n", label, ms);
        return ms;
    };

    float t0 = run_mode("0. No branch (Baseline)", branch_divergence_kernel<0>);
    float t1 = run_mode("1. Wave-uniform branch (SCC scalar)", branch_divergence_kernel<1>);
    float t2 = run_mode("2. 2-way Lane-divergent (EXEC serialized)", branch_divergence_kernel<2>);
    float t3 = run_mode("3. 4-way Lane-divergent (EXEC serialized)", branch_divergence_kernel<3>);

    std::printf("  Wave-uniform overhead vs Baseline         : %.2fx\n", t1 / t0);
    std::printf("  2-way divergence slowdown vs Uniform      : %.2fx\n", t2 / t1);
    std::printf("  4-way divergence slowdown vs Uniform      : %.2fx\n", t3 / t1);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
