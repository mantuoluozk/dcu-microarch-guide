#include "common.h"

// Low-ILP: single dependent accumulator per thread (relies on TLP / Occupancy to hide latency)
__global__ void low_ilp_kernel(const float* __restrict__ a,
                               float* __restrict__ out,
                               int n,
                               int steps) {
    extern __shared__ char dummy_lds[];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    if (gid >= n) return;

    float acc = 0.0f;
    int idx = gid;
    for (int s = 0; s < steps; ++s) {
        acc += a[idx];
        idx += stride;
        if (idx >= n) idx -= n;
    }
    if (threadIdx.x == 0 && dummy_lds[0] == 123) {
        out[0] = 0.0f;
    }
    out[gid] = acc;
}

// High-ILP: 4 independent loads issued before accumulating (ILP hides latency even at lower Occupancy)
__global__ void high_ilp_kernel(const float* __restrict__ a,
                                float* __restrict__ out,
                                int n,
                                int steps) {
    extern __shared__ char dummy_lds[];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    if (gid >= n) return;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int idx0 = gid;
    int idx1 = (gid + stride) % n;
    int idx2 = (gid + 2 * stride) % n;
    int idx3 = (gid + 3 * stride) % n;

    for (int s = 0; s < steps; s += 4) {
        float v0 = a[idx0];
        float v1 = a[idx1];
        float v2 = a[idx2];
        float v3 = a[idx3];
        acc0 += v0;
        acc1 += v1;
        acc2 += v2;
        acc3 += v3;
        idx0 = (idx0 + 4 * stride) % n;
        idx1 = (idx1 + 4 * stride) % n;
        idx2 = (idx2 + 4 * stride) % n;
        idx3 = (idx3 + 4 * stride) % n;
    }
    if (threadIdx.x == 0 && dummy_lds[0] == 123) {
        out[0] = 0.0f;
    }
    out[gid] = (acc0 + acc1) + (acc2 + acc3);
}

int main() {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    constexpr int N = 1 << 24; // 64 MB
    constexpr int STEPS = 32;
    constexpr int block_size = 256;
    const int grid_size = prop.multiProcessorCount * 16; // Enough blocks to populate all CUs

    float *d_a = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_a, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, grid_size * block_size * sizeof(float)));
    HIP_CHECK(hipMemset(d_a, 1, N * sizeof(float)));

    std::printf("=== Exp 5: Occupancy vs ILP Latency Hiding (CU=%d, Block=%d) ===\n",
                prop.multiProcessorCount, block_size);

    const size_t lds_sizes_kb[] = {0, 16, 32, 48};
    for (size_t kb : lds_sizes_kb) {
        size_t lds_bytes = kb * 1024;
        if (lds_bytes > prop.sharedMemPerBlock) continue;

        int max_blocks_low = 0, max_blocks_high = 0;
        HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_low, low_ilp_kernel, block_size, lds_bytes));
        HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_high, high_ilp_kernel, block_size, lds_bytes));

        float t_low = benchmark_kernel([&]() {
            hipLaunchKernelGGL(low_ilp_kernel, dim3(grid_size), dim3(block_size),
                               lds_bytes, 0, d_a, d_out, N, STEPS);
        });
        float t_high = benchmark_kernel([&]() {
            hipLaunchKernelGGL(high_ilp_kernel, dim3(grid_size), dim3(block_size),
                               lds_bytes, 0, d_a, d_out, N, STEPS);
        });

        int waves_low = max_blocks_low * (block_size / prop.warpSize);
        int waves_high = max_blocks_high * (block_size / prop.warpSize);
        std::printf("  LDS=%2zu KB | Low-ILP: %2d Blks/CU (%2d Waves) -> %7.4f ms | "
                    "High-ILP: %2d Blks/CU (%2d Waves) -> %7.4f ms\n",
                    kb, max_blocks_low, waves_low, t_low,
                    max_blocks_high, waves_high, t_high);
    }

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
