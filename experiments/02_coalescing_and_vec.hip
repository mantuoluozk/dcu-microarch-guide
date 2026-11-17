#include "common.h"

__global__ void copy_float_coalesced(const float* __restrict__ src,
                                     float* __restrict__ dst,
                                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

__global__ void copy_float4_coalesced(const float4* __restrict__ src,
                                      float4* __restrict__ dst,
                                      int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        dst[idx] = src[idx];
    }
}

__global__ void copy_float_strided(const float* __restrict__ src,
                                   float* __restrict__ dst,
                                   int n,
                                   int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Modulo by power-of-two n keeps all accesses in-bounds while forcing strided pattern
        int s_idx = static_cast<int>((static_cast<long long>(idx) * stride) & (n - 1));
        dst[s_idx] = src[s_idx];
    }
}

int main() {
    // 64M floats = 256 MB per buffer (exceeds L2 cache)
    constexpr int N = 1 << 26;
    constexpr size_t bytes = static_cast<size_t>(N) * sizeof(float);

    float *d_src = nullptr, *d_dst = nullptr;
    HIP_CHECK(hipMalloc(&d_src, bytes));
    HIP_CHECK(hipMalloc(&d_dst, bytes));
    HIP_CHECK(hipMemset(d_src, 1, bytes));

    constexpr int block_size = 256;
    const int grid_f1 = (N + block_size - 1) / block_size;
    const int n4 = N / 4;
    const int grid_f4 = (n4 + block_size - 1) / block_size;

    auto report = [&](const char* name, float ms) {
        // Effective bandwidth = read + write = 2 * bytes
        double gbs = (2.0 * bytes / 1e9) / (ms / 1e3);
        std::printf("  %-26s : %8.4f ms | %8.2f GB/s\n", name, ms, gbs);
    };

    std::printf("=== Exp 2: Global Memory Coalescing & Vectorization (256MB copy) ===\n");
    report("Coalesced float  (32-bit)", benchmark_kernel([&]() {
        hipLaunchKernelGGL(copy_float_coalesced, dim3(grid_f1), dim3(block_size), 0, 0,
                           d_src, d_dst, N);
    }));

    report("Coalesced float4 (128-bit)", benchmark_kernel([&]() {
        hipLaunchKernelGGL(copy_float4_coalesced, dim3(grid_f4), dim3(block_size), 0, 0,
                           reinterpret_cast<const float4*>(d_src),
                           reinterpret_cast<float4*>(d_dst), n4);
    }));

    // Note: odd stride is coprime with power-of-two N, so it visits all elements without collision;
    // even strides test cache-line waste with power-of-two striding.
    const int strides[] = {2, 4, 8, 16, 32};
    for (int s : strides) {
        char label[64];
        std::snprintf(label, sizeof(label), "Strided float (stride=%d)", s);
        report(label, benchmark_kernel([&]() {
            hipLaunchKernelGGL(copy_float_strided, dim3(grid_f1), dim3(block_size), 0, 0,
                               d_src, d_dst, N, s);
        }));
    }

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));
    return 0;
}
