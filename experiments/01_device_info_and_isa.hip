#include "common.h"
#include <vector>

__global__ void sample_fma_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  float alpha,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = a[idx];
        float vb = b[idx];
        c[idx] = alpha * va + vb;
    }
}

int main() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    std::printf("Detected %d DCU/HIP device(s)\n\n", count);

    for (int dev = 0; dev < count; ++dev) {
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, dev));
        std::printf("=== Device %d: %s ===\n", dev, prop.name);
        std::printf("  gcnArchName                : %s\n", prop.gcnArchName);
        std::printf("  CU count (multiProcessor)  : %d\n", prop.multiProcessorCount);
        std::printf("  warpSize (Wavefront size)  : %d\n", prop.warpSize);
        std::printf("  Max threads / CU           : %d (Max Waves/CU = %d)\n",
                    prop.maxThreadsPerMultiProcessor,
                    prop.maxThreadsPerMultiProcessor / prop.warpSize);
        std::printf("  Max threads / Block        : %d\n", prop.maxThreadsPerBlock);
        std::printf("  LDS (sharedMemPerBlock)    : %zu KB\n", prop.sharedMemPerBlock / 1024);
        std::printf("  Regs / Block (32-bit)      : %d\n", prop.regsPerBlock);
        std::printf("  L2 Cache Size              : %d KB\n", prop.l2CacheSize / 1024);
        std::printf("  Global Mem                 : %.2f GB\n",
                    static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
        std::printf("  Memory Bus Width           : %d-bit\n", prop.memoryBusWidth);
        std::printf("  Memory Clock Rate          : %.2f MHz\n", prop.memoryClockRate / 1000.0);
        std::printf("  Core Clock Rate            : %.2f MHz\n", prop.clockRate / 1000.0);
        double theo_bw_gbs = 2.0 * (prop.memoryClockRate * 1e3) * (prop.memoryBusWidth / 8.0) / 1e9;
        std::printf("  Theoretical Peak HBM BW*   : %.1f GB/s (DDR approx)\n\n", theo_bw_gbs);
    }

    HIP_CHECK(hipSetDevice(0));
    constexpr int N = 1 << 20;
    std::vector<float> h_a(N, 1.5f), h_b(N, 2.0f), h_c(N, 0.0f);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    HIP_CHECK(hipMalloc(&d_a, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_c, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), N * sizeof(float), hipMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    float ms = benchmark_kernel([&]() {
        hipLaunchKernelGGL(sample_fma_kernel, grid, block, 0, 0, d_a, d_b, d_c, 3.0f, N);
    });
    HIP_CHECK(hipMemcpy(h_c.data(), d_c, N * sizeof(float), hipMemcpyDeviceToHost));
    std::printf("Sample FMA check: c[0]=%.2f (expected 6.50), avg time=%.4f ms\n", h_c[0], ms);

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    return 0;
}
