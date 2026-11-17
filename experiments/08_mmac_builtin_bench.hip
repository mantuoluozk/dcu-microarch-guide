#include "common.h"

using float4_t = float __attribute__((ext_vector_type(4)));
using half4_t = _Float16 __attribute__((ext_vector_type(4)));

constexpr int ITERS = 4096;

// 1. VALU FP32 FMA Peak (8 independent accumulators per thread)
__global__ void valu_fp32_peak_kernel(float* __restrict__ out, float seed) {
    float a0 = seed + 0.1f, a1 = seed + 0.2f, a2 = seed + 0.3f, a3 = seed + 0.4f;
    float a4 = seed + 0.5f, a5 = seed + 0.6f, a6 = seed + 0.7f, a7 = seed + 0.8f;
    float b = 1.0001f, c = 0.0001f;

    for (int i = 0; i < ITERS; ++i) {
        a0 = a0 * b + c; a1 = a1 * b + c; a2 = a2 * b + c; a3 = a3 * b + c;
        a4 = a4 * b + c; a5 = a5 * b + c; a6 = a6 * b + c; a7 = a7 * b + c;
    }
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    out[gid] = (a0 + a1 + a2 + a3) + (a4 + a5 + a6 + a7);
}

// 2. HCU Builtin MMAC FP16 (16x16x16 -> FP32) Throughput (8 independent accumulators)
__global__ void mmac_fp16_peak_kernel(float* __restrict__ out, _Float16 seed) {
    half4_t a = {seed, seed, seed, seed};
    half4_t b = {(_Float16)1.0, (_Float16)1.0, (_Float16)1.0, (_Float16)1.0};
    float4_t c0 = {0, 0, 0, 0}, c1 = {0, 0, 0, 0}, c2 = {0, 0, 0, 0}, c3 = {0, 0, 0, 0};
    float4_t c4 = {0, 0, 0, 0}, c5 = {0, 0, 0, 0}, c6 = {0, 0, 0, 0}, c7 = {0, 0, 0, 0};

    for (int i = 0; i < ITERS; ++i) {
        c0 = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c0);
        c1 = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c1);
        c2 = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c2);
        c3 = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c3);
        c4 = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c4);
        c5 = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c5);
        c6 = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c6);
        c7 = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c7);
    }
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float4_t s = (c0 + c1 + c2 + c3) + (c4 + c5 + c6 + c7);
    out[gid] = s[0] + s[1] + s[2] + s[3];
}

// 3. Single MMAC instruction latency (1 wave, dependent chain)
__global__ void mmac_fp16_latency_kernel(float* __restrict__ out,
                                         unsigned long long* __restrict__ out_cycles,
                                         _Float16 seed) {
    half4_t a = {seed, seed, seed, seed};
    half4_t b = {(_Float16)1.0, (_Float16)1.0, (_Float16)1.0, (_Float16)1.0};
    float4_t c = {0, 0, 0, 0};

    unsigned long long t0 = clock64();
    for (int i = 0; i < ITERS; ++i) {
        c = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c);
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) {
        out[0] = c[0];
        out_cycles[0] = t1 - t0;
    }
}

int main() {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    constexpr int block_size = 256;
    const int grid_size = prop.multiProcessorCount * 16; // 1024 blocks -> 4096 Wave64
    const size_t total_threads = static_cast<size_t>(grid_size) * block_size;
    const size_t total_waves = total_threads / prop.warpSize;

    float* d_out = nullptr;
    unsigned long long* d_cyc = nullptr;
    HIP_CHECK(hipMalloc(&d_out, total_threads * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_cyc, sizeof(unsigned long long)));

    std::printf("=== Exp 9: VALU vs HCU Builtin MMAC Peak & Latency (%s, CU=%d) ===\n",
                prop.name, prop.multiProcessorCount);

    // VALU FP32 Peak: each thread does 8 FMAs (16 FLOPs) per iter
    float t_valu = benchmark_kernel([&]() {
        hipLaunchKernelGGL(valu_fp32_peak_kernel, dim3(grid_size), dim3(block_size), 0, 0, d_out, 1.0f);
    });
    double valu_flops = static_cast<double>(total_threads) * ITERS * 8 * 2;
    std::printf("  VALU FP32 Peak   : %8.4f ms | %7.2f TFLOPS\n",
                t_valu, (valu_flops / 1e12) / (t_valu / 1e3));

    // MMAC FP16 Peak: each Wave64 does 8 * (2 * 16 * 16 * 16 = 8192 FLOPs) per iter
    float t_mmac = benchmark_kernel([&]() {
        hipLaunchKernelGGL(mmac_fp16_peak_kernel, dim3(grid_size), dim3(block_size), 0, 0, d_out, (_Float16)0.5);
    });
    double mmac_flops = static_cast<double>(total_waves) * ITERS * 8 * (2.0 * 16 * 16 * 16);
    std::printf("  MMAC 16x16x16 FP16: %8.4f ms | %7.2f TFLOPS (%.2fx vs VALU FP32)\n",
                t_mmac, (mmac_flops / 1e12) / (t_mmac / 1e3), t_valu / t_mmac * (mmac_flops / valu_flops));

    // MMAC Latency (1 block, 1 wave)
    hipLaunchKernelGGL(mmac_fp16_latency_kernel, dim3(1), dim3(64), 0, 0, d_out, d_cyc, (_Float16)0.5);
    HIP_CHECK(hipDeviceSynchronize());
    unsigned long long h_cyc = 0;
    HIP_CHECK(hipMemcpy(&h_cyc, d_cyc, sizeof(h_cyc), hipMemcpyDeviceToHost));
    std::printf("  MMAC 16x16x16 Dependent Latency : %.2f cycles / instruction\n",
                static_cast<double>(h_cyc) / ITERS);

    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_cyc));
    return 0;
}
