#include "common.h"

// Both arms of the vectorization test launch N/4 threads and copy N floats.
__global__ void copy_scalar_x4(const float* __restrict__ src,
                               float* __restrict__ dst,
                               int plane_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < plane_size) {
        #pragma unroll
        for (int plane = 0; plane < 4; ++plane) {
            int offset = idx + plane * plane_size;
            dst[offset] = src[offset];
        }
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
                                   unsigned column_bits,
                                   unsigned row_bits) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned column = idx & ((1u << column_bits) - 1);
    unsigned row = idx >> column_bits;
    // Rotate index bits. Each address has exactly one writer in every variant.
    unsigned offset = (column << row_bits) | row;
    dst[offset] = src[offset];
}

__global__ void init_input(float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        src[idx] = static_cast<float>(idx / 16 + 1);
    }
}

__global__ void check_copy(const float* src, const float* dst, int n, int* failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && src[idx] != dst[idx]) {
        atomicExch(failed, 1);
    }
}

int main() {
    constexpr unsigned index_bits = 26;
    constexpr int elements = 1 << index_bits;
    constexpr int block_size = 256;
    constexpr size_t bytes = static_cast<size_t>(elements) * sizeof(float);
    constexpr int scalar_grid = elements / block_size;
    constexpr int vector_grid = scalar_grid / 4;

    float *src = nullptr, *dst = nullptr;
    int* failed = nullptr;
    HIP_CHECK(hipMalloc(&src, bytes));
    HIP_CHECK(hipMalloc(&dst, bytes));
    HIP_CHECK(hipMalloc(&failed, sizeof(int)));
    hipLaunchKernelGGL(init_input, dim3(scalar_grid), dim3(block_size), 0, 0, src, elements);
    HIP_CHECK(hipGetLastError());

    auto measure = [&](const char* label, auto launch) {
        HIP_CHECK(hipMemset(dst, 0, bytes));
        launch();
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipMemset(failed, 0, sizeof(int)));
        hipLaunchKernelGGL(check_copy, dim3(scalar_grid), dim3(block_size), 0, 0,
                           src, dst, elements, failed);
        HIP_CHECK(hipGetLastError());
        int host_failed = 0;
        HIP_CHECK(hipMemcpy(&host_failed, failed, sizeof(int), hipMemcpyDeviceToHost));
        if (host_failed) {
            std::fprintf(stderr, "%s failed full-buffer validation\n", label);
            std::exit(EXIT_FAILURE);
        }
        for (int sample = 1; sample <= 5; ++sample) {
            float ms = benchmark_kernel(launch);
            HIP_CHECK(hipGetLastError());
            double gbs = (2.0 * bytes / 1e9) / (ms / 1e3);
            std::printf("%s,%d,%.6f,%.2f,pass\n", label, sample, ms, gbs);
        }
    };

    std::puts("variant,sample,ms,effective_GB_per_s,validation");
    measure("scalar_x4", [&]() {
        hipLaunchKernelGGL(copy_scalar_x4, dim3(vector_grid), dim3(block_size), 0, 0,
                           src, dst, elements / 4);
    });
    measure("float4", [&]() {
        hipLaunchKernelGGL(copy_float4_coalesced, dim3(vector_grid), dim3(block_size), 0, 0,
                           reinterpret_cast<const float4*>(src),
                           reinterpret_cast<float4*>(dst), elements / 4);
    });
    for (unsigned bits = 0; bits <= 5; ++bits) {
        char label[32];
        std::snprintf(label, sizeof(label), "stride_%u", 1u << bits);
        measure(label, [&]() {
            hipLaunchKernelGGL(copy_float_strided, dim3(scalar_grid), dim3(block_size), 0, 0,
                               src, dst, index_bits - bits, bits);
        });
    }

    HIP_CHECK(hipFree(failed));
    HIP_CHECK(hipFree(src));
    HIP_CHECK(hipFree(dst));
}
