#include "common.h"

#include <vector>

__global__ void tutorial_add(const float* a, const float* b, float* c, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        c[i] = a[i] + b[i];
    }
    return;
}

int main() {
    constexpr int count = 1000;
    constexpr int threads = 256;
    constexpr int blocks = (count + threads - 1) / threads;
    constexpr float untouched = -1.0f;

    std::vector<float> a(count), b(count), c(blocks * threads, untouched);
    for (int i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
    const size_t input_bytes = a.size() * sizeof(float);
    const size_t output_bytes = c.size() * sizeof(float);
    HIP_CHECK(hipMalloc(&device_a, input_bytes));
    HIP_CHECK(hipMalloc(&device_b, input_bytes));
    HIP_CHECK(hipMalloc(&device_c, output_bytes));
    HIP_CHECK(hipMemcpy(device_a, a.data(), input_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_b, b.data(), input_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_c, c.data(), output_bytes, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(tutorial_add, dim3(blocks), dim3(threads), 0, 0,
                       device_a, device_b, device_c, count);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(c.data(), device_c, output_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(device_c));
    HIP_CHECK(hipFree(device_b));
    HIP_CHECK(hipFree(device_a));

    for (int i = 0; i < blocks * threads; ++i) {
        const float expected = i < count ? a[i] + b[i] : untouched;
        if (c[i] != expected) {
            std::fprintf(stderr, "Mismatch at output index %d\n", i);
            return EXIT_FAILURE;
        }
    }
    hipDeviceProp_t device{};
    HIP_CHECK(hipGetDeviceProperties(&device, 0));
    std::printf("%s %s | %d blocks x %d threads | wave size %d | %d outputs verified, %d untouched\n",
                device.name, device.gcnArchName, blocks, threads, device.warpSize, count,
                blocks * threads - count);
}
