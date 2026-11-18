#include "common.h"

#include <vector>

__global__ void tutorial_add(const float* a, const float* b, float* c, int count) {
    // blockIdx.x 决定这个 Block 的起始位置，threadIdx.x 给出线程在块内的偏移。
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 最后一个 Block 有多余线程，只让有效索引读写数组。
    if (i < count) {
        c[i] = a[i] + b[i];
    }
    return;
}

int main() {
    constexpr int count = 1000;
    constexpr int threads = 256;
    // 向上取整，让 Grid 覆盖全部元素。1000 个元素需要 4 个 Block。
    constexpr int blocks = (count + threads - 1) / threads;
    constexpr float untouched = -1.0f;

    // 输出数组预留到启动线程总数，并填入初始值，用来检查尾部没有被误写。
    std::vector<float> a(count), b(count), c(blocks * threads, untouched);
    for (int i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    // Host 上的数组先复制到 DCU 可访问的设备内存。
    float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
    const size_t input_bytes = a.size() * sizeof(float);
    const size_t output_bytes = c.size() * sizeof(float);
    HIP_CHECK(hipMalloc(&device_a, input_bytes));
    HIP_CHECK(hipMalloc(&device_b, input_bytes));
    HIP_CHECK(hipMalloc(&device_c, output_bytes));
    HIP_CHECK(hipMemcpy(device_a, a.data(), input_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_b, b.data(), input_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_c, c.data(), output_bytes, hipMemcpyHostToDevice));

    // 启动参数指定 Grid 的 Block 数和每个 Block 的线程数。
    // 两个 0 分别表示不额外申请动态 LDS、使用默认 stream。
    hipLaunchKernelGGL(tutorial_add, dim3(blocks), dim3(threads), 0, 0,
                       device_a, device_b, device_c, count);
    HIP_CHECK(hipGetLastError());
    // Host 等待 Kernel 执行完成，再把结果复制回 CPU 检查。
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(c.data(), device_c, output_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(device_c));
    HIP_CHECK(hipFree(device_b));
    HIP_CHECK(hipFree(device_a));

    // 有效位置必须等于 CPU 算出的和，尾部的初始值必须保持原样。
    for (int i = 0; i < blocks * threads; ++i) {
        const float expected = i < count ? a[i] + b[i] : untouched;
        if (c[i] != expected) {
            std::fprintf(stderr, "Mismatch at output index %d\n", i);
            return EXIT_FAILURE;
        }
    }
    // wave 宽度从设备属性查询，本例没有在 launch 时设置它。
    hipDeviceProp_t device{};
    HIP_CHECK(hipGetDeviceProperties(&device, 0));
    std::printf("%s %s | %d blocks x %d threads | wave size %d | %d outputs verified, %d untouched\n",
                device.name, device.gcnArchName, blocks, threads, device.warpSize, count,
                blocks * threads - count);
}
