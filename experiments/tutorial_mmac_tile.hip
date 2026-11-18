#include "common.h"

using Half4 = _Float16 __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));

constexpr int width = 16;
constexpr int elements = width * width;

__global__ void multiply_tile(float* output) {
    const int axis = threadIdx.x % width;
    const int start = 4 * (threadIdx.x / width);
    Half4 a, b;
    for (int j = 0; j < 4; ++j) {
        const int k = start + j;
        a[j] = 2 * axis + k;       // A[axis, k]
        b[j] = (k == axis);    // B[k, axis]，单位矩阵
    }
    Float4 partial = {1, 1, 1, 1};
    Float4 result = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, partial);
    for (int j = 0; j < 4; ++j) {
        const int col = threadIdx.x / width + 4 * j;
        output[axis * width + col] = result[j];
    }
    return;
}

int main() {
    float actual[elements];
    float* device_output = nullptr;
    HIP_CHECK(hipMalloc(&device_output, sizeof(actual)));
    HIP_CHECK(hipMemset(device_output, 0, sizeof(actual)));
    hipLaunchKernelGGL(multiply_tile, dim3(1), dim3(64), 0, 0, device_output);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(actual, device_output, sizeof(actual), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(device_output));

    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            float expected = 1;
            for (int k = 0; k < width; ++k) {
                expected += (2 * row + k) * (k == col);
            }
            if (actual[row * width + col] != expected) {
                std::fprintf(stderr, "D[%d,%d]: got %g, expected %g\n",
                             row, col, actual[row * width + col], expected);
                return EXIT_FAILURE;
            }
        }
    }
    std::printf("D[2,3] = %g | 256 outputs verified\n", actual[2 * width + 3]);
}
