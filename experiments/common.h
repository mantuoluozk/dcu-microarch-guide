#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call)                                                      \
    do {                                                                     \
        hipError_t err = (call);                                             \
        if (err != hipSuccess) {                                             \
            std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, hipGetErrorString(err));                  \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

template <typename Func>
float benchmark_kernel(Func launch_fn, int warmup = 5, int iters = 20) {
    for (int i = 0; i < warmup; ++i) {
        launch_fn();
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch_fn();
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return total_ms / static_cast<float>(iters);
}
