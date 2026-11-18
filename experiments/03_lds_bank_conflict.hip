#include "common.h"
#include <cstring>
#include <vector>

constexpr unsigned BLOCK_SIZE = 256;
constexpr unsigned GROUP_SIZE = 32;
constexpr unsigned GROUP_WORDS = 32 * 33;
constexpr unsigned LDS_ITERS = 2048;

struct Layout {
    const char* name;
    unsigned degree;
};

constexpr Layout layouts[] = {
    {"degree1", 1}, {"degree2", 2}, {"degree4", 4},
    {"degree8", 8}, {"degree16", 16}, {"degree32", 32},
    {"padding33", 0}, {"xor", 0},
};

__host__ __device__ unsigned lds_offset(unsigned tid, unsigned layout, unsigned degree) {
    const unsigned lane = tid % GROUP_SIZE;
    const unsigned base = (tid / GROUP_SIZE) * GROUP_WORDS;
    constexpr unsigned column = 7;
    if (layout == 6) return base + lane * 33 + column;
    if (layout == 7) return base + lane * 32 + (lane ^ column);
    const unsigned banks_used = GROUP_SIZE / degree;
    return base + (lane / banks_used) * 32 + (lane % banks_used + column) % 32;
}

__global__ void lds_read_kernel(unsigned* out, unsigned layout, unsigned degree) {
    constexpr unsigned words = (BLOCK_SIZE / GROUP_SIZE) * GROUP_WORDS;
    __shared__ volatile unsigned tile[words];
    const unsigned tid = threadIdx.x;
    for (unsigned i = tid; i < words; i += BLOCK_SIZE) {
        tile[i] = i + 1;
    }
    __syncthreads();

    const unsigned offset = lds_offset(tid, layout, degree);
    unsigned sum = 0;
    #pragma unroll 4
    for (unsigned i = 0; i < LDS_ITERS; ++i) {
        sum += tile[offset];
    }
    out[blockIdx.x * BLOCK_SIZE + tid] = sum;
}

int main(int argc, char** argv) {
    constexpr unsigned count = sizeof(layouts) / sizeof(layouts[0]);
    unsigned first = 0, last = count;
    const bool profiling = argc == 3 && std::strcmp(argv[2], "profile") == 0;
    if (argc > 1) {
        while (first < count && std::strcmp(layouts[first].name, argv[1]) != 0) {
            ++first;
        }
        if (first == count || (argc != 2 && !profiling)) {
            std::fprintf(stderr, "Usage: %s [degree1|degree2|degree4|degree8|degree16|degree32|padding33|xor [profile]]\n",
                         argv[0]);
            return EXIT_FAILURE;
        }
        last = first + 1;
    }

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    const unsigned blocks = prop.multiProcessorCount * 8;
    std::vector<unsigned> result(blocks * BLOCK_SIZE);
    unsigned* out = nullptr;
    HIP_CHECK(hipMalloc(&out, result.size() * sizeof(result[0])));

    std::puts("variant,sample,ms,validation");
    for (unsigned index = first; index < last; ++index) {
        const auto& layout = layouts[index];
        auto launch = [&]() {
            hipLaunchKernelGGL(lds_read_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
                               out, index, layout.degree);
            HIP_CHECK(hipGetLastError());
        };
        launch();
        HIP_CHECK(hipMemcpy(result.data(), out, result.size() * sizeof(result[0]),
                            hipMemcpyDeviceToHost));
        for (size_t i = 0; i < result.size(); ++i) {
            const unsigned expected = LDS_ITERS * (lds_offset(i % BLOCK_SIZE, index, layout.degree) + 1);
            if (result[i] != expected) {
                std::fprintf(stderr, "%s: validation failed at %zu\n", layout.name, i);
                return EXIT_FAILURE;
            }
        }
        if (profiling) {
            for (int sample = 0; sample < 5; ++sample) launch();
            HIP_CHECK(hipDeviceSynchronize());
        } else {
            for (int sample = 1; sample <= 5; ++sample) {
                std::printf("%s,%d,%.6f,pass\n", layout.name, sample, benchmark_kernel(launch));
            }
        }
    }
    HIP_CHECK(hipFree(out));
}
