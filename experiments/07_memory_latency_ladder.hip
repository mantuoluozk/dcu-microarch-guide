#include "common.h"
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

// Launch one thread for a dependent load chain. LaneIndexed keeps the address
// lane-dependent in the compiler, so its ISA can be compared with the uniform path.
template <bool LaneIndexed>
__global__ void global_chase_kernel(const int* __restrict__ next_idx,
                                    int* __restrict__ out,
                                    unsigned long long* __restrict__ out_cycles,
                                    int steps) {
    int idx = LaneIndexed ? threadIdx.x : 0;
    // Warmup cache for working sets that fit in L1/L2
    for (int i = 0; i < steps; ++i) {
        idx = next_idx[idx];
    }

    asm volatile("s_waitcnt 0" ::: "memory");
    unsigned long long t0 = clock64();
    for (int i = 0; i < steps; ++i) {
        idx = next_idx[idx];
    }
    asm volatile("s_waitcnt 0" ::: "memory");
    unsigned long long t1 = clock64();

    out[0] = idx;
    out_cycles[0] = t1 - t0;
}

// Pointer-chasing kernel in LDS (on-chip shared memory)
__global__ void lds_chase_kernel(int* __restrict__ out,
                                 unsigned long long* __restrict__ out_cycles,
                                 int n_elems,
                                 int steps) {
    __shared__ int s_next[4096]; // 16 KB LDS ring
    for (int i = 0; i < n_elems; ++i) {
        // Stride-17 prime step mod n_elems avoids sequential prefetcher patterns
        s_next[i] = (i + 17) % n_elems;
    }
    __syncthreads();

    int idx = 0;
    for (int i = 0; i < steps; ++i) {
        idx = s_next[idx];
    }
    asm volatile("s_waitcnt 0" ::: "memory");
    unsigned long long t0 = clock64();
    for (int i = 0; i < steps; ++i) {
        idx = s_next[idx];
    }
    asm volatile("s_waitcnt 0" ::: "memory");
    unsigned long long t1 = clock64();

    out[0] = idx;
    out_cycles[0] = t1 - t0;
}

int main(int argc, char** argv) {
    // Optional size (KiB; 0 runs the full sweep), seed, and probe spacing (bytes).
    const size_t sizes_kib[] = {
        4, 8, 12, 16, 18, 20, 24, 28, 32, 48, 64, 128, 256, 384, 512, 768,
        1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 32768, 65536
    };
    unsigned long options[] = {0, 42, 64};
    if (argc > 5) {
        std::fprintf(stderr, "Usage: %s [size_KiB_or_0 [seed [spacing_bytes [scalar|lane]]]]\n", argv[0]);
        return EXIT_FAILURE;
    }
    for (int i = 1; i < argc && i <= 3; ++i) {
        char* end = nullptr;
        options[i - 1] = std::strtoul(argv[i], &end, 10);
        if (end == argv[i] || *end || argv[i][0] == '-') {
            std::fprintf(stderr, "Invalid numeric argument: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }
    const auto [requested_kib, seed, spacing] = options;
    if (requested_kib > 65536 || seed > 0xffffffffUL ||
        (spacing != 32 && spacing != 64 && spacing != 128)) {
        std::fprintf(stderr, "Size must be 0-65536 KiB, seed uint32, spacing 32/64/128 bytes.\n");
        return EXIT_FAILURE;
    }

    const bool lane_indexed = argc == 5 && std::strcmp(argv[4], "lane") == 0;
    if (argc == 5 && !lane_indexed && std::strcmp(argv[4], "scalar") != 0) {
        std::fprintf(stderr, "Address mode must be scalar or lane.\n");
        return EXIT_FAILURE;
    }

    int* d_out = nullptr;
    unsigned long long* d_cycles = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cycles, sizeof(unsigned long long)));

    std::puts("path,size_KiB,seed,spacing_bytes,hops,sample,clock64_ticks_per_hop");
    auto measure = [&](const char* path, size_t kib, unsigned long spacing_bytes, int steps, auto launch) {
        for (int sample = 1; sample <= 5; ++sample) {
            launch();
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
            int endpoint = -1;
            unsigned long long ticks = 0;
            HIP_CHECK(hipMemcpy(&endpoint, d_out, sizeof(endpoint), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&ticks, d_cycles, sizeof(ticks), hipMemcpyDeviceToHost));
            if (endpoint != 0 || ticks == 0) {
                std::fprintf(stderr, "%s: ring did not close or clock64 did not advance\n", path);
                std::exit(EXIT_FAILURE);
            }
            std::printf("%s,%zu,%lu,%lu,%d,%d,%.3f\n", path, kib, seed, spacing_bytes,
                        steps, sample, static_cast<double>(ticks) / steps);
        }
    };

    if (!requested_kib) {
        constexpr int lds_elements = 4096;
        measure("lds", 16, 0, lds_elements, [&]() {
            hipLaunchKernelGGL(lds_chase_kernel, dim3(1), dim3(1), 0, 0,
                               d_out, d_cycles, lds_elements, lds_elements);
        });
    }

    auto run_global = [&](size_t kib) {
        const size_t total_ints = kib * 1024 / sizeof(int);
        const int spacing_ints = static_cast<int>(spacing / sizeof(int));
        const int nodes = static_cast<int>(total_ints / spacing_ints);
        // Both phases traverse whole rings, with at least 4096 dependent loads each.
        const int steps = nodes * ((4096 + nodes - 1) / nodes);
        std::vector<int> permutation(nodes);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
        std::shuffle(permutation.begin(), permutation.end(), rng);

        std::vector<int> buffer(total_ints, 0);
        for (int i = 0; i < nodes; ++i) {
            buffer[permutation[i] * spacing_ints] = permutation[(i + 1) % nodes] * spacing_ints;
        }

        int* device_buffer = nullptr;
        HIP_CHECK(hipMalloc(&device_buffer, total_ints * sizeof(int)));
        HIP_CHECK(hipMemcpy(device_buffer, buffer.data(), total_ints * sizeof(int), hipMemcpyHostToDevice));
        measure(lane_indexed ? "global_lane" : "global_scalar", kib, spacing, steps, [&]() {
            if (lane_indexed) {
                hipLaunchKernelGGL((global_chase_kernel<true>), dim3(1), dim3(1), 0, 0,
                                   device_buffer, d_out, d_cycles, steps);
            } else {
                hipLaunchKernelGGL((global_chase_kernel<false>), dim3(1), dim3(1), 0, 0,
                                   device_buffer, d_out, d_cycles, steps);
            }
        });
        HIP_CHECK(hipFree(device_buffer));
    };

    if (requested_kib) {
        run_global(requested_kib);
    } else {
        for (size_t kib : sizes_kib) {
            run_global(kib);
        }
    }
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_cycles));
}
