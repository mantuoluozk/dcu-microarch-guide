#include "common.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

// Pointer-chasing kernel in Global Memory (L1 -> L2 -> HBM)
// Runs on 1 thread to eliminate TLP hiding and expose pure round-trip latency.
__global__ void global_chase_kernel(const int* __restrict__ next_idx,
                                    int* __restrict__ out,
                                    unsigned long long* __restrict__ out_cycles,
                                    int steps) {
    int idx = 0;
    // Warmup cache for working sets that fit in L1/L2
    for (int i = 0; i < steps; ++i) {
        idx = next_idx[idx];
    }

    unsigned long long t0 = clock64();
    for (int i = 0; i < steps; ++i) {
        idx = next_idx[idx];
    }
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
    unsigned long long t0 = clock64();
    for (int i = 0; i < steps; ++i) {
        idx = s_next[idx];
    }
    unsigned long long t1 = clock64();

    out[0] = idx;
    out_cycles[0] = t1 - t0;
}

int main() {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    double core_ghz = prop.clockRate / 1e6; // clockRate is in kHz

    int* d_out = nullptr;
    unsigned long long* d_cycles = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cycles, sizeof(unsigned long long)));

    std::printf("=== Exp 8: Memory Latency Ladder (Pointer-Chasing, %s @ %.2f GHz) ===\n",
                prop.name, core_ghz);

    // 1. LDS Latency
    constexpr int LDS_ELEMS = 4096;
    constexpr int STEPS = 4096;
    hipLaunchKernelGGL(lds_chase_kernel, dim3(1), dim3(1), 0, 0, d_out, d_cycles, LDS_ELEMS, STEPS);
    HIP_CHECK(hipDeviceSynchronize());
    unsigned long long h_cyc = 0;
    HIP_CHECK(hipMemcpy(&h_cyc, d_cycles, sizeof(h_cyc), hipMemcpyDeviceToHost));
    double lds_cyc = static_cast<double>(h_cyc) / STEPS;
    std::printf("  %-18s : %7.1f cycles (%6.2f ns)\n", "LDS (16 KB)", lds_cyc, lds_cyc / core_ghz);

    // 2. Global Memory Working-Set Sweep (L1 -> L2 -> HBM)
    const size_t sizes_kb[] = {4, 16, 32, 64, 256, 1024, 4096, 8192, 16384, 65536};
    std::mt19937 rng(42);

    for (size_t kb : sizes_kb) {
        // Stride of 64 bytes (16 ints) ensures each hop hits a distinct cache line
        constexpr int LINE_INTS = 16;
        size_t total_ints = (kb * 1024) / sizeof(int);
        size_t num_lines = total_ints / LINE_INTS;

        std::vector<int> perm(num_lines);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);

        std::vector<int> h_buf(total_ints, 0);
        for (size_t i = 0; i < num_lines; ++i) {
            int cur = perm[i] * LINE_INTS;
            int nxt = perm[(i + 1) % num_lines] * LINE_INTS;
            h_buf[cur] = nxt;
        }

        int* d_buf = nullptr;
        HIP_CHECK(hipMalloc(&d_buf, total_ints * sizeof(int)));
        HIP_CHECK(hipMemcpy(d_buf, h_buf.data(), total_ints * sizeof(int), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(global_chase_kernel, dim3(1), dim3(1), 0, 0,
                           d_buf, d_out, d_cycles, STEPS);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(&h_cyc, d_cycles, sizeof(h_cyc), hipMemcpyDeviceToHost));

        double cyc = static_cast<double>(h_cyc) / STEPS;
        std::printf("  Working Set %5zu KB : %7.1f cycles (%6.2f ns)\n",
                    kb, cyc, cyc / core_ghz);

        HIP_CHECK(hipFree(d_buf));
    }

    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_cycles));
    return 0;
}
