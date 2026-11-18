#include "common.h"
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

constexpr unsigned NODES = 64;
constexpr unsigned BLOCK_SIZE = 256;

__global__ void initialize_rings(unsigned* next, unsigned threads, unsigned chains) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned node = index / threads;
    const unsigned gid = index % threads;
    next[index] = ((node + chains) % NODES) * threads + gid;
}

template <unsigned Chains>
__global__ void chase_rings(const unsigned* __restrict__ next, unsigned* out, unsigned threads) {
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned index[Chains];
    #pragma unroll
    for (unsigned chain = 0; chain < Chains; ++chain) {
        index[chain] = chain * threads + gid;
    }
    #pragma unroll 1
    for (unsigned step = 0; step < NODES / Chains; ++step) {
        #pragma unroll
        for (unsigned chain = 0; chain < Chains; ++chain) {
            index[chain] = next[index[chain]];
        }
    }
    unsigned sum = 0;
    #pragma unroll
    for (unsigned chain = 0; chain < Chains; ++chain) {
        sum += index[chain];
    }
    out[gid] = sum;
}

template <unsigned Chains>
void run(unsigned blocks, unsigned lds_bytes, unsigned wave_size,
         unsigned* next, unsigned* out, std::vector<unsigned>& result, bool profiling) {
    const unsigned threads = blocks * BLOCK_SIZE;
    hipLaunchKernelGGL(initialize_rings, dim3(blocks * NODES), dim3(BLOCK_SIZE), 0, 0,
                       next, threads, Chains);
    HIP_CHECK(hipGetLastError());

    int blocks_per_cu = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu, chase_rings<Chains>, BLOCK_SIZE, lds_bytes));
    if (blocks_per_cu == 0) {
        std::fprintf(stderr, "No resident blocks for %u chains with %u bytes of LDS\n", Chains, lds_bytes);
        std::exit(EXIT_FAILURE);
    }

    auto launch = [&]() {
        hipLaunchKernelGGL((chase_rings<Chains>), dim3(blocks), dim3(BLOCK_SIZE),
                           lds_bytes, 0, next, out, threads);
        HIP_CHECK(hipGetLastError());
    };
    launch();
    HIP_CHECK(hipMemcpy(result.data(), out, result.size() * sizeof(result[0]),
                        hipMemcpyDeviceToHost));
    for (unsigned gid = 0; gid < threads; ++gid) {
        const unsigned expected = Chains * gid + threads * (Chains * (Chains - 1) / 2);
        if (result[gid] != expected) {
            std::fprintf(stderr, "%u chains: validation failed at %u\n", Chains, gid);
            std::exit(EXIT_FAILURE);
        }
    }

    if (profiling) {
        for (int sample = 0; sample < 5; ++sample) launch();
        HIP_CHECK(hipDeviceSynchronize());
    } else {
        const unsigned waves = blocks_per_cu * (BLOCK_SIZE / wave_size);
        for (int sample = 1; sample <= 5; ++sample) {
            const float ms = benchmark_kernel(launch);
            const double gb_per_s = (double(threads) * NODES * sizeof(unsigned) / 1e6) / ms;
            std::printf("%u,%u,%u,%u,%d,%.6f,%.3f,pass\n",
                        Chains, lds_bytes / 1024, blocks_per_cu, waves, sample, ms, gb_per_s);
        }
    }
}

int main(int argc, char** argv) {
    const unsigned lds_sizes[] = {0, 16, 32, 48};
    const unsigned chain_sizes[] = {1, 2, 4, 8};
    const bool profiling = argc == 4 && std::strcmp(argv[3], "profile") == 0;
    unsigned requested_chains = 0, requested_lds = 0;
    if (argc != 1) {
        int matched = 0;
        if (argc == 3 || profiling) {
            for (unsigned chains : chain_sizes) {
                if (std::to_string(chains) == argv[1]) {
                    requested_chains = chains;
                    ++matched;
                }
            }
            for (unsigned kb : lds_sizes) {
                if (std::to_string(kb) == argv[2]) {
                    requested_lds = kb;
                    ++matched;
                }
            }
        }
        if (matched != 2) {
            std::fprintf(stderr, "Usage: %s [1|2|4|8 0|16|32|48 [profile]]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    if (requested_chains && requested_lds * 1024 > prop.sharedMemPerBlock) {
        std::fprintf(stderr, "Requested dynamic LDS exceeds the device limit\n");
        return EXIT_FAILURE;
    }
    const unsigned blocks = prop.multiProcessorCount * 16;
    const unsigned threads = blocks * BLOCK_SIZE;
    unsigned *next = nullptr, *out = nullptr;
    HIP_CHECK(hipMalloc(&next, size_t(threads) * NODES * sizeof(unsigned)));
    HIP_CHECK(hipMalloc(&out, threads * sizeof(unsigned)));
    std::vector<unsigned> result(threads);
    std::puts("chains,lds_KiB,max_blocks_per_CU,max_waves_per_CU,sample,ms,logical_read_GB_per_s,validation");

    for (unsigned kb : lds_sizes) {
        if (kb * 1024 > prop.sharedMemPerBlock || (requested_chains && kb != requested_lds)) continue;
        auto measure = [&](auto chains) {
            if (!requested_chains || requested_chains == chains) {
                run<decltype(chains)::value>(blocks, kb * 1024, prop.warpSize, next, out, result, profiling);
            }
        };
        measure(std::integral_constant<unsigned, 1>{});
        measure(std::integral_constant<unsigned, 2>{});
        measure(std::integral_constant<unsigned, 4>{});
        measure(std::integral_constant<unsigned, 8>{});
    }
    HIP_CHECK(hipFree(out));
    HIP_CHECK(hipFree(next));
}
