#include "common.h"
#include <vector>
#include <cmath>

constexpr int BLOCK_SIZE = 256;

__global__ void reduce_lds_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int cols) {
    __shared__ float sdata[BLOCK_SIZE];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int c = tid; c < cols; c += BLOCK_SIZE) {
        sum += in[row * cols + c];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = sdata[0];
    }
}

__global__ void reduce_shuffle_kernel(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int cols) {
    // 256 threads / 64 (Wave64) = max 4 waves; allocate 8 to also support Wave32 safely
    __shared__ float wave_sums[8];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % warpSize;
    int wave_id = tid / warpSize;
    int num_waves = BLOCK_SIZE / warpSize;

    float sum = 0.0f;
    for (int c = tid; c < cols; c += BLOCK_SIZE) {
        sum += in[row * cols + c];
    }

    // Intra-wave reduction via register shuffle (no LDS, no barrier)
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (lane == 0) {
        wave_sums[wave_id] = sum;
    }
    __syncthreads();

    if (wave_id == 0) {
        float final_sum = (lane < num_waves) ? wave_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            final_sum += __shfl_down(final_sum, offset);
        }
        if (lane == 0) {
            out[row] = final_sum;
        }
    }
}

int main() {
    constexpr int ROWS = 8192;
    constexpr int COLS = 4096; // 8192 * 4096 * 4B = 128 MB
    constexpr size_t in_bytes = static_cast<size_t>(ROWS) * COLS * sizeof(float);
    constexpr size_t out_bytes = static_cast<size_t>(ROWS) * sizeof(float);

    std::vector<float> h_in(static_cast<size_t>(ROWS) * COLS, 1.0f);
    std::vector<float> h_out_lds(ROWS, 0.0f), h_out_shfl(ROWS, 0.0f);

    float *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, in_bytes));
    HIP_CHECK(hipMalloc(&d_out, out_bytes));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), in_bytes, hipMemcpyHostToDevice));

    std::printf("=== Exp 6: Row Reduction (LDS+Barrier vs Wave Shuffle, %dx%d) ===\n",
                ROWS, COLS);

    float t_lds = benchmark_kernel([&]() {
        hipLaunchKernelGGL(reduce_lds_kernel, dim3(ROWS), dim3(BLOCK_SIZE), 0, 0,
                           d_in, d_out, COLS);
    });
    HIP_CHECK(hipMemcpy(h_out_lds.data(), d_out, out_bytes, hipMemcpyDeviceToHost));

    float t_shfl = benchmark_kernel([&]() {
        hipLaunchKernelGGL(reduce_shuffle_kernel, dim3(ROWS), dim3(BLOCK_SIZE), 0, 0,
                           d_in, d_out, COLS);
    });
    HIP_CHECK(hipMemcpy(h_out_shfl.data(), d_out, out_bytes, hipMemcpyDeviceToHost));

    std::printf("  Pure LDS + Barriers   : %8.4f ms (out[0]=%.1f)\n", t_lds, h_out_lds[0]);
    std::printf("  Wave Shuffle + 1 LDS  : %8.4f ms (out[0]=%.1f)\n", t_shfl, h_out_shfl[0]);
    std::printf("  Speedup               : %.2fx\n", t_lds / t_shfl);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
