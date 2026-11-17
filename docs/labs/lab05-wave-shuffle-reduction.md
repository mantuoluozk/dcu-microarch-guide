# 实验 5：Wavefront Shuffle 规约 vs LDS 规约

> **源码：** [06_shuffle_vs_lds.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/06_shuffle_vs_lds.hip)  
> **关联理论：** [第 3 章：Wavefront 内跨 Lane 通信（Shuffle / DPP）](../03-memory/memory-hierarchy.md)

---

## 1. 实验原理：什么时候根本不需要写回 LDS？

在 RMSNorm、LayerNorm、Softmax 等算子中，经常需要把一行数据规约成一个标量（Sum 或 Max）：
- **传统纯 LDS 规约：** 把线程部分和写入 `__shared__`，随后在 `128 → 64 → 32 → 16 → 8 → 4 → 2 → 1` 每一步都执行一次 `__syncthreads()`（对应 ISA 中的 `s_waitcnt lgkmcnt(0)` + `s_barrier`），同步开销极高。
- **Wavefront Shuffle + 极简 LDS：** 同一 Wave64 内的 64 个线程直接通过 `__shfl_down` 在寄存器级别完成 `32 → 16 → 8 → 4 → 2 → 1` 规约（零 LDS 读写、零 Barrier）；只有每个 Wave 的 `lane == 0` 把结果写进仅有几个 `float` 的 LDS，做 **1 次** `__syncthreads()` 后由 Wave 0 收尾。

---

## 2. 完整源码与逐段解读

```cpp title="experiments/06_shuffle_vs_lds.hip"
#include "common.h"
#include <vector>

constexpr int BLOCK_SIZE = 256;

// 方式 A：纯 LDS 树状规约（需要 8 次 __syncthreads()）
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
    if (tid == 0) out[row] = sdata[0];
}

// 方式 B：Wave 内 __shfl_down 寄存器规约 + 仅 1 次 LDS 汇总
__global__ void reduce_shuffle_kernel(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int cols) {
    __shared__ float wave_sums[8]; // 兼容 Wave64 (4 waves) 与 Wave32 (8 waves)
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % warpSize;
    int wave_id = tid / warpSize;
    int num_waves = BLOCK_SIZE / warpSize;

    float sum = 0.0f;
    for (int c = tid; c < cols; c += BLOCK_SIZE) {
        sum += in[row * cols + c];
    }

    // 1. Wave 内寄存器级规约（无 LDS、无 Barrier）
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    // 2. 仅每个 Wave 的 Lane 0 写入 LDS
    if (lane == 0) wave_sums[wave_id] = sum;
    __syncthreads();

    // 3. Wave 0 汇总全部 Wave 的部分和
    if (wave_id == 0) {
        float final_sum = (lane < num_waves) ? wave_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            final_sum += __shfl_down(final_sum, offset);
        }
        if (lane == 0) out[row] = final_sum;
    }
}
```

---

## 3. 复现命令与 BW100 实测数据

```bash
cd /exp/dcu_microarch_exp
make 06_shuffle_vs_lds && ./06_shuffle_vs_lds
```

| 规约实现（`8192 × 4096`，128 MB） | `__syncthreads()` 次数 | LDS 占用 | 平均耗时 (ms) | 校验值 `out[0]` |
|---|---:|---:|---:|---:|
| `Pure LDS + Barriers` | `9` | `1024 B` | `0.1357` | `4096.0` |
| **`Wave Shuffle + 1 LDS`** | **`1`** | **`32 B`** | **`0.1286`** | `4096.0` |

---

## 4. 现象拆解与工程结论

1. **为什么这里只有 `1.06x`（`~7 us`）差距？** 对 `8192 × 4096`（128 MB）的输入，绝大部分时间花在从 HBM 读入 128 MB 数据（按实测极限 `1365 GB/s` 算理论读取下限约 `0.094 ms`），因此省去 8 次 Barrier 带来的纯同步收益约为 `7 us`。
2. **何时 Shuffle 会拉开巨大差距？** 当 `cols` 较小（例如 Hidden Size = 128/256）或算子内部需要多轮高频规约（如 FlashAttention 的 Online Softmax）时，省去 LDS 读写与多轮 `s_barrier` 会带来极其可观的提速。
