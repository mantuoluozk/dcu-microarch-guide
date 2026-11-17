# 实验 3：SIMT 分支发散与编译器 Predicated If-Conversion

> **源码：** [04_branch_divergence.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/04_branch_divergence.hip)  
> **关联理论：** [第 2 章：SIMT 分支发散与 EXEC / VCC / SCC](../02-compute/cu-and-simt.md)

---

## 1. 实验原理：硬件如何处理 Wavefront 内的 `if / else`？

Wave64 中 64 个 Lane 共享一条指令流（PC）：
- **Wave-uniform 分支：** 全 Wave 所有 Lane 走相同分支，走标量比较 `SCC` + `s_cbranch_scc`，未命中路径直接跳过。
- **Lane-divergent 分支：** 同一个 Wave 内部分 Lane 走 `if`、另一部分走 `else`。硬件通过 `VCC` 更新 64-bit `EXEC` 掩码，**先执行 `if` 路径，再翻转 `EXEC` 执行 `else` 路径**，形成串行化开销。
- **编译器的 Predicated If-Conversion 优化：** 当 `if / else` 分支体极短（如只有 1~2 条指令）时，编译器往往不生成跳转指令，而是**两条路径都计算**，再用 `v_cndmask_b32` 根据 `VCC` 选出最终值。

---

## 2. 完整源码与逐段解读

```cpp title="experiments/04_branch_divergence.hip"
#include "common.h"

constexpr int BRANCH_ITERS = 4096;

template <int MODE>
__global__ void branch_divergence_kernel(const float* __restrict__ in,
                                         float* __restrict__ out,
                                         int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    float x = in[gid];
    float y = 1.0001f;

    for (int i = 0; i < BRANCH_ITERS; ++i) {
        if constexpr (MODE == 0) {
            // 基准：无分支
            x = x * 1.00001f + y;
        } else if constexpr (MODE == 1) {
            // Wave-uniform 分支设计：按 Wavefront 编号分奇偶
            if (((threadIdx.x / warpSize) & 1) == 0) {
                x = x * 1.00001f + y;
            } else {
                x = x * 0.99999f - y;
            }
        } else if constexpr (MODE == 2) {
            // 2-way Lane-divergent：同一 Wave 内奇偶 Lane 分叉
            if ((threadIdx.x & 1) == 0) {
                x = x * 1.00001f + y;
            } else {
                x = x * 0.99999f - y;
            }
        } else if constexpr (MODE == 3) {
            // 4-way Lane-divergent：同一 Wave 内 4 路分叉
            int lane_mod = threadIdx.x & 3;
            if (lane_mod == 0) {
                x = x * 1.00001f + y;
            } else if (lane_mod == 1) {
                x = x * 0.99999f - y;
            } else if (lane_mod == 2) {
                x = x * 1.00002f + y * 0.5f;
            } else {
                x = x * 0.99998f - y * 0.5f;
            }
        }
    }
    out[gid] = x;
}
```

---

## 3. 复现命令与 BW100 实测结果

```bash
cd /exp/dcu_microarch_exp
make 04_branch_divergence && ./04_branch_divergence
```

| 分支模式 | 平均耗时 (ms) | 相对 Baseline 倍率 |
|---|---:|---:|
| `0. No branch (Baseline)` | `1.6253` | `1.00x` |
| `1. Wave-uniform branch` | `4.0450` | `2.49x` |
| `2. 2-way Lane-divergent` | `4.0451` | `2.49x` |
| `3. 4-way Lane-divergent` | **`11.2303`** | **`6.91x`**（相对 2-way 慢 `2.78x`） |

---

## 4. 现象拆解与重要教训

1. **为什么 Mode 1 与 Mode 2 耗时完全相等（`4.0450 ms` vs `4.0451 ms`）？**
   - 分支体内只有 1 条 FMA，LLVM 后端直接对其做了 **Predicated If-Conversion**：两路 FMA 都算出来，再用 `v_cndmask_b32` 选值，完全没有走 `s_cbranch_scc`。
   - 同时，由于 `warpSize` 在 HIP 中未作为编译期常量，`threadIdx.x / warpSize` 未能被静态证明为 Wave-uniform 标量条件。若要严格触发 `s_cbranch_scc` 标量分支，可使用 `blockIdx.x & 1` 并把分支体写得足够重。
2. **为什么 Mode 3 飙升到 `11.2303 ms`？**
   - 4 路分支超出了简单 `v_cndmask` 的化简阈值，触发了真实的 `EXEC` 掩码保存、修改与 4 路串行执行，有效算力跌至约 `1/4 ~ 1/3`。
