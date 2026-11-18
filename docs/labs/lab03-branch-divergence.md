# E03 · SIMT 分支发散与编译器 Predicated If-Conversion

**研究问题**　同一个 wave 的 lane 走不同分支时，ISA 和吞吐如何变化？

历史 BW100 记录对比了同向分支、lane 发散和谓词化。观察结果取决于分支体、掩码和编译器实际生成的指令。

[适用环境](../baselines/bw100-gfx936.md) · [复现命令](#复现命令)

## BW100 结果

| 分支模式 | 平均耗时 (ms) | 相对 Baseline 倍率 |
|---|---:|---:|
| `0. No branch (Baseline)` | `1.6253` | `1.00x` |
| `1. Wave-uniform branch` | `4.0450` | `2.49x` |
| `2. 2-way Lane-divergent` | `4.0451` | `2.49x` |
| `3. 4-way Lane-divergent` | **`11.2303`** | **`6.91x`**（相对 2-way 慢 `2.78x`） |

## 复现命令

```bash
cd experiments
make 04_branch_divergence && ./04_branch_divergence
```

> **源码：** [04_branch_divergence.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/04_branch_divergence.hip)  
> **关联理论：** [1.2 尾部 wave 与 EXEC / VCC / SCC](../02-compute/cu-and-simt.md#一个-wave-怎么执行)

## 实验原理：硬件如何处理 Wavefront 内的 `if / else`？

Wave64 中 64 个 Lane 共享一条指令流（PC）：

- **Wave-uniform 分支：** 全 Wave 所有 Lane 走相同分支，走标量比较 `SCC` + `s_cbranch_scc`，未命中路径直接跳过。
- **Lane-divergent 分支：** 同一个 Wave 内部分 Lane 走 `if`、另一部分走 `else`。硬件通过 `VCC` 更新 64-bit `EXEC` 掩码，**先执行 `if` 路径，再翻转 `EXEC` 执行 `else` 路径**，形成串行化开销。
- **编译器的 Predicated If-Conversion 优化：** 当 `if / else` 分支体极短（如只有 1~2 条指令）时，编译器往往不生成跳转指令，而是**两条路径都计算**，再用 `v_cndmask_b32` 根据 `VCC` 选出最终值。

## 现象拆解与重要教训

1. **为什么 Mode 1 与 Mode 2 耗时完全相等（`4.0450 ms` vs `4.0451 ms`）？**
   - 分支体内只有 1 条 FMA，LLVM 后端直接对其做了 **Predicated If-Conversion**：两路 FMA 都算出来，再用 `v_cndmask_b32` 选值，完全没有走 `s_cbranch_scc`。
   - 同时，由于 `warpSize` 在 HIP 中未作为编译期常量，`threadIdx.x / warpSize` 未能被静态证明为 Wave-uniform 标量条件。若要严格触发 `s_cbranch_scc` 标量分支，可使用 `blockIdx.x & 1` 并把分支体写得足够重。
2. **为什么 Mode 3 飙升到 `11.2303 ms`？**
   - 4 路分支超出了简单 `v_cndmask` 的化简阈值，触发了真实的 `EXEC` 掩码保存、修改与 4 路串行执行，有效算力跌至约 `1/4 ~ 1/3`。

::: details 查看源码与实现细节

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

:::

::: details 查看原始终端输出

```text
$ ./04_branch_divergence
=== Exp 4: SIMT Branch Divergence & EXEC Mask (4096 iters) ===
  0. No branch (Baseline)                    :   1.6253 ms
  1. Wave-uniform branch (SCC scalar)        :   4.0450 ms
  2. 2-way Lane-divergent (EXEC serialized)  :   4.0451 ms
  3. 4-way Lane-divergent (EXEC serialized)  :  11.2303 ms
  Wave-uniform overhead vs Baseline         : 2.49x
  2-way divergence slowdown vs Uniform      : 1.00x
  4-way divergence slowdown vs Uniform      : 2.78x
```

:::
