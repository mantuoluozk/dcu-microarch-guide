# 实验 6：Triton GEMM（`tl.dot`）vs `rocBLAS` 与 `hipprof` 根因破案

> **源码：** [08_triton_gemm_mmac.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/08_triton_gemm_mmac.py)  
> **关联章节：** [第 2 章：MMAC 矩阵引擎](../02-compute/cu-and-simt.md) & [第 3 章：`s_waitcnt` 异步流水](../03-memory/memory-hierarchy.md)

---

## 1. 实验设计

在 `4096 × 4096 × 4096` FP16 矩阵乘（总计算量 `137.44 GFLOPs`）上，对比 `torch.matmul`（底层 rocBLAS）与不同 `(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)` 配置的 Triton GEMM，并用 `hipprof --pmc --pmc-type 3` 抓取硬件计数器定位差距根因。

---

## 2. BW100 实测算力与静态资源对比

| 实现 / 配置 | 耗时 (ms) | 算力 (TFLOPS) | LDS (B) | VGPR / SGPR | `matrix_ops` | `s_waitcnt` |
|---|---:|---:|---:|---:|---:|---:|
| **`torch.matmul` (rocBLAS)** | **`0.535`** | **`256.80`** | - | - | - | - |
| Triton (`BM=128, BN=128, BK=32, w=4, s=1`) | `1.416` | **`97.08`** | `8704` | `143 / 27` | `32` | `27` |
| Triton (`BM=128, BN=128, BK=32, w=4, s=2`) | `1.500` | `91.61` | `16384` | `160 / 36` | `64` | `32` |
| Triton (`BM=128, BN=128, BK=64, w=8, s=2`) | `2.465` | `55.76` | `32768` | `128 / 44` | `64` | `56` |
| Triton (`BM=128, BN=256, BK=32, w=8, s=2`) | `2.566` | `53.56` | `24576` | `128 / 44` | `64` | `65` |

---

## 3. `hipprof` PMC 硬件计数器破案（为什么 Triton 慢 4.8x？）

通过 `hipprof --pmc --pmc-type 3 -o prof_out/exp08 python3 advanced/08_triton_gemm_mmac.py` 抓包，拿到 `rocBLAS`（`Cijk_Ailk_Bljk_HBH_MT256x256x16...`）与 Triton `gemm_kernel`（`BM=128, BN=256, BK=32`）的硬件级对账单：

| 关键硬件 PMC 指标 | `rocBLAS` (`MT256x256x16`) | Triton `gemm_kernel` | 诊断结论 |
|---|---:|---:|---|
| 平均耗时 | **`530.78 us`** | `2556.83 us` | 慢 `4.82x` |
| VGPR / SGPR | `212 / 112` | `128 / 48` | - |
| **寄存器溢出 (`Scratch`)** | **`0 B`** | **`96 B`（发生 Spill！）** | **Triton 产生了 96B/thread 显存栈溢出** |
| 全局读指令 (`SQ_INSTS_VMEM_RD`) | **`1,048,576`** | `8,429,568` | Triton 多发射了 **8x** 全局加载指令 |
| LDS 指令 (`SQ_INSTS_LDS`) | **`4,194,304`** | `21,266,432` | Triton 多做了 **5x** LDS 读写 |
| **`SQ_LDS_BANK_CONFLICT`** | `8,388,608` | **`34,078,720`** | Triton LDS 冲突是 `rocBLAS` 的 **4.06x** |
| **`SQ_WAIT_INST_LDS`** | **`3,934,304`** | **`99,050,034`** | **Triton 有 9900 万周期卡在等 LDS（25.2x）！** |
| L2 命中率 (`TCC_HIT %`) | `81.82%` | **`91.17%`** | `GROUP_M=8` 的 L2 Swizzle 非常有效 |

**破案总结：**
1. L2 命中率（`91.17%`）证明外部访存并不是瓶颈。
2. Triton 慢的真正元凶是：**`96 B` 寄存器 Spill**、**8 倍 `VMEM_RD` 指令**，以及 **高达 9900 万周期的 `SQ_WAIT_INST_LDS`（5 倍 LDS 指令 + 4 倍 Bank Conflict）**。
3. 从 `rocBLAS` 的 Kernel 名 `MT256x256x16` 可以看出：在 BW100 上采用 **`BK=16`（小 K 步长降低 LDS/VGPR 压力）+ `kpack` / `waves_per_eu` 调优**，是下一步把 Triton GEMM 拉高到 180+ TFLOPS 的关键方向。
