# 实验 4：Occupancy 悬崖、ILP 延迟隐藏与 Wave Shuffle 规约

> **源码：** [05_occupancy_and_ilp.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/05_occupancy_and_ilp.hip) & [06_shuffle_vs_lds.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/06_shuffle_vs_lds.hip)  
> **关联章节：** [第 4 章：Occupancy 与延迟隐藏](../04-scheduling/occupancy-and-roofline.md)

---

## 1. Occupancy 悬崖与 TLP / ILP 延迟隐藏（`05_occupancy_and_ilp`）

固定 `Block = 256`（每 Block 4 个 Wave64），通过动态申请 `0 / 16 / 32 / 48 KB` LDS 人为压缩每 CU 驻留的 Block 与 Wavefront 数量，对比 `Low-ILP`（单路串行加载累加）与 `High-ILP`（4 路独立加载累加）：

| 动态 LDS 占用 | 每 CU 驻留 Block 数 | 每 CU 驻留 Wave 数（Occupancy） | `Low-ILP` (ms) | `High-ILP` (ms) |
|---|---:|---:|---:|---:|
| `0 KB` | `8` | `32` (`80%`) | **`0.0386`** | `0.0484` |
| `16 KB` | `4` | `16` (`40%`) | `0.0491` | `0.0501` |
| `32 KB` | `2` | `8` (`20%`) | `0.0806` | **`0.0779`** |
| `48 KB` | `1` | `4` (`10%`) | `0.1045` | `0.1218` |

**分析：**
1. `Low-ILP` 对 Occupancy 极为敏感：当驻留 Wave 从 `32` 跌到 `4` 时，调度器无法靠 TLP 掩盖 HBM 延迟，耗时恶化 **`2.71x`**。
2. 在 `8 Waves/CU`（32 KB LDS）甜点处，`High-ILP` 靠单 Wave 内部并发请求反超了 `Low-ILP`；而在 `0/48 KB` 处略慢，是因为该微基准中 4 路索引使用了额外的 `% n` 取模 VALU 指令。

---

## 2. Wavefront Shuffle vs LDS 规约（`06_shuffle_vs_lds`）

对 `8192 × 4096` 矩阵（128 MB）做按行求和规约：

| 规约实现 | 同步机制 | 耗时 (ms) | 校验值 `out[0]` |
|---|---|---:|---:|
| `Pure LDS + Barriers` | 8 次 `__syncthreads()` | `0.1357` | `4096.0` |
| **`Wave Shuffle + 1 LDS`** | `__shfl_down` + 1 次 `__syncthreads()` | **`0.1286`** | `4096.0` |

在 `8192 × 4096` 这种大矩阵下，整体耗时主要由 HBM 读入 128 MB 决定，省去 7 次 `s_barrier` 带来了约 `7 us`（`1.06x`）的纯同步收益；当单行较短或循环内高频规约时，寄存器级 Shuffle 收益会更加显著。
