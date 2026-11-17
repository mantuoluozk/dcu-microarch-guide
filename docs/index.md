# 海光 DCU 微架构与算子优化指南

欢迎来到 **DCU Microarchitecture & Kernel Optimization Guide**！  
本站专注于以**理论 + 真实海光 DCU（BW100 / `gfx936`）实测 + ISA 汇编 + `hipprof` 硬件计数器**的方式，系统拆解 GPU/DCU 微架构与大模型核心算子优化。

---

## 内容导航

| 模块 | 内容简介 | 入口 |
|---|---|---|
| **微架构原理** | 从 Grid/Block/Wave64/CU、EXEC 分支掩码、LDS Bank 映射到 `s_waitcnt`、Occupancy 与 Roofline 模型 | [DCU 微架构与 Kernel 执行机制](architecture/dcu-microarch.md) |
| **实战教程（持续更新）** | 第 0~8 课手把手实操：HIP 微基准、Triton 融合算子、`lightop` 对比与 `hipprof` PMC 抓取 | [BW100 微架构实战教程](tutorials/bw100-hands-on.md) |
| **实验运行指南** | `experiments/` 全部源码的编译、ISA 导出与 `hipprof` 抓包方法 | [实验手册与编译指南](tutorials/experiment-guide.md) |
| **实测数据报告** | BW100 硬件规格、软件栈链路与 10+ 个实验的完整量化结果 | [BW100 微架构与算子实测报告](reports/bw100-benchmark-report.md) |

---

## BW100（`gfx936`）实测核心指标速览

- **硬件规模：** 64 CUs，Wave64，每 CU 最多 40 Waves，64 KB LDS，8 MB L2 Cache，64 GB HBM。
- **实测 Roofline 基准：**
  - HBM 实测峰值带宽（`float4` 合并访存）：**`1364.93 GB/s`**
  - FP16 实测算力（`rocBLAS` `MT256x256x16`）：**`256.80 TFLOPS`**
  - 实测 Roofline 拐点（Ridge Point）：**`~188.1 FLOPs/Byte`**
- **典型微架构现象实测：**
  - **32-way LDS Bank Conflict：** 导致 `10.4 亿` 次硬件冲突与 **19.62x** 减速；XOR Swizzle 在零额外 LDS 下完全清零冲突。
  - **Memory-bound 算子融合（RMSNorm）：** Triton Fused 达到 `1293.82 GB/s`（**19.21x** 提速，逼近实测 HBM 峰值 `94.8%`）。
  - **`hipprof` PMC 诊断 Triton GEMM：** 捕获 `9900 万` 周期 `SQ_WAIT_INST_LDS`、`96B` 寄存器 Spill 与 8 倍 `VMEM_RD` 指令，精准定位与 `rocBLAS` 的差距。

---

## 配套实验代码

所有可复现实验代码均位于仓库的 [`experiments/`](https://github.com/mantuoluozk/dcu-microarch-guide/tree/main/experiments) 目录，涵盖：
- `01 ~ 06`：HIP 底层微基准（设备探测、合并访存、LDS Bank Conflict、SIMT 分支发散、Occupancy vs ILP、Wave Shuffle）。
- `07 ~ 10`：Triton DSL（RMSNorm、GEMM `tl.dot`）、软件栈探测与 `lightop` 专项算子对比。
- `parse_hipprof_pmc.py`：`hipprof --pmc --pmc-type 3` 硬件计数器 CSV 自动解析工具。
