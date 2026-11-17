# 海光 DCU 微架构与算子优化指南

欢迎来到 **DCU Microarchitecture & Kernel Optimization Guide**！  
本站采用**“一章一主题、零冗余对照”**的编排方式，结合真实 **海光 BW100 (`gfx936`)** 硬件实测、AMDGCN ISA 汇编与 `hipprof` 硬件 PMC 计数器，系统拆解 DCU 微架构与大模型算子优化。

---

## 全站章节导航

### 一、微架构原理篇
1. [第 1 章：整体架构、执行模型与软件栈链路](01-overview/software-stack.md) — Grid/Block/Wave64/CU 映射与 PyTorch → Triton/`lightop` → HIP/DTK → HSA 全链路。
2. [第 2 章：线程映射、SIMT 分支发散与计算单元](02-compute/cu-and-simt.md) — `EXEC`/`VCC`/`SCC` 分支掩码、Scheduler、SALU、VALU、SFU 与 MMAC 矩阵引擎。
3. [第 3 章：存储体系、合并访存与异步等待](03-memory/memory-hierarchy.md) — VGPR/SGPR、LDS 32-Bank 映射与 Swizzle、L1/L2 一致性、HBM Coalescing、`s_waitcnt` 与 Wave Shuffle。
4. [第 4 章：Kernel 生命周期、Occupancy 与 Roofline](04-scheduling/occupancy-and-roofline.md) — 资源分配粒度与悬崖效应、TLP vs ILP 延迟隐藏、Roofline 拐点定量判断。

### 二、BW100 实战实验篇
- [实验导览：环境探测、ISA 导出与 `hipprof` 工具链](labs/lab-setup.md)
- [实验 1：全局内存合并访存与 `float4` 向量化](labs/lab01-memory-coalescing.md)（实测 HBM 峰值 `1364.93 GB/s`）
- [实验 2：LDS Bank Conflict 与 Padding / XOR Swizzle 消除](labs/lab02-lds-bank-conflict.md)（32-way 冲突慢 `19.62x`、`10.4 亿`次硬件冲突）
- [实验 3：SIMT 分支发散与 Predicated If-Conversion](labs/lab03-branch-divergence.md)（短分支谓词化 vs 4-way `EXEC` 串行化）
- [实验 4：Occupancy 悬崖、ILP 与 Wave Shuffle 规约](labs/lab04-occupancy-and-shuffle.md)（32→4 Waves 性能退化 `2.71x`）
- [实验 5：Memory-bound 算子融合（PyTorch vs `lightop` vs Triton RMSNorm）](labs/lab05-triton-lightop-rmsnorm.md)（融合提速 `19.21x`）
- [实验 6：Triton GEMM vs `rocBLAS` 与 `hipprof` PMC 根因破案](labs/lab06-triton-gemm-vs-rocblas.md)（抓出 `96B` Spill 与 `9900 万` 周期 LDS Wait）

### 三、算子案例与后续路线
- [案例 1：LSTM Persistent MMAC Kernel 硬件协同拆解](cases/lstm-persistent-kernel.md)
- [案例 2：大模型关键算子迁移与 Kernel 优化分析清单](cases/llm-operators-playbook.md)
- [后续实验路线图与更新日志](roadmap.md)

---

## 配套实验代码

所有可复现实验源码均位于仓库的 [`experiments/`](https://github.com/mantuoluozk/dcu-microarch-guide/tree/main/experiments) 目录。
