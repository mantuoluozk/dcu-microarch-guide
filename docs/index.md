# 海光 DCU 微架构与算子优化指南

欢迎来到 **DCU Microarchitecture & Kernel Optimization Guide**！  
本站服务于**“面向下一代海光 DCU 的微架构研究与大模型推理优化”**研究专题（周期：2026.11 — 2027.12），采用**“一实验一章、源码级逐行解读、实测数据 + ISA + `hipprof` 硬件计数器相互印证”**的实战方式编写。

---

## 全站章节导航

### 零、研究专题实施方案
- [面向下一代海光 DCU 的微架构研究与大模型推理优化专题实施方案](00-plan/research-plan.md) — 项目定位、双主线技术路线、2026~2027 里程碑及四大研究方向。

### 一、微架构原理篇
1. [1.1 整体架构、执行模型与软件栈链路](01-overview/software-stack.md) — Grid/Block/Wave64/CU 映射与 PyTorch → Triton/`lightop` → HIP/DTK → HSA 全链路（含图文笔记图解）。
2. [1.2 线程映射、SIMT 分支发散与计算单元](02-compute/cu-and-simt.md) — `EXEC`/`VCC`/`SCC` 分支掩码、Scheduler、SALU、VALU、SFU 与 MMAC 矩阵引擎。
3. [1.3 存储体系、合并访存与异步等待](03-memory/memory-hierarchy.md) — VGPR/SGPR、LDS 32-Bank 映射与 Swizzle、L1/L2 一致性、HBM Coalescing、`s_waitcnt` 与 Wave Shuffle。
4. [1.4 Kernel 生命周期、Occupancy 与 Roofline](04-scheduling/occupancy-and-roofline.md) — 资源分配粒度与悬崖效应、TLP vs ILP 延迟隐藏、Roofline 拐点定量判断。

### 二、BW100 实战实验篇（一实验一章，含完整源码与解读）
- [实验导览：环境探测、ISA 导出与 `hipprof` 工具链](labs/lab-setup.md)
- [实验 1：全局内存合并访存与 `float4` 向量化](labs/lab01-memory-coalescing.md)（实测 HBM 峰值 `1364.93 GB/s`）
- [实验 2：LDS Bank Conflict 与 Padding / XOR Swizzle 消除](labs/lab02-lds-bank-conflict.md)（32-way 冲突慢 `19.62x`、`10.4 亿`次硬件冲突）
- [实验 3：SIMT 分支发散与 Predicated If-Conversion](labs/lab03-branch-divergence.md)（短分支谓词化 vs 4-way `EXEC` 串行化）
- [实验 4：Occupancy 悬崖与 TLP / ILP 延迟隐藏](labs/lab04-occupancy-ilp.md)（32→4 Waves 性能退化 `2.71x`）
- [实验 5：Wavefront Shuffle 规约 vs LDS 规约](labs/lab05-wave-shuffle-reduction.md)（`__shfl_down` 省去 8 次 Barrier）
- [实验 6：Memory-bound 算子融合（PyTorch vs `lightop` vs Triton RMSNorm）](labs/lab06-triton-lightop-rmsnorm.md)（融合提速 `19.21x`）
- [实验 7：Triton GEMM vs `rocBLAS` 与 `hipprof` PMC 根因破案](labs/lab07-triton-gemm-vs-rocblas.md)（抓出 `96B` Spill 与 `9900 万` 周期 LDS Wait）

### 三、算子案例与后续路线
- [案例 1：LSTM Persistent MMAC Kernel 硬件协同拆解](cases/lstm-persistent-kernel.md)
- [案例 2：大模型关键算子迁移与 Kernel 优化分析清单](cases/llm-operators-playbook.md)
- [后续实验路线图与更新日志](roadmap.md)
