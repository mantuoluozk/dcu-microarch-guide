# 海光 DCU 微架构与算子优化指南

欢迎来到 **DCU Microarchitecture & Kernel Optimization Guide**！  
本站服务于**“面向下一代海光 DCU 的微架构研究与大模型推理优化”**研究专题，采用**“先微架构原理 → 再 Profiling 工具链 → 再底层/高级算子实战 → 最后大模型推理优化”**的四段式体系编写。

---

## 全站四大部分导航

### [专题实施方案](00-plan/research-plan.md)
- 项目定位、双主线技术路线、四大研究方向与 2027 年核心交付目标。

### 第一部分：DCU 微架构原理
1. [1.1 整体架构、执行模型与软件栈链路](01-overview/software-stack.md) — Grid/Block/Wave64/CU 映射与 PyTorch → Triton/`lightop` → HIP/DTK → HSA 全链路。
2. [1.2 线程映射、SIMT 分支与计算单元（含 MMAC）](02-compute/cu-and-simt.md) — `EXEC`/`VCC`/`SCC` 分支掩码、Scheduler、SALU、VALU、SFU 与 MMAC 矩阵引擎。
3. [1.3 存储体系、合并访存与异步等待](03-memory/memory-hierarchy.md) — VGPR/SGPR、LDS 32-Bank 映射与 Swizzle、L1/L2 一致性、HBM Coalescing、`s_waitcnt` 与 Wave Shuffle。
4. [1.4 Kernel 生命周期、Occupancy 与 Roofline 模型](04-scheduling/occupancy-and-roofline.md) — 资源分配粒度与悬崖效应、TLP vs ILP 延迟隐藏、Roofline 拐点定量判断。
5. [1.5 微架构自动化分析与建模（待补充）](05-modeling/auto-analysis.md) — 自动化微架构体检、Roofline 诊断与 Skill 沉淀规划。

### 第二部分：Profiling 与性能分析工具链
1. [2.1 工具链全景与软件栈探测](profiling/01-toolchain-overview.md) — `torch.profiler`、`hipprof`、ISA 导出与容器软件栈/硬件规格探测。
2. [2.2 硬件 PMC 采集与自研解析工具](profiling/02-hipprof-and-parser.md) — `hipprof --pmc` 指标速查与自研 `parse_hipprof_pmc.py` 解析工具。
3. [2.3 ISA 反汇编分析与仿真（DCUSIM 待补充）](profiling/03-isa-and-sim.md) — `.s` 汇编 5 大检查点与 DCUSIM 预留章节。

### 第三部分：底层微基准与算子实战
- [实验导览与环境准备](labs/lab-setup.md)
- **HIP 底层微基准：**
  - [实验 1：全局内存合并访存与 `float4` 向量化](labs/lab01-memory-coalescing.md)（`1364.93 GB/s`）
  - [实验 2：LDS Bank Conflict 与 Padding / XOR Swizzle 消除](labs/lab02-lds-bank-conflict.md)（32-way 冲突慢 `19.62x`）
  - [实验 3：SIMT 分支发散与 Predicated If-Conversion](labs/lab03-branch-divergence.md)
  - [实验 4：Occupancy 悬崖与 TLP / ILP 延迟隐藏](labs/lab04-occupancy-ilp.md)
  - [实验 5：Wavefront Shuffle 规约 vs LDS 规约](labs/lab05-wave-shuffle-reduction.md)
  - [实验 6：用“寻宝游戏”测出 L1 / L2 / HBM 真实延迟阶梯](labs/lab08-memory-latency-ladder.md)（`57 → 216 → 407 cycles`）
  - [实验 7：用 HCU Builtin 直呼 MMAC 矩阵引擎](labs/lab09-mmac-builtin-and-valu.md)（`333.49 TFLOPS` vs VALU `20.21 TFLOPS`）
- **Triton / `lightop` 高级算子：**
  - [实验 8：Memory-bound 算子融合（PyTorch vs `lightop` vs Triton RMSNorm）](labs/lab06-triton-lightop-rmsnorm.md)（提速 `19.21x`）
  - [实验 9：Triton GEMM vs `rocBLAS` 与 `hipprof` PMC 根因破案](labs/lab07-triton-gemm-vs-rocblas.md)
  - [实验 10：Triton GEMM 调优实战（待补充）](labs/lab10-triton-gemm-tuning.md)

### 第四部分：大模型推理优化实战
1. [4.1 vLLM / SGLang 推理热点分析（待补充）](llm-opt/01-framework-profiling.md)
2. [4.2 GEMM 与 MoE 算子优化（待补充）](llm-opt/02-gemm-and-moe.md)
3. [4.3 Attention 与 KV Cache 优化（待补充）](llm-opt/03-attention-kvcache.md)
4. [4.4 案例参考：LSTM Persistent MMAC Kernel 拆解](cases/lstm-persistent-kernel.md)
5. [4.5 通用算子优化六步法与研发反馈清单](cases/llm-operators-playbook.md)
