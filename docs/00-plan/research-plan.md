# 面向下一代海光 DCU 的微架构研究与大模型推理优化专题实施方案

## 核心推进节奏（三个重点）

1. **2026 年 11 月 — 2027 年上半年：** 利用现有 DCU 建立微架构研究、Kernel 开发、Profiling 和性能建模能力，为下一代 DCU 做技术储备。
2. **2027 年年中 — 年底：** 下一代 DCU 到位后，迅速开展硬件特性验证、关键算子适配优化及 SGLang / vLLM 端到端性能优化。
3. **贯穿整个项目：** 通过实际测试发现当前硬件和软件栈的性能瓶颈，向 DCU 架构、编译器和算子研发部门反馈问题与改进建议，建立双向协同机制。

---

## 一、项目背景、定位与总体技术路线

### 1.1 项目背景与定位

随着大模型推理逐步向长上下文、大参数量、MoE 架构和高并发场景发展，对 GPU 的矩阵计算能力、存储带宽、访存效率、执行调度和多卡协同能力提出了更高要求。现有工作更多集中在模型适配、算子调用和系统级调优，对 GPU 微架构、指令执行、计算访存流水及硬件瓶颈的系统性研究仍需加强。

本专题定位为**面向现有及下一代海光 DCU 的底层性能研究与优化项目**：以 GPU 微架构为基础，以大模型推理关键算子为研究对象，以 Profiling、性能建模和 Kernel 优化为技术手段，以 SGLang / vLLM 推理框架为应用验证平台。

### 1.2 总体技术路线（双主线并行）

```text
DCU 微架构学习与分析 → GPU Kernel 编程与性能建模 → 关键算子瓶颈定位与优化
        → 推理框架集成验证 → 下一代 DCU 适配优化 → 架构优化建议与方法论沉淀
```

- **主线一：软件优化能力建设** — 围绕现有 DCU 开展基础 Kernel 实验、GEMM / Attention / MoE 算子优化、Profiling 工具链建设及端到端性能验证。
- **主线二：下一代 DCU 技术储备与架构反馈** — 积累微架构性能数据与优化经验，提出下一代 DCU 的计算、访存、指令、编译器及性能工具需求，并与研发部门形成双向反馈。

---

## 二、项目总体目标

### 2.1 DCU 微架构研究目标
1. 系统掌握计算单元组织、SIMT 执行模型、MMAC 矩阵计算、寄存器、LDS、Cache、HBM 及 Wavefront 调度机制。
2. 建立面向主要 DCU 型号的微架构特性档案，结合 Microbenchmark 与 Roofline 模型形成理论分析与实测评估方法。
3. 对比现有及下一代 DCU 微架构变化，评估硬件演进对 Kernel 优化策略的影响。

### 2.2 Profiling 与 Kernel 开发能力建设目标
1. 梳理现有 DCU Profiling 工具及硬件性能计数器（如 `hipprof` PMC、DCUSIM），建立从模型热点到微架构瓶颈的标准分析流程。
2. 掌握 HIP C++、Triton / TileLang 编程方法，研究 Tile 划分、数据布局、寄存器与 LDS 资源使用及计算访存流水重叠。
3. 建立基础计算、访存及典型算子的 Benchmark 测试集与 ISA 反汇编分析能力。

### 2.3 大模型关键算子与推理框架优化目标
1. 围绕 **GEMM、FlashAttention / PagedAttention、MoE、稀疏计算及 KV Cache 访存**开展瓶颈分析与优化。
2. **2027 年完成至少 2 类关键算子的系统性优化研究**，形成可复现的 Kernel 实现、性能对比数据及技术报告。
3. 在 **SGLang / vLLM** 中选择典型 Dense、MoE 及长上下文模型，评估优化对 **TTFT、TPOT、吞吐量及 SLA 最大并发**的实际收益，并遵循非侵入式集成规范合入算子库或插件层。

### 2.4 下一代 DCU 技术储备与软硬件协同目标
1. 新硬件到位后第一时间完成基础计算、访存、指令与关键算子基线测试。
2. 针对实际优化中暴露的计算、访存、指令调度及编译器问题，向 DCU 架构与研发部门提供有实验数据支撑的技术反馈。

---

## 三、四大核心研究内容

### 3.1 海光 DCU 微架构与 SIMT 执行模型
- **DCU 微架构特性分析：** 计算单元、寄存器、LDS、Cache、HBM 容量/带宽/延迟实测与微架构档案。
- **矩阵计算与指令系统：** MMAC 矩阵乘加指令形状、数据类型、操作数布局、寄存器映射、延迟与吞吐。
- **调度与资源限制：** Occupancy、VGPR/SGPR/LDS 分配粒度、寄存器 Spill、Wavefront 调度与资源竞争。
- **理论性能建模：** 计算/访存密集型 Roofline 性能建模。

### 3.2 Profiling 与性能分析工具链
- **框架级：** PyTorch、vLLM / SGLang 在 Prefill / Decode 阶段的热点算子采集。
- **Kernel 级：** `hipprof`、硬件 PMC 计数器、Trace 与瓶颈分类。
- **微架构级：** 结合 DCUSIM（以研发侧提供版本为准）与 ISA 汇编交叉分析指令流水与停顿。
- **工具整合：** 自动化采集、解析、对比脚本沉淀。

### 3.3 DCU 编程与编译器
- **HIP C++：** Vector Add、Reduction、Transpose、GEMM 等基础算子库与访存/同步优化。
- **Triton / TileLang：** Tile 划分、IR/Backend lowering、Auto-tuning 与配置调优。
- **编译器与 ISA：** 从源码到设备端 ISA 的寄存器分配、MMAC 指令映射与编译器优化空间识别。

### 3.4 大模型关键算子优化
- **GEMM：** MMAC 利用率、Tile 划分、寄存器 Blocking、Shape-aware 与 Persistent Kernel。
- **MoE：** Grouped GEMM、Token 分组重排、小矩阵效率、专家负载均衡及算子融合。
- **稀疏计算：** 稀疏矩阵与稀疏注意力的非规则访存与负载优化。
- **KV Cache 与 Attention：** PagedAttention、非连续访存、Cache 局部性与长上下文带宽利用。
- **端到端验证：** SGLang / vLLM 集成与 TTFT / TPOT / 吞吐量评估。

---

## 四、配套参考资料

- 开源参考项目：[Hello-ROCm：解密 AI 加速器（decode-ai-accelerator）](https://datawhalechina.github.io/hello-rocm/zh/03-infra/decode-ai-accelerator)，用于对照学习 ROCm 软件栈追踪、Occupancy 分析与底层硬件执行链路。
- 内部技术积累：持续整理海光 DCU 内置指令（builtin / MMAC 原语）、硬件 PMC 指标说明与典型 Kernel 汇编分析案例，结合实验逐步沉淀为本站在册教程。

---

**项目最终目标：** 形成一支具备 DCU 微架构分析、底层 Kernel 优化、性能瓶颈定位及大模型推理系统优化能力的技术团队，为下一代海光 DCU 的适配和性能提升建立持续可复用的工程能力。
