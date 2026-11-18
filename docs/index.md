# 海光 DCU 微架构与算子优化指南

欢迎来到 **DCU Microarchitecture & Kernel Optimization Guide**！  
本站服务于**“面向下一代海光 DCU 的微架构研究与大模型推理优化”**研究专题，采用**“先微架构原理 → 再 Profiling 工具链 → 再底层/高级算子实战 → 最后大模型推理优化”**的四段式体系编写。

---

## 研究入口

- [读原理](01-overview/software-stack.md)　理解执行、存储和调度。
- [跑实验](labs/index.md)　按问题查找实验与复现命令。
- [查 BW100 实测基线](baselines/bw100-gfx936.md)　下载环境、逐轮计时、PMC 与 ISA。

## 研究内容

### [专题实施方案](00-plan/research-plan.md)
- 项目定位、双主线技术路线、四大研究方向与 2027 年核心交付目标。

### 第一部分：DCU 微架构原理

前四节用同一个 1000 元素的向量加法串起执行模型、计算路径、存储和调度。第五节用独立例子讲 MMAC，最后用三张白板图复盘。每节有一道思考题。

1. [1.1 从 Kernel 启动到 CU](01-overview/software-stack.md) — 手算 Grid、Block、wave 和尾部有效线程。
2. [1.2 CU、Wavefront 与计算路径](02-compute/cu-and-simt.md) — 从 CU 内的 ALU 和寄存器讲到 SIMT、SIMD 与执行掩码。
3. [1.3 数据路径、缓存与同步](03-memory/memory-hierarchy.md) — 沿实际 ISA 读出加载、等待、计算与写回。
4. [1.4 调度、Occupancy 与延迟隐藏](04-scheduling/occupancy-and-roofline.md) — 用 E04 数据解释驻留和独立加载链。
5. [1.5 MMAC 矩阵乘加](02-compute/mmac.md) — 看一个 wave 如何协作计算矩阵块。
6. [1.6 三分钟白板复述](01-overview/whiteboard-review.md) — 自己画图，再展开参考和综合题答案。

### 第二部分：Profiling 与性能分析工具链
1. [2.1 工具链全景与软件栈探测](profiling/01-toolchain-overview.md) — `torch.profiler`、`hipprof`、ISA 导出与容器软件栈/硬件规格探测。
2. [2.2 硬件 PMC 采集与自研解析工具](profiling/02-hipprof-and-parser.md) — `hipprof --pmc` 指标速查与自研 `parse_hipprof_pmc.py` 解析工具。
3. [2.3 ISA 反汇编分析与仿真（DCUSIM 待补充）](profiling/03-isa-and-sim.md) — `.s` 汇编 5 大检查点与 DCUSIM 预留章节。

### 第三部分：底层微基准与算子实战

从 [实验索引](labs/index.md) 查看 E01–E10 的研究问题、复测状态和入口。首次运行先看 [环境准备](labs/lab-setup.md)。

### 第四部分：大模型推理优化

- [算子优化案例与检查清单](cases/llm-operators-playbook.md) — GEMM、融合 RMSNorm 与 LSTM Persistent 的分析入口。
- [研究路线图](roadmap.md) — 记录 BW100 复测进度以及下一步推理 shape 和跨代对照。
