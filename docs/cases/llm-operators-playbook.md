# 案例 2：大模型关键算子迁移与 Kernel 优化分析清单

## 八、把这些知识迁移到大模型关键算子

掌握以上内容后，可以用同样的方法研究 GEMM、Attention、MoE 等算子。

| 算子 | 重点硬件资源 | 典型性能问题 |
|---|---|---|
| GEMM | MMAC、VGPR、LDS、HBM | 矩阵指令利用率不足、Tile 不合理、资源压力过高 |
| FlashAttention | MMAC、LDS、VGPR、特殊函数 | QKV 数据搬运、Softmax、计算访存重叠 |
| PagedAttention | VMEM、Cache、HBM、Scheduler | KV 访问不规则、访存带宽或并行度不足 |
| MoE Grouped GEMM | MMAC、Scheduler、VGPR | 小矩阵利用率不足、专家负载不均衡 |
| RMSNorm | VALU、LDS、VMEM | Reduction、访存和同步开销 |
| LSTM Persistent Kernel | MMAC、LDS、VGPR、Scheduler | Recurrent 计算、状态复用、资源驻留限制 |

这些算子虽然计算公式不同，但底层优化思路有共通之处：

**先分析执行资源和数据路径，再判断性能瓶颈，最后调整 Kernel 组织方式。**

---

## 九、实际优化时应该怎样分析一个 Kernel？

建议将后续所有 Kernel 优化统一成以下分析流程。

### 9.1 明确计算任务

需要掌握：

- Kernel 输入输出 Shape。
- 数据类型。
- 计算量。
- 读写数据量。
- Batch、Sequence Length 等业务参数。

### 9.2 明确线程映射

需要掌握：

- Grid Size。
- Block Size。
- 每 Block Wavefront 数量。
- 每 Wavefront 负责哪些数据。
- 每个 Lane 负责哪些矩阵片段。

### 9.3 明确资源使用

需要掌握：

- VGPR、SGPR 用量。
- LDS 用量。
- Occupancy。
- 是否存在寄存器 Spill。
- Workgroup 驻留数量。

### 9.4 明确指令执行

需要掌握：

- 生成了哪些矩阵指令。
- MMAC、VALU、LDS、VMEM 等指令的使用情况。
- 是否存在长指令依赖链。
- 计算与数据搬运是否有效重叠。

### 9.5 明确性能瓶颈

根据实际 Profiling 与 Microbenchmark，判断主要问题属于：

- 计算吞吐不足。
- 访存带宽不足。
- 内存延迟。
- 并行度不足。
- 资源限制。
- 指令依赖或调度停顿。
- Kernel 启动与同步开销。

### 9.6 优化与验证

每次只调整少量变量，形成可对比实验。

例如：

- Block Size：64 / 128 / 256 / 512。
- 不同矩阵 Tile。
- 不同 LDS Layout。
- 不同寄存器 Blocking。
- Persistent 与非 Persistent。
- 不同 Wavefront 任务映射。

最终不仅记录性能提升，还需要解释性能变化的原因。

---

## 十、需要从海光 DCU 研发侧进一步确认的问题

结合现有 z100 架构资料以及后续下一代 DCU 研究需求，建议重点确认：

| 研究方向 | 需要确认的具体信息 |
|---|---|
| CU / SIMD | CU 组织、SIMD 数量、物理 Lane 宽度及 Wavefront Size |
| Scheduler | Wavefront 驻留上限、调度和指令发射机制 |
| MMAC | 指令形状、数据类型、Lane Layout、延迟和吞吐 |
| VGPR / SGPR | 容量、分配粒度、资源占用限制及 Spill 行为 |
| LDS | 容量、Bank 组织、访问粒度、吞吐及冲突规则 |
| Cache / HBM | 缓存层级、带宽、访问延迟和数据路径 |
| 编译器 | HIP/Triton 到 ISA 的映射、寄存器分配及优化选项 |
| Profiling | 硬件计数器、指令执行、Stall、访存与矩阵计算指标 |

应优先获取与实际运行芯片型号匹配的参数和工具说明。

对于下一代 DCU，需要特别关注哪些微架构变化会影响当前已经积累的 Kernel 优化策略。

---

## 十一、最终总结：怎样理解整个 DCU 微架构？

可以把 DCU 看成一个具备大量并行计算资源的系统。

**任务组织方面：**

Grid 负责组织全部计算任务，Block 负责组织线程协作，Wavefront 是重要的硬件执行单位，CU 则提供实际计算资源。

**指令执行方面：**

Scheduler 选择就绪 Wavefront，SALU 处理标量与控制操作，VALU 处理向量计算，MMAC 完成高吞吐矩阵乘加，访存单元负责数据加载和存储。

**数据存储方面：**

VGPR/SGPR 保存指令执行所需的寄存器值，LDS 支持 Block 内的数据共享和复用，Cache 减少部分全局访存成本，HBM 保存大容量模型数据。

**性能优化方面：**

Occupancy 影响可驻留并参与延迟隐藏的 Wavefront 数量；寄存器和 LDS 决定部分资源使用上限；指令流水和数据依赖影响计算效率；访存模式和数据复用影响数据供给效率。

对于实际算子优化，不能只关注某一种硬件资源。

真正高效的 Kernel，需要在计算、访存、寄存器、LDS、指令调度与并行度之间取得平衡。

因此可以把整个专题的核心理解为：

**GPU Kernel 优化，本质上是让计算任务以适合目标微架构的方式组织执行，并让硬件计算能力、数据供给能力和调度能力尽可能协调工作。**
