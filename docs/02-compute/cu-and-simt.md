---
outline: 2
---

<script setup>
import MicroarchDiagram from "../.vitepress/theme/MicroarchDiagram.vue"
</script>

# 1.2 · CU、Wavefront 与计算路径

Block 分到 CU 后，CU 会安排其中的 wave 执行。先拆开看这个计算单元里有哪些资源，再看向量加法的尾部线程怎样执行。

::: info 先认识这几个词

**CU**（Compute Unit）是承载 Block 的硬件计算单元。**ALU**（Arithmetic Logic Unit，算术逻辑单元）是计算路径中的部件，做加法、比较、位运算等。**寄存器**保存指令要用的值。**LDS**（Local Data Share）是 CU 上的片上共享存储，同一 Block 的线程可用它交换数据。

:::

## 1. CU 怎么组织执行

一个 CU 需要做三件事：让 wave 驻留，选择哪条指令可以发射，再交给合适的执行路径。CU 里有调度器、寄存器和不同用途的执行路径，ALU 负责其中的算术与逻辑运算。

- **设备分派 Block。** 设备决定给哪个 CU 分配一个 Block。
- **CU 调度 wave。** 已驻留的 Wave A 等数据时，若 Wave B 就绪且相应执行资源可用，调度器可让 Wave B 发射指令。Wave A 的执行状态仍然保留。

这是功能分工。一个线程的逻辑 lane 编号用于表示它在 wave 里的位置，物理 ALU 会被硬件反复用于不同指令和线程组。具体资源组织随架构而变。[HIP 硬件实现](https://rocm.docs.amd.com/projects/HIP/en/docs-6.3.3/understand/hardware_implementation.html)

## 2. 先看寄存器，再看执行路径 {#cu-内各路径做什么}

| 寄存器 | 英文全称 | 存什么 |
|---|---|---|
| SGPR | Scalar General-Purpose Register | 标量通用寄存器。一个 wave 共用的值，例如编译器可证明统一的基地址或参数。 |
| VGPR | Vector General-Purpose Register | 向量通用寄存器。一个 wave 内可以每个 lane 各有一个值，例如各线程的数组索引、读到的数据或运算结果。 |

::: tip 想一想这个值有没有随 lane 变化

向量加法中，数组基地址对一个 wave 可以相同，`a[i]` 和 `b[i]` 随线程变化。这个区别有助于理解 SGPR 和 VGPR 的分工，具体放在哪个寄存器由编译器和目标指令决定。

:::

| 功能 | 主要职责 |
|---|---|
| **SALU**（Scalar Arithmetic Logic Unit，标量算术逻辑单元） | 处理 wave 级的标量运算和部分控制，例如统一的地址或循环状态。 |
| **VALU**（Vector Arithmetic Logic Unit，向量算术逻辑单元） | 对有效 lane 各自的数据运算，例如向量加法中的 FP32 加法。 |
| **VMEM**（Vector Memory Operations，向量访存） | 发出按 lane 的内存操作，例如读取 A、B 或写回 C。 |
| 特殊函数路径（常称 **SFU**，Special Function Unit） | 部分目标提供指数、倒数或平方根倒数等专门指令。具体映射看目标 ISA。 |

矩阵块的乘加还有专门的指令路径，称为 **MMAC**（Matrix Multiply-Accumulate），见 [1.5 MMAC](mmac.md)。

<MicroarchDiagram kind="compute" />

## 3. SIMT 与 SIMD 对应哪一层

- **SIMT**（Single Instruction, Multiple Threads，单指令多线程）描述编程模型。HIP 里每个线程有自己的编号和数据，硬件把同一个 wave 中参与的线程组织在一条指令流中。
- **SIMD**（Single Instruction, Multiple Data，单指令多数据）描述执行方式。一条指令对多份数据做同一种操作。向量加法里，参与的 lane 执行同一种加法，各自用自己的操作数。

在一维 Block 中，`wave_id = threadIdx.x / warpSize`，`lane_id = threadIdx.x % warpSize`。在 BW100 的 Wave64 上，线程 130 是 **Wave 2 的 Lane 2**。最后一个 wave 从索引 960 开始。索引 999 对应 Lane 39，会写入结果；紧挨着的索引 1000 对应 Lane 40，已经越界。同一条写指令怎样只让前者参与？下一段的 EXEC 记录了这个选择。

::: details 进阶阅读 · 一个 wave 与物理执行宽度

早期部分 GCN 的 SIMD 物理宽度是 16，一条覆盖 64 个逻辑 lane 的普通向量指令可能分成数批处理。观察 BW100 时要分别核对 wave 的逻辑宽度、具体指令的物理执行方式、等待结果的延迟、以及单位时间能完成的吞吐。[HIP 编程模型](https://rocm.docs.amd.com/projects/HIP/en/docs-6.3.3/understand/programming_model.html)

:::

## 4. 尾部 wave 如何处理不同条件 {#一个-wave-怎么执行}

在 1.1 的最后一个 wave 中，40 个 lane 的索引满足 `i < count`。同一 wave 中条件不同时，参与向量指令的 lane 可由执行掩码控制。

<MicroarchDiagram kind="lanes" />

| 名称 | 作用 |
|---|---|
| **EXEC**（Execution Mask） | 保存当前参与向量执行的 lane 掩码。 |
| **VCC**（Vector Condition Code） | 可保存向量比较等操作的逐 lane 结果。 |
| **SCC**（Scalar Condition Code） | 保存某些标量指令产生的条件状态，可用于标量控制流。 |

向量加法在 BW100 上实际生成了向量比较和 `s_and_saveexec_b64`。它保存旧 EXEC，再用比较结果更新参与的 lane。含 `else` 的控制流可能分别用两组掩码执行两侧，再在汇合处恢复原掩码。全 wave 一致的条件有机会用标量分支，短分支也可能被编译成诸如 `v_cndmask_b32` 的谓词选择指令，性能要结合实际汇编判断。[向量加法编译命令](../01-overview/software-stack.md#动手运行) · [E03 分支实验](../labs/lab03-branch-divergence.md) · [LLVM AMDGPU 寄存器参考](https://llvm.org/docs/AMDGPUUsage.html)

## 5. 动手实战 · 找出尾部线程的执行掩码

先按 [1.1 向量加法](../01-overview/software-stack.md#动手运行)编译并运行。在同一个 `experiments/` 目录里，用下面的命令筛选 BW100 的设备汇编。没有设备时，直接读下面的两行实测片段。

```bash
grep -nE 'v_cmp|saveexec|global_load|global_store' *tutorial*-hip-amdgcn-*-gfx936.s
```

```asm
v_cmp_gt_i32_e32  vcc, s1, v0
s_and_saveexec_b64 s[0:1], vcc
```

这一对指令在本例中对应 `count > i`。先把逐 lane 的比较结果写入 VCC，再把它和原 EXEC 取交集。自己算一下最后一个 wave 哪些位会保留。这里的合法索引是 960–999，因此 EXEC 的低 40 位保留，高 24 位被清掉。后面的数组读写只由有效 lane 参与，1.1 的 `24 untouched` 是从主机端验证的结果。

## 6. 思考题

仍看向量加法的最后一个 wave。假设编译器先比较 `i < 1000`，再按比较结果更新 EXEC，哪几个 lane 会执行数组读写？数组里每个线程拿到的 `a[i]` 用 SGPR 还是 VGPR 表示更合适？

::: details 查看答案

这个 wave 从索引 960 开始，Lane 0–39 对应索引 960–999，会执行数组读写。Lane 40–63 对应越界索引。`a[i]` 随 lane 变化，适合用 **VGPR** 表示；可以证明整个 wave 统一的值才有机会走 SGPR。

:::

[继续 1.3 · 数据从哪里来](../03-memory/memory-hierarchy.md)
