---
outline: 2
---

<script setup>
import MicroarchDiagram from "../.vitepress/theme/MicroarchDiagram.vue"
</script>

# 1.6 · 三分钟白板复述

用同一个 1000 元素的向量加法，把**任务怎么分、数据怎么走、等待怎么藏**讲一遍。每张图先试着自己画，再展开参考。

::: info 先把三组词连起来

**Grid、Block、wave 和 CU**说明任务怎么分派。**全局内存、寄存器和 LDS**说明数据去哪。**TLP 和 ILP**说明等待期间还有哪些工作可以发射。

:::

## 1. 任务怎么分 {#任务怎么分}

画出一次 launch 的 Grid，展开最后一个 Block 里的 wave，标出最后一个 wave 有多少个线程对应有效数据。

::: details 查看参考图

<MicroarchDiagram kind="mapping" />

4 个 Block，每 Block 256 个线程。在 BW100 上每 Block 有 4 个 wave，一个 Block 的 wave 在同一个 CU 上执行。最后一个 wave 的全局索引是 960–1023，前 **40 个 lane** 对应有效数据。[回到 1.1](software-stack.md)

:::

## 2. 数据怎么走 {#数据怎么走}

选一个有效 lane，从 A、B 的地址画到 C 的写回。指定寄存器和运算路径，再说一种需要 LDS 的场景。

::: details 查看参考图

<MicroarchDiagram kind="memory" />

向量加载把输入放进 VGPR，结果就绪后，VALU 做加法，最后写回 C。全局地址访问可能由缓存服务。分块协作时，同一个 Block 的线程可以用 LDS 交换数据。[回到 1.2](../02-compute/cu-and-simt.md) · [回到 1.3](../03-memory/memory-hierarchy.md)

:::

## 3. 等待怎么藏 {#等待怎么藏}

画出 Wave A 发出加载到使用结果之间的间隔。说出其他 wave 和 Wave A 本身分别可以做什么。

::: details 查看参考图

<MicroarchDiagram kind="latency" />

如果另一个 wave 就绪，调度器可以让它发射，这是 TLP 带来的机会。同一 wave 有不依赖加载结果的指令，也可以先发射，这是 ILP 带来的机会。驻留上限来自资源约束，实际速度要看依赖、发射和内存路径。[回到 1.4](../04-scheduling/occupancy-and-roofline.md)

:::

## 4. 思考题

现在改做一个 `16×16×16` 的矩阵乘加 tile。A、B 的数据先由一个 Block 写入 LDS，再由其中一个 wave 读取。相比向量加法，数据怎样多走了一步？什么时候用 Block 屏障，什么时候等待加载？最后哪条执行路径做乘加？

::: details 查看答案

从全局内存取来的片段先写到 **LDS**，同一 Block 的线程通过适当的 Block 屏障协调写入后的读取。读到寄存器的片段要满足指令依赖，才能被后续计算消费。各 lane 按目标指令布局提供 A、B 和累加片段，wave 走 **MMAC** 路径完成矩阵块的乘加，后续再按 Kernel 的输出布局写回。例子可对照 [1.3 同步](../03-memory/memory-hierarchy.md#区分三种等待) 和 [1.5 MMAC](../02-compute/mmac.md)。

:::

编程模型和同步语义可参照 DTK 26.04《HIP C++ 编程指南》《HIP Runtime API 开发手册》。设备参数与指令测量见 [BW100 实测档案](../baselines/bw100-gfx936.md)，z100 的组织见 [架构参考](software-stack.md#_1-2-dcu-的物理硬件组织)。

[继续 · Profiling 工具与 ISA](../profiling/01-toolchain-overview.md) · [回到实验索引](../labs/index.md)
