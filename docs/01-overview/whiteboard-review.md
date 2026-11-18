<script setup>
import MicroarchDiagram from "../.vitepress/theme/MicroarchDiagram.vue"
</script>

# 1.5 · 三分钟白板复述

合上前四章，从 `C[i] = A[i] + B[i]` 开始。用三分钟画三张图，依次回答**任务怎么分、数据怎么走、等待怎么藏**。可以先讲一遍，再展开参考。

## 任务怎么分

用 `N=1000`、`blockDim=256`、BW100 Wave64 的例子。画出软件里的 Grid 和 Block，把一个 Block 展开成 wave，再画出承载它们的 CU。说出最后一个 wave 的有效线程数。

::: details 查看参考图与讲法

<MicroarchDiagram kind="mapping" />

一次 launch 提交 4 个 Block，每个 Block 256 个线程。BW100 上每个 Block 有 4 个 wave，一个 Block 的 wave 在分配到的同一个 CU 上执行。同一个 CU 可以驻留多个 Block。最后一个 wave 的全局索引是 960–1023，40 个线程对应有效元素。[回到 1.1](software-stack.md)

:::

## 数据怎么走

在一个有效 lane 上，从 A、B 的地址画到 C 的写回。标出操作数进入哪类寄存器、在哪里计算、何时需要等待。再说一种需要 LDS 的场景。

::: details 查看参考图与讲法

<MicroarchDiagram kind="memory" />

向量加法对全局地址发加载，结果进入 VGPR，由 VALU 做逐 lane 加法，最后写回 C。访问可能命中缓存。示例 ISA 用 `s_waitcnt vmcnt(0)` 确保相关操作满足依赖，才使用加载结果。分块协作时，线程可以先把数据写进 LDS，再以 Block 屏障协调消费者。[回到 1.2](../02-compute/cu-and-simt.md) · [回到 1.3](../03-memory/memory-hierarchy.md)

:::

## 等待怎么藏

画出 Wave A 发出加载和使用结果之间的间隔。说出同一个 wave 和其他 wave 分别可以做什么，再解释为什么仅靠驻留数量算不出速度。

::: details 查看参考图与讲法

<MicroarchDiagram kind="latency" />

有其他就绪 wave 时，调度器可以选择它发射，这是线程级并行提供的机会。同一 wave 有与加载结果无关的指令时，可以先发出独立工作，这是指令级并行提供的机会。驻留上限由资源决定，真实耗时还取决于指令依赖、数据路径和可用发射资源。[回到 1.4](../04-scheduling/occupancy-and-roofline.md)

:::

## 换一道题还能讲出来吗？

### 改启动参数

仍处理 1000 个元素。把 Block 改成 128 个线程，Grid 有多少个 Block？最后一个 Block 和最后一个 wave 分别有多少有效线程？

::: details 查看答案

Grid 有 **8 个 Block**，每个 Block 有 **2 个 wave**。最后一个 Block 从索引 896 开始，有 **104 个有效线程**；最后一个 wave 从 960 开始，仍然是 **40 个有效线程**。这些是由 launch 参数和 Wave64 算出的逻辑数量，执行顺序由设备决定。

:::

### 换同步场景

线程 A 把数据写进 LDS，同一个 Block 的线程 B 接下来要读。A 等到自己的写操作完成，B 就一定能读吗？Host 在哪一层等待？

::: details 查看答案

B 还需要与 A 在 Block 范围按约定协调，保证写入与读取的顺序并满足可见性。本例用 `__syncthreads()`。`s_waitcnt` 针对执行该指令的 wave 等待指定类别的未完成操作。Host 用相关 stream 或设备同步观察提交的工作何时完成。

:::

### 用证据解释反例

E04 在 4 wave/CU 上限时 8 条链最快，在 32 wave/CU 上限时 2 条链最快。这个结果能支持什么结论？还不能把原因归给哪个指标？

::: details 查看答案

数据支持**独立加载链的收益取决于驻留条件**。E04 固定每线程的逻辑读取量，也观察到链数变化伴随循环次数、发射顺序和 VGPR 变化。驻留上限是 HIP API 的估计；现有实验没有隔离哪一个因素导致 32 wave/CU 下 8 条链更慢。把预测、观测和机制假设分开说。[查看 E04 原始证据](../labs/lab04-occupancy-ilp.md)

:::

## 脱稿检查

- 能把 **Grid、Block、wave、lane、CU** 放到各自的位置，并算对尾部有效线程。
- 能说出 **SALU、VALU、MMAC** 分别处理哪类工作，指出向量加法走哪条路径。
- 能在图上区分 **寄存器、LDS、Cache、HBM**，再说明 `s_waitcnt`、Block 屏障和 Host 同步各自在等谁。
- 能用 **E04** 解释驻留与独立工作的关系，说出一个已经观测到的结果和一个仍待验证的机制。

编程模型与同步语义可查 DTK 26.04《HIP C++ 编程指南》《HIP Runtime API 开发手册》。BW100 对照值和具体指令来自 [实测档案](../baselines/bw100-gfx936.md)；z100 结构仅作为 <a href="./software-stack.html#_1-2-dcu-的物理硬件组织" target="_self">架构参考</a>。

[继续 · Profiling 工具与 ISA](../profiling/01-toolchain-overview.md) · [回到实验索引](../labs/index.md)
