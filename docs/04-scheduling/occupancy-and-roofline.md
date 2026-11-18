---
outline: 2
---

<script setup>
import MicroarchDiagram from "../.vitepress/theme/MicroarchDiagram.vue"
</script>

# 1.4 · 调度、Occupancy 与延迟隐藏

一个 wave 在等数据时，CU 可以选择其他就绪的 wave。要理解什么时候有工作可做，需要同时看能驻留多少 wave、当前指令的依赖，以及执行资源是否可用。

::: info 先认识这几个词

**驻留**表示一组工作已经分到 CU，并占用执行所需资源。**就绪**表示这条指令的依赖满足，并具备发射条件。**Occupancy**（占用率）用硬件支持的最大驻留 wave 数作为参照，衡量能驻留或实际驻留多少 wave。

:::

## 1. CU 最多能驻留多少 Block

Block 需要 wave 驻留槽位、一定数量的寄存器，以及程序申请的 LDS。单个资源通常给出一个上限，整体还要同时满足其他资源约束。例如，只看容量为 64 KiB 的 LDS：

| 每 Block 申请的 LDS | LDS 容量能容纳的 Block 数 |
|---|---:|
| 16 KiB | 4 |
| 32 KiB | 2 |
| 48 KiB | 1 |

VGPR、SGPR 或 LDS 的用量跨过一个硬件分配档位，驻留上限可能跟着下降。分配粒度看目标设备。寄存器不足时，编译器可能把一些线程私有值溢出到 **scratch**，产生额外内存访问。像 `__launch_bounds__(256)` 这样的声明为编译器提供 Block 线程数上限，实际用多少线程仍由调用时的启动参数决定。查看编译后的 **code object（设备代码及资源元数据）** 和实测计时，有助于检查这些影响。

::: tip 驻留上限、实际驻留、发射利用率各回答一个问题

- **驻留上限**看资源允许多少 wave。假如一个 CU 的硬件最多支持 40 个 wave，当前 Kernel 的资源只允许 20 个，理论 Occupancy 就是 50%。E04 的横轴来自 HIP occupancy API。
- **实际驻留**看执行过程中同时保有多少 wave。小 Grid 也可能无法填满所有 CU。
- **发射利用率**看有多少可用发射机会被实际使用。已驻留的 wave 可能正等待数据。

:::

## 2. 两种延迟隐藏方式

**TLP**（Thread-Level Parallelism，线程级并行）利用其他就绪 wave 的工作。**ILP**（Instruction-Level Parallelism，指令级并行）利用同一个 wave 中与正在等待的结果无关的指令。

<MicroarchDiagram kind="latency" />

用向量加法举例，A 和 B 的加载可以发出，随后的加法要等它们回来。调度器可以在资源允许时从其他 wave 选指令发射。更复杂的 Kernel 还可以在当前 wave 中穿插独立地址计算、下一次加载或已就绪的运算。Block 做完并释放资源后，设备可以继续分派其他待执行的 Block。

## 3. 用 E04 看不同驻留下的独立加载链

![BW100 独立加载链对照。4 wave/CU 的驻留上限下 8 条链最快，32 wave/CU 的驻留上限下 2 条链最快。](/data/bw100-20261118/figures/e04-occupancy-ilp.png)

E04 用动态 LDS 改变驻留上限，每个线程始终读同一组 64 个节点。上限是 **4 wave/CU** 时，8 条链把耗时从约 0.4560 ms 降到 0.2278 ms；上限是 **32 wave/CU** 时，2 条链最快。链数增加也改变循环次数、地址发射顺序和 VGPR 用量。数据说明收益随配置变化，具体因素的贡献还需要独立对照。[E04 方法与 PMC](../labs/lab04-occupancy-ilp.md)

寄存器用量的变化同样要看整条路径。E09 只把 `BK` 从 32 降到 16，报告的 VGPR 从 144 降到 112，LDS 仍为 8704 B，scratch 仍为 0，耗时却从 1.4120 ms 增至 1.4973 ms。[E09 GEMM 消融](../labs/lab07-triton-gemm-vs-rocblas.md)

## 4. 用 Roofline 估算性能上限 {#_6-5-roofline-模型与-bw100-对照}

把运算量记为 $F$，实际 HBM 读写字节数记为 $B_\mathrm{HBM}$。**计算强度** $I=F/B_\mathrm{HBM}$ 表示每搬一字节做多少运算。Roofline 用计算吞吐和带宽约束估算上限：

$$
P \le \min(P_\mathrm{compute},\ I \cdot BW_\mathrm{HBM})
$$

两条上限相交的位置为 $P_\mathrm{compute}/BW_\mathrm{HBM}$。靠近哪条线可以帮助提出瓶颈假设，再用资源、计数器和消融验证。

::: warning 计算强度的字节数要用同一口径

BW100 上，E07 的 MMAC 指令微基准为 333.49 TFLOPS，E01 的向量拷贝有效带宽为 1.367 TB/s，两者相除约为 **244 FLOPs/B**。E01 采用逻辑读写字节，因此这个数字是经验参考。严格的 HBM Roofline 需要实测的 HBM 流量。

:::

E09 的 `4096³` GEMM 若按 A、B 各读一次与 C 写一次计算，约为 1365 FLOPs/逻辑字节。若要在严格的 HBM Roofline 上定位，还要测实际流量。[BW100 实测档案](../baselines/bw100-gfx936.md)

## 5. 思考题

某个 Wave64 Kernel 每 Block 有 256 个线程，CU 提供 64 KiB LDS。只看 LDS 容量，把每 Block 的 LDS 用量从 **16 KiB** 提高到 **32 KiB**，Block 和 wave 的驻留上限各怎么变？能据此预测性能一定降低吗？

::: details 查看答案

每 Block 有 4 个 wave。只看 LDS，Block 上限从 **4 降到 2**，wave 上限从 **16 降到 8**。其他资源可能让实际可驻留数更少。耗时还受数据复用、指令数量和独立工作量等影响，比较计时才能确认性能变化。

:::

[继续 1.5 · 一个 wave 怎样做矩阵乘加](../02-compute/mmac.md)
