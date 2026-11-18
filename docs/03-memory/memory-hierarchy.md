---
outline: 2
---

<script setup>
import MicroarchDiagram from "../.vitepress/theme/MicroarchDiagram.vue"
</script>

# 1.3 · 数据路径、缓存与同步

每个有效线程先读取自己的 A 和 B，等数据就绪后做加法，再写回 C。需要线程合作时，还会用到 LDS 和 Block 同步。

::: info 先认识这几个词

**寄存器**暂存指令正在使用的数据。**LDS**（Local Data Share）是同一 Block 的线程可以显式使用的片上共享存储。**Cache** 是硬件管理的缓存。**HBM**（High Bandwidth Memory，高带宽内存）是这里的设备显存。

:::

## 1. 数据分别放在哪里

| 位置 | 谁在用 | 常见用途 |
|---|---|---|
| SGPR / VGPR | 指令通过寄存器读写操作数 | SGPR 放 wave 共用值，VGPR 可以放各 lane 的值。[寄存器分工](../02-compute/cu-and-simt.md#cu-内各路径做什么) |
| LDS | 同一 Block 内的线程按程序安排使用 | 用 `__shared__` 暂存 tile，减少重复加载，或跨 wave 交换数据。 |
| L1 / L2 Cache | 硬件按访存路径、局部性与缓存策略工作 | 局部或跨 CU 复用。BW100 报告 8 MiB L2，实际延迟转折还受设备和地址模式影响。 |
| HBM | 存放全局地址对应的大容量数据 | 模型权重、输入输出、激活和 KV Cache 等。 |

::: tip LDS 与 Cache 的区别

LDS 的数据布局和复用由程序安排，同一 CU 上多个 Block 各有自己的 LDS 分配区。Cache 由硬件按访问情况保存和替换数据。向量加法的线程没有共享中间结果，直接用寄存器计算。

:::

带宽描述一段时间能搬多少数据，延迟描述一个请求从发起到能使用结果要多久。E06 用单线程依赖链突出延迟，E01 用批量拷贝观察吞吐，两种测试使用不同口径。[E06 延迟阶梯与 TCC](../labs/lab08-memory-latency-ladder.md) · [E01 有效拷贝带宽](../labs/lab01-memory-coalescing.md)

## 2. 地址分布如何影响访存

### 全局内存的合并与向量化

**合并访存**看相邻 lane 的地址是否能由少量事务服务。例如，多个 lane 各读一个相邻的 `float`，通常比各读一个相隔很远的 `float` 更容易合并。**向量化访存**看每个 lane 的一条指令搬多少数据，合法对齐且编译成宽指令的 `float4` 可以一次搬 16 字节。两者可独立改变，E01 分别测了 stride 和 `float4`。[E01 两组对照](../labs/lab01-memory-coalescing.md)

### LDS 的 bank 冲突

LDS 可以划分为多个能并行服务的 bank。用 E02 的简化模型理解，一组 32 个 lane 分别访问对齐的 4 字节元素：

$$\text{Bank ID} = \left\lfloor \frac{\text{Byte Address}}{4} \right\rfloor \bmod 32$$

如果大家访问同一个 bank 的**不同地址**，服务可能发生冲突。如果读取的是同一个地址，硬件可能广播。访问宽度和指令会影响服务方式。对一列 32 对齐跨度的元素，Padding 可以把行跨度改成 33，XOR Swizzle 可以重排对应于 lane 的列号，使不同地址分散到不同 bank。[E02 LDS 实测](../labs/lab02-lds-bank-conflict.md)

## 3. 从全局地址到加法 {#跟一条加载指令}

**VMEM** 是向量访存，常用于 lane 各读不同地址。**SMEM** 是标量访存，可用于 wave 共用的地址或参数。访问 LDS 时可看到 `ds_*` 指令，也可能有指向共享地址空间的 `flat_*` 指令，如 E02 中的实现。具体指令由编译器根据地址和目标生成。

<MicroarchDiagram kind="memory" />

本例在 BW100、DTK 26.04、DCC 25.10.0、`-O3` 下生成如下设备端 ISA。片段省略了地址计算等指令。

```asm
global_load_dword  v4, v[2:3], off
; 省略地址计算
global_load_dword  v2, v[2:3], off
; 省略其他指令
s_waitcnt vmcnt(0)
v_add_f32_e32      v2, v4, v2
global_store_dword v[0:1], v2, off
```

两条加载的结果分别进入 `v4` 和 `v2`，随后加法才消费这两个结果。发出加载后，如果发射资源允许，wave 还可先做与结果无关的工作。例如矩阵 Kernel 可以提前发出下一块 tile 的加载，计算已经就绪的当前 tile，再等待和消费下一块。[主例和编译命令](../01-overview/software-stack.md#动手运行)

## 4. 谁在等什么 {#区分三种等待}

| 谁在等 | 在等什么 | 例子 |
|---|---|---|
| 当前 wave | 指定类别的未完成操作降到要求的阈值 | 本例用 `s_waitcnt vmcnt(0)` 保证向量加载完成。`lgkmcnt` 覆盖的操作可包括 LDS 和标量加载，具体看目标 ISA。 |
| 同一个 Block 的线程 | 其他线程到达同一个同步点，按约定交换数据 | 线程写入 LDS，再用 `__syncthreads()` 协调读取。 |
| Host | 已提交到 stream 或设备的工作完成 | `hipStreamSynchronize(stream)` 等待该 stream 中的工作。 |

::: tip 等待自己的数据和让同组线程会合是两件事

`s_waitcnt` 约束执行该指令的 wave。Block 屏障协调同一个 Block 的线程。跨 Block 交换数据需要相应的原子操作、内存顺序或更高层同步机制。

:::

如果数据只需要在一个 wave 的 lane 之间交换，可用 `__shfl_down`、`__shfl_xor` 等 Shuffle 原语。目标代码可能使用 DPP（Data Parallel Primitives）或 `ds_bpermute` 等。跨 wave 规约常先做 wave 内规约，再由每个 wave 写一个部分结果到 LDS，在 Block 同步后汇总。[E05 Wave Shuffle 实验](../labs/lab05-wave-shuffle-reduction.md) · [LLVM AMDGPU 同步参考](https://llvm.org/docs/AMDGPUExecutionSynchronization.html)

## 5. 思考题

同一个 Block 的 Wave A 写入 LDS，Wave B 随后要读这批数据。只让 Wave A 用 `s_waitcnt` 等自己的写操作完成，Wave B 就能直接读吗？应该怎么协调？

::: details 查看答案

还需要协调读写双方。Wave A 等待的是自己发出的操作，Wave B 可能早就执行到读取位置。此例让同一个 Block 的线程在写入后都到达 `__syncthreads()`，再执行读取。

:::

[继续 1.4 · CU 如何减少等待造成的空闲](../04-scheduling/occupancy-and-roofline.md)
