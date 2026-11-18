---
outline: 2
---

<script setup>
import MicroarchDiagram from "../.vitepress/theme/MicroarchDiagram.vue"
</script>

# 1.3 · 数据路径、缓存与同步

在 1.1 的向量加法中，CPU 把数组 A、B 复制到 DCU，然后启动 Kernel。拿线程 7 举例，它从设备数组读取 `A[7]` 和 `B[7]`，把两个值放进寄存器，算完再写到 `C[7]`。这一节沿着这三个数走一遍，再看相邻线程需要互相传数据时要做什么。

## 1. 数据分别放在哪里

| 位置 | 用向量加法理解 | 谁决定放什么 |
|---|---|---|
| 主机内存 | CPU 初始化的 `a`、`b` 和最终用来校验的 `c` | Host 程序分配，`hipMemcpy` 在主机和设备之间复制 |
| 设备全局内存（本例由 HBM 显存提供） | `hipMalloc` 分配的设备端 A、B、C 数组，可被各 Block 访问 | 程序分配数组，通过全局地址访问 |
| L1 / L2 缓存 | 读 `A[7]` 时，硬件可能从已经缓存的数据中取值。L1 靠近 CU，L2 可服务多个 CU | 硬件根据访问路径和缓存策略管理 |
| 寄存器 | 线程 7 加载到的两个输入和加法结果可暂存在自己对应的 VGPR lane 位置，数组基地址等统一值可用 SGPR | 编译器按数据和指令分配 |
| LDS（片上共享存储） | 向量加法各算各的。若线程 7 要把自己的值交给同一 Block 的线程 8，可以用 `__shared__` 数组暂存 | Kernel 决定谁写、谁读，以及何时同步 |

这里的“全局内存”描述程序用什么地址访问数组。一次全局加载能否由缓存直接响应，要看当时的访问情况。LDS 则有程序明确指定的读写位置，同一个 CU 上的不同 Block 各用自己的 LDS 分配区。

<MicroarchDiagram kind="memory" />

## 2. BW100 的 L1 和 L2 有多大

在 **BW100（gfx936）、DTK 26.04** 上，`rocminfo` 的 BW100 设备条目打印 `L1: 32 KB` 和 `L2: 8192 KB`。按二进制单位写就是 **32 KiB** 和 **8 MiB**。其中 L2 也与 [HIP 设备属性记录](../baselines/bw100-gfx936.md)一致。

`rocminfo` 的摘要每层只显示一个数。再看 Linux KFD 驱动（向运行时提供设备拓扑）的缓存明细，可以看到一级缓存分了几类。

| BW100 驱动报告 | 每条记录的容量 | 条目数 |
|---|---:|---:|
| L1 数据缓存 | 32 KiB | 64 |
| L1 数据缓存 | 16 KiB | 64 |
| L1 指令缓存 | 32 KiB | 64 |
| L2 数据缓存 | 8 MiB | 1 |

指令缓存服务取指，数据缓存服务数据读取。L1 表中的容量对应单条拓扑记录。两类数据缓存分别服务哪些加载，可结合下方实验理解。驱动在这里将两类都标为数据缓存，E06 的结果与向量路径可利用 32 KiB、标量路径可利用 16 KiB 的解释相符。工作集的转折也会随地址间距变化，缓存的共享范围和替换行为需要更细的对照。[KFD 拓扑实现](https://kernel.googlesource.com/pub/scm/linux/kernel/git/torvalds/linux/+/69050f8d6d075dc01af7a5f2f550a8067510366f/drivers/gpu/drm/amd/amdkfd/kfd_topology.c#1618) · [E06 完整结果](../labs/lab08-memory-latency-ladder.md)

BW100 的 HIP 设备属性另有一个 **每 Block 最多 64 KiB LDS** 的报告值。LDS 是第 1 节里给线程交换数据的共享空间，这个上限和上表的缓存容量是两项不同的设备属性。[HIP 设备属性探测源码](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/01_device_info_and_isa.hip)

::: details 查看自己机器的缓存报告

先运行 `rocminfo`，找到 `Marketing Name` 为 BW100 且 `Name` 为 gfx936 的设备条目，查看 `Cache Info`。要看驱动明细，把下面的 8 换成同一条目里显示的 `Node` 编号。这个步骤适用于能访问该拓扑的 Linux DTK 环境。

```bash
rocminfo
node=8
grep -h -E '^(level|size|type) ' /sys/class/kfd/kfd/topology/nodes/"$node"/caches/*/properties |
  paste - - - | sort | uniq -c
```

每行开头是同类记录的数量，`level` 是缓存级别，`size` 在此报告以 KiB 计。`type 9` 对应设备数据缓存，`type 10` 对应设备指令缓存。

:::

## 3. 从全局地址到加法 {#跟一条加载指令}

先看一行 HIP 代码 `c[i] = a[i] + b[i]`。按 [1.1 的命令](../01-overview/software-stack.md#动手运行)编译后，在 BW100 的设备汇编里可以沿着寄存器找这五步。以下片段省略了地址计算等指令。

```asm
global_load_dword  v4, v[2:3], off
global_load_dword  v2, v[2:3], off
s_waitcnt vmcnt(0)
v_add_f32_e32      v2, v4, v2
global_store_dword v[0:1], v2, off
```

1. 两条 `global_load_dword` 分别读取 A、B，把结果带回 `v4`、`v2`。每个有效 lane 都有自己的数据。
2. `s_waitcnt vmcnt(0)` 等这个 wave 的相关向量访存完成。发出加载指令到数据可用之间有一段时间。
3. `v_add_f32_e32` 读取这两个值，把加法结果放回 `v2`，`global_store_dword` 再把它写到 C。

像这样按 lane 访问数组的加载称为向量访存（VMEM）。如果编译器能证明一个地址和结果由整个 wave 共用，也可使用标量访存（SMEM），例如 E06 的 `s_load_dword`。想练习跟踪指令，可以在 1.1 生成的设备端 `.s` 中圈出两次加载的目标寄存器，再看加法是否读取了它们。

## 4. 谁在等什么 {#区分三种等待}

假设一个 128 线程的 Block 要交换数据。在 BW100 上，线程 0 和线程 64 属于不同 wave。让每个线程先写入自己的 `tile[threadIdx.x]`，全 Block 一起执行 `__syncthreads()`，之后线程 64 才能读线程 0 写的 `tile[0]`。同一个 Block 的线程需要到达同一个屏障，尾部线程也要参与同步，可用条件控制屏障前后的读写。

| 谁在等 | 等什么 | 在本节的例子 |
|---|---|---|
| 一个 wave | 自己已经发出的相关操作完成 | `s_waitcnt vmcnt(0)` 之后，向量加法才使用加载结果 |
| 同一个 Block 的线程 | 其他线程到达共同的同步点，以便安全读取这轮写入的数据 | 写 LDS 后执行 `__syncthreads()`，再读另一 wave 写的数据 |
| CPU 上的 Host | 已提交的设备工作完成 | 1.1 中，`hipDeviceSynchronize()` 返回后，Host 复制 C 回来校验 |

同一 wave 内交换值还可以用 Shuffle，例如 `__shfl_down`。需要把 Block 内所有 wave 的结果汇总时，可以先各 wave 自己规约，再通过 LDS 和 Block 屏障合并。[E05 Wave Shuffle 实验](../labs/lab05-wave-shuffle-reduction.md)

## 5. 动手实战 · 看数据变多后加载何时变慢

这个小实验让一个线程沿一组打乱顺序的地址反复读取，每次读到的值是下一次要访问的位置。后一跳必须等待前一跳。我们只改这组数据占用的地址范围，观察完成一跳平均要等多久。这里的**工作集**就是这些地址所覆盖的范围。

在 BW100 的 DTK 环境中，从仓库根目录运行。

```bash
cd experiments
hipcc -O3 -std=c++17 --offload-arch=gfx936 07_memory_latency_ladder.hip -o 07_memory_latency_ladder
./07_memory_latency_ladder 16 42 64 scalar
./07_memory_latency_ladder 24 42 64 scalar
./07_memory_latency_ladder 32 42 64 lane
./07_memory_latency_ladder 48 42 64 lane
```

第一个参数是工作集大小（KiB），`42` 固定地址随机顺序的种子，`64` 是节点间距（字节）。`scalar` 和 `lane` 分别让编译器生成标量和向量加载。每个配置输出五轮，比较最后一列 `clock64_ticks_per_hop` 的中位数。

| 路径 | 小工作集 | 较大工作集 | BW100 复测中位数 |
|---|---:|---:|---:|
| 标量加载 | 16 KiB | 24 KiB | 约 58 → 220 tick/跳 |
| 向量加载 | 32 KiB | 48 KiB | 约 153 → 285 tick/跳 |

没有卡时也可以读上表。先比较**同一行**的两个数，再把变慢前的工作集大小和驱动报告对照。稳定的延迟上升提示原来能更快取得的数据现在更难复用。每跳也包含循环和地址计算开销，读数会随设备配置变化。这个实验问的是等一次读取要多久。若要看许多线程一起工作时每秒能搬多少字节，可以做 E01 的带宽实验。[E06 完整扫点和源码](../labs/lab08-memory-latency-ladder.md) · [E01 带宽实验](../labs/lab01-memory-coalescing.md)

::: details 进阶阅读 · 合并访存、向量化和 LDS bank

- **合并访存**看同一个 wave 的相邻 lane 是否访问相邻地址。Lane 0–3 分别读 `a[0..3]`，通常比各自读间隔很远的位置更容易用少量内存事务完成。
- **向量化访存**看单个 lane 的一条指令能搬多少字节。满足对齐等条件且编译为宽指令时，`float4` 可以一次搬 16 字节。[E01 两组对照](../labs/lab01-memory-coalescing.md)
- **LDS bank 冲突**看访问是否集中在同一个 bank 的不同地址。用 E02 的简化模型，每个对齐的 4 字节元素按 `元素索引 % 32` 分配 bank。索引 0、32、64 落在同一个 bank，把行跨度改为 33 后，索引 0、33、66 则落在不同 bank。访问同一个地址时，硬件可能广播。具体服务方式还取决于访问指令。[E02 LDS 实测](../labs/lab02-lds-bank-conflict.md)

:::

## 6. 思考题

Wave A 写 LDS，Wave B 读这批数据。如果只让 Wave A 等待自己的写操作完成，Wave B 什么时候才可以安全读？

::: details 查看答案

Wave B 可能比 Wave A 更早到达读取位置。让同一 Block 的所有线程在写入后执行 `__syncthreads()`，再开始读取，就能协调读写双方。

:::

[继续 1.4 · CU 如何减少等待造成的空闲](../04-scheduling/occupancy-and-roofline.md)
