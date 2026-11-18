---
outline: 2
---

<script setup>
import MicroarchDiagram from "../.vitepress/theme/MicroarchDiagram.vue"
</script>

# 1.3 · 数据路径、缓存与同步

**读完要能讲出来**　操作数从哪里来？什么时候能用？什么时候需要等其他线程？

## 基础概念 · 寄存器、LDS、Cache 与 HBM

### 4.1 VGPR：向量寄存器

VGPR（Vector General Purpose Register）用于保存向量运算数据。

对于一个 Wavefront，不同 Lane 通常拥有各自的 VGPR 值。

例如：

```cpp
float a = input[idx];
float b = a * 2.0f;
```

编译器可能把 a、b 保存到 VGPR 中。

VGPR 常用于：

- 向量计算的输入与输出。
- GEMM 中的矩阵片段。
- 中间累加结果。
- Thread 私有的临时变量。
- 部分地址计算结果。

**VGPR 的优化核心是控制寄存器压力。**

如果一个 Kernel 每 Thread 需要大量 VGPR，就会占用更多硬件寄存器资源，可能减少能够同时驻留的 Wavefront。

如果寄存器需求超过可用范围，编译器可能把部分数据溢出到 Scratch Memory。

Scratch 虽然用于保存线程私有数据，但通常由显存体系承载，访问成本远高于寄存器。

### 4.2 SGPR：标量寄存器

SGPR（Scalar General Purpose Register）主要保存 Wavefront 级别共享的标量值。

典型用途：

- Kernel 参数。
- 部分基地址。
- 循环控制变量。
- Wave-uniform 标量计算结果。

SGPR 和 VGPR 的核心差别：

| 对比 | VGPR | SGPR |
|---|---|---|
| 逻辑数据范围 | 不同 Lane 可以有不同值 | 通常保存 Wave-uniform 值 |
| 主要执行路径 | VALU、向量访存、部分矩阵计算 | SALU、标量操作 |
| 主要用途 | 每线程数据、矩阵片段、累加结果 | 地址、参数、循环控制 |
| 性能影响 | 寄存器压力、Occupancy、Spill | 标量资源限制、指令效率 |

需要注意，程序变量具体映射到哪种寄存器，由编译器的数据流分析和目标指令决定，并不完全取决于源代码变量类型。

### 4.3 LDS：片上共享存储

LDS（Local Data Share）是一种由软件显式管理的高速片上存储。

在 HIP 中通常通过 `__shared__` 声明数据。

例如：

```cpp
__shared__ float tile[16][16];
```

它可以被同一 Block 的多个 Thread 使用。

LDS 最重要的两个用途：

**1. 数据复用**

多个线程需要使用相同的数据时，可以先从 Global Memory 加载到 LDS，再由多个线程重复读取。

例如 GEMM 中的 A/B Tile。

**2. 线程间通信**

一个 Thread 可以向 LDS 写入数据，另一个 Thread 在适当同步后读取。

例如 Reduction、Transpose 和 LSTM 中间状态共享。

LDS 属于 CU 级物理资源，但数据的逻辑共享范围通常是单个 Workgroup。

多个同时驻留在同一个 CU 上的 Block，可以使用同一个物理 LDS 容量的不同分配区域，但不能因此直接共享数据。

### 4.4 LDS Bank Conflict

LDS 通常由多个 Bank 组成，以提高并行访问能力。

不同线程访问不同 Bank 时，可能并行完成访问。

当多个访问集中到同一个 Bank，且硬件无法合并为广播等特殊操作时，就可能发生 Bank Conflict。

例如：

```text
理想访问：
Lane 0 → Bank 0
Lane 1 → Bank 1
Lane 2 → Bank 2
...

冲突访问：
Lane 0 → Bank 0
Lane 1 → Bank 0
Lane 2 → Bank 0
...
```

以 E02 使用的 **32 bank、每个 bank 4 字节** 教学模型说明地址映射。实际服务方式还取决于访问宽度、指令和目标设备：


$$\text{Bank ID} = \left\lfloor \frac{\text{Byte Address}}{4} \right\rfloor \bmod 32$$


理解这个公式能直接解释许多常见现象：

- **32-bit 读取：** 在上述模型中，32 个 lane 连续访问不同的 `float` 地址时落到不同 bank。行跨度为 32 时按列读，不同地址会映射到同一个 bank。实际耗时与冲突不是简单的固定倍率。
- **相同地址的读取：** 硬件可能用广播服务多个 lane，需区分同地址和同 bank 的不同地址。
- **宽指令访问：** 每个 lane 访问 8 或 16 字节时会覆盖多个 bank，分组与冲突规则看目标 ISA 和对应指令。

常见的 LDS 优化方法包括：

- Padding（例如将 `__shared__ float tile[32][32]` 改为 `tile[32][33]`，错开列方向的 Bank 对齐）。
- Swizzle Layout（通过异或 `row ^ col` 重排存储索引，在适用的访问模式下分散 bank 访问，且不增加布局的行跨度，在 FlashAttention / GEMM 中极为常用）。
- 调整数据排列方式与向量化读写粒度。
- 改变线程与数据的对应关系。
- 减少不必要的 LDS 访问（能用寄存器 Shuffle 解决的就不进 LDS）。

这也是为什么两个数学运算完全相同的 GEMM Kernel，可能仅因 LDS Layout 不同就产生明显性能差异。

### 4.5 Cache：L1、L2 与全局内存合并访问（Coalescing）

GPU Cache 主要用于降低对更远端存储的访问成本。

一般可以理解为：

- **Vector L1** 靠近计算单元，服务部分向量加载。E06 在 BW100 上分别观察标量与向量路径的延迟变化。
- **共享 L2** 为多 CU 的部分访存请求提供复用。E06 通过 TCC 计数观察到达对应层级的访问。
- **HBM** 为设备数据提供大容量存储，例如输入、权重和输出。

但不同 DCU 型号中 L1/L2 的实际结构、容量、Cache Line 大小和一致性机制需要通过实测或芯片手册分别确认。

跨 Block 通信使用 HIP 提供的原子、同步与内存语义。缓存修饰符需结合目标 ISA 解读。

LDS 和 Cache 的区别是：

**LDS 主要由程序显式管理，Cache 主要由硬件缓存机制管理。**

使用 LDS 不代表所有数据都必须经过 L1/L2，也不代表所有 Kernel 都需要先将数据复制到 LDS。

#### 合并访存（Memory Coalescing）与向量化访存

理解全局内存（HBM / L2 / L1）性能的关键在于 **Memory Coalescing**：

- **合并访存**看相邻 lane 的地址能否由少量事务服务。连续访问通常比大步长散布到更多地址区域更有效。实际事务粒度由硬件和路径决定。
- **向量化访存**看每个 lane 的单条指令搬多少数据。例如对齐且编译成宽指令的 `float4` 访问可搬 16 字节。合并度与向量宽度可单独变化，E01 分别测量它们。

### 4.6 HBM：全局显存

HBM 用于保存大容量模型数据，是大模型推理中非常重要的硬件资源。

典型数据包括：

- 模型权重。
- 输入输出 Tensor。
- 中间激活。
- KV Cache。
- 多个 Kernel 之间传递的数据。

HBM 的两个重要性能指标：

**带宽：** 单位时间能够传输多少数据。

**延迟：** 一次数据访问从发起到获得结果需要多长时间。

高带宽不等于低延迟。

GPU 通过大量并行访存请求、缓存复用及 Wavefront 调度等机制提高整体有效吞吐。

## 区分三种等待

| 谁在等 | 在等什么 | 举例 |
|---|---|---|
| 这个 wave 的后续指令 | 对应类别的未完成操作降到指令要求的阈值 | 向量加法路径中的 `s_waitcnt vmcnt(0)` 等待本例的向量内存操作结果 |
| 同一个 Block 的线程 | 到达同一个屏障，协调共享数据访问 | 每个线程写入 LDS，对等线程需要在 `__syncthreads()` 后才能按约定读取 |
| Host | 已提交工作的完成 | `hipStreamSynchronize(stream)` 等待该 stream 中的工作 |

可以按顺序复述：**加载发出 → 依赖满足 → 消费寄存器结果 → 写回**。如果该步骤还向同一 Block 的其他线程交付 LDS 数据，就在生产者写入和消费者读取之间加入合适的 Block 同步。Host 想知道何时可以读取最终输出，再从 Host 侧等待相关工作完成。具体等待计数器范围以目标 ISA 为准。[LLVM AMDGPU 同步参考](https://llvm.org/docs/AMDGPUExecutionSynchronization.html)

## 案例 · 从全局地址到计算

<MicroarchDiagram kind="memory" />

向量加法按 lane 读取各自元素，线程之间不交换数据，因此主例没有分配 LDS。分块矩阵乘加等需要线程协作或数据复用的 Kernel 可以用 LDS 暂存 tile。

## 跟一条加载指令

下面节选了主例在 BW100、DTK 26.04、DCC 25.10.0、`-O3` 下生成的设备 ISA。省略两次加载间的地址计算和其他指令。

```asm
global_load_dword  v4, v[2:3], off
; 省略地址计算
global_load_dword  v2, v[2:3], off
; 省略其他指令
s_waitcnt vmcnt(0)
v_add_f32_e32      v2, v4, v2
global_store_dword v[0:1], v2, off
```

两条加载的结果分别落在 `v4` 和 `v2`。依赖数据准备好后，向量加法才能使用它们。等待发生前，CU 可能发射同一 wave 中可执行的独立指令，也可能调度其他就绪 wave。[主例和编译命令](../01-overview/software-stack.md#动手运行)

## 用三组数据检验直觉

- **E01 合并访存。** 比较相邻 lane 地址间距和每线程访问宽度。合并度和向量化是两组独立对照，图中的 GB/s 按逻辑读写量计算。[E01](../labs/lab01-memory-coalescing.md)
- **E02 LDS bank 冲突。** 看同一共享内存里，为什么改变地址分布会改变耗时和冲突计数。只读循环的 Padding、XOR 结果提供了对照。[E02](../labs/lab02-lds-bank-conflict.md)
- **E06 缓存与延迟。** 看不同工作集下标量与向量加载的转折，配合 TCC 命中率图理解证据边界。延迟以 `clock64` tick 计，随路径和访问模式变化。[E06](../labs/lab08-memory-latency-ladder.md)

::: details 复述题 · 写完 LDS 后，只执行 s_waitcnt，其他 wave 就可以读了吗？

`s_waitcnt` 处理执行该指令的 wave 中指定类别的未完成操作。Block 内其他 wave 的到达和共享数据交换需要相应的屏障与内存语义。这个教学场景用 `__syncthreads()` 协调写入和读取。

:::

[继续 1.4 · 看 CU 如何隐藏等待](../04-scheduling/occupancy-and-roofline.md)

## 进阶阅读

::: details 查看访存指令、等待计数器和跨 lane 通信

### 数据与访存路径补充

GPU 的另一个核心问题是数据如何存储、访问和复用。

可以把这些存储资源分为两类：

**软件显式使用的存储资源：** 寄存器、LDS、全局内存。

**硬件缓存与访存体系：** L1、L2、Memory Controller 等。

#### 4.7 VMEM 与数据搬运单元

除了计算单元，还要理解访存指令的执行路径。

在 AMD 风格架构中，常见的逻辑分类包括：

- VMEM：向量内存访问。
- SMEM：标量内存访问。
- LDS/DS：本地共享存储访问。
- Cache 与内存控制器：处理缓存请求和外部显存访问。

例如：

```text
Global Memory Load
        ↓
VMEM 访存请求
        ↓
L1 / L2 / HBM
        ↓
数据返回寄存器
        ↓
VALU / MMAC 消费数据
```

而当程序使用 LDS 时，通常还会涉及：

```text
Global Memory
        ↓
加载到寄存器
        ↓
写入 LDS
        ↓
同步 / 等待数据就绪
        ↓
从 LDS 读取矩阵片段
        ↓
MMAC 计算
```

部分架构支持专门的数据搬运机制，可以缩短或优化上述路径。

##### 异步访存与 `s_waitcnt` 硬件同步机制

理解计算与访存重叠（Overlap）的关键在于：**GPU 的访存指令（VMEM、LDS、SMEM）通常是异步发射的**。

发出加载后，若指令与资源允许，一个 wave 可以继续发射不依赖该结果的独立指令。编译器在需要保证依赖完成的地方插入等待。以下是常见 AMDGPU 计数器的简化说明，具体编码与覆盖范围依目标 ISA 而变：

- `vmcnt(k)`：等待未完成的 **VMEM**（全局显存/L1/L2）指令数量降到 `k` 及以下。
- `lgkmcnt(k)`：等待未完成的 **LDS / SMEM** 指令数量降到 `k` 及以下。

```text
Software Pipelining（双缓冲重叠）在指令流上的本质：
1. 发射第 k+1 轮的 global_load
2. 计算第 k 轮已经就绪的 tile
3. 在消费第 k+1 轮数据前等待所需结果
4. 进入第 k+1 轮计算
```

Profiler 中的等待计数应结合其采样与聚合定义、实际 ISA、内存路径和驻留条件解释。E04 与 E09 展示了如何对照独立链数和配置变化。

具体采用什么路径，需要查看目标 ISA 和实际生成的指令。

#### 4.8 Wavefront 内跨 Lane 通信（Shuffle / DPP）与 LDS 的取舍

在 4.3 节我们提到同一个 Block 内可以通过 LDS 交换数据，但如果只需要在**同一个 Wavefront（64 个 Lane）内部**交换数据（例如 Softmax / RMSNorm 中的 Max / Sum Reduction），往往有更快的硬件路径：

- **HIP 原语：** `__shfl_down`、`__shfl_xor`、`__shfl_up`、`__shfl`。
- **底层硬件机制：** 编译器通常会将其映射为寄存器级的 **DPP（Data Parallel Primitives）** 或 `ds_bpermute` / `v_perm` 类指令，直接在 SIMD 通道或极轻量级交叉开关上完成 Lane 间寄存器值传递。

| 对比维度 | 寄存器级 Shuffle / DPP | LDS (`__shared__`) + `__syncthreads()` |
|---|---|---|
| 通信范围 | 仅限同一 Wavefront 内（64 Lane） | 整个 Block 内（跨 Wavefront） |
| 延迟与开销 | 极低（无需写回 LDS，无需全 Block Barrier） | 较高（需 LDS 读写 + `lgkmcnt` + `s_barrier`） |
| 典型场景 | Wave 内规约（Reduction）、局部数据重排 | 跨 Wave 规约、GEMM Tile 共享、矩阵转置 |

因此，在编写 Reduction 或 Attention Softmax 时，标准做法是 **“先在 Wave 内用 `__shfl_down` / `__shfl_xor` 规约到 Lane 0，再仅由各 Wave 的 Lane 0 写入 LDS 做跨 Wave 汇总”**，这样可以大幅削减 LDS 访问和同步开销。

---

:::
