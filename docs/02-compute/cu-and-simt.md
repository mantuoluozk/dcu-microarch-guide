---
outline: 2
---

<script setup>
import MicroarchDiagram from "../.vitepress/theme/MicroarchDiagram.vue"
</script>

# 1.2 · CU、Wavefront 与计算路径

CU 以 wave 为单位执行线程，同一个 wave 里的线程可以处理不同的数据。本章介绍 CU 如何调度 wave，以及标量、向量和矩阵运算各走什么路径。

## 基础概念 · Thread、Wavefront 与 Block

### 2.1 Thread：最小的逻辑计算任务

GPU Thread 是执行 Kernel 程序的一个逻辑实例。

例如：

`int tid = blockIdx.x * blockDim.x + threadIdx.x;`

每个 Thread 拥有自己的逻辑线程编号，可以访问不同的数据。

在向量加法中，不同线程分别计算不同位置的输出。

在 GEMM 中，一个 Thread 可能持有多个输出元素的累加结果。

在 MMAC Kernel 中，Thread 也可能只是负责一个矩阵片段的一部分，并通过 Wavefront 协作完成矩阵计算。

需要注意：

**一个 Thread 并不等于一个独占的物理 ALU。**

硬件通过 SIMD 执行单元，在不同周期复用物理执行资源，从而执行大量逻辑线程。

### 2.2 Wavefront：GPU 的重要调度单位

在 Wave64 架构中，64 个 Thread 组成一个 Wavefront。

例如：

```text
Wavefront 0
├── Lane 0
├── Lane 1
├── Lane 2
├── ...
└── Lane 63
```

每个 Lane 拥有自己的向量数据值，但整个 Wavefront 共享一条指令执行流。

当执行：

`c[i] = a[i] + b[i]`

同一个 Wavefront 的多个 Lane 执行相同的加法指令，但使用不同的操作数。

在 Wave64 环境中，可以通过以下方式理解线程映射：

```cpp
int wave_id = threadIdx.x / 64;
int lane_id = threadIdx.x % 64;
```

如果 `threadIdx.x = 130`，则：

- Wavefront 编号为 2。
- Lane 编号为 2。

这里编号从 0 开始。

为了兼容其他 Wavefront Size，实际代码更适合通过 `warpSize` 或设备属性获取目标硬件的执行宽度，而不是在所有场景中写死 64。

### 2.4 Block：线程协作与资源分配单位

一个 Block 由多个 Thread 组成，可以包含一个或多个 Wavefront。

Block 内线程能够：

- 使用属于该 Block 的 LDS 数据。
- 通过同步操作协调执行。
- 分工执行同一个计算任务。
- 协作完成矩阵 Tile 或 Reduction。

通常一个 Block 会分配到一个 CU，并在该 CU 内执行到结束。

一个 CU 可以同时驻留多个 Block，前提是硬件资源允许。

不同 Block 不共享同一份 LDS，也不能直接依赖普通 `__syncthreads()` 实现跨 Block 同步。

### SIMT 与 SIMD

**SIMT** 描述逻辑线程怎样执行同一 Kernel。一个 wave 的线程可以持有不同数据，也可以根据条件参与或屏蔽一条指令。**SIMD** 描述硬件如何让一条指令处理多份数据。wave 的 lane 数由目标架构和运行模式决定，物理执行宽度还要看具体指令路径。

## 案例 · 一个 wave 怎么执行

接上向量加法。一条向量指令在同一个 wave 中由有效 lane 执行，各 lane 可以持有不同的索引和操作数。调度器在资源和依赖允许时选择就绪 wave。这样一个 wave 发出的加载尚未完成，CU 仍可能发射其他就绪工作。

<MicroarchDiagram kind="lanes" />

BW100 / DCC 的这次编译在边界检查处实际生成了向量比较和 `s_and_saveexec_b64`。可以把 **VCC** 理解为某些向量比较的 lane 结果，把 **EXEC** 理解为控制哪些 lane 参与向量执行的掩码。**SCC** 则承载标量指令设置的条件结果。换一个分支体或编译器，控制流可能变成谓词选择或其他序列。[运行与导出方法](../01-overview/software-stack.md#动手运行) · [E03 分支与 ISA](../labs/lab03-branch-divergence.md)

## 基础概念 · CU 的计算与调度

### 3.1 Scheduler：Wavefront 调度器

Scheduler 的作用是管理已经驻留在 CU 中的 Wavefront，选择能够继续执行指令的 Wavefront。

假设：

- Wave 0 正在等待内存数据。
- Wave 1 有可执行的加法指令。
- Wave 2 正在等待矩阵指令结果。
- Wave 3 已经准备好执行计算。

调度器可以选择适当的就绪 Wavefront 发射指令，让执行单元尽可能保持工作。

这种方式是 GPU 延迟隐藏的重要基础。

但需要注意两个区别：

- **Workgroup 分配：** 决定一个 Block 放到哪个 CU。
- **Wavefront 调度：** 决定 CU 内已经驻留的 Wavefront 何时执行指令。

它们属于不同层面的调度过程。

### 3.2 SALU：标量计算单元

SALU（Scalar Arithmetic Logic Unit）负责 Wavefront 级别的标量运算和部分控制操作。

典型工作包括：

- 标量地址计算。
- 循环控制。
- 分支相关处理。
- 部分 Kernel 参数操作。
- Wave-uniform 数据计算。

SALU 与 SGPR 联系紧密。

例如，同一个 Wavefront 中所有 Lane 使用相同的循环计数器，编译器可能使用标量执行路径处理。

SALU 的价值在于：不必让所有 Lane 都重复执行相同的标量计算。

### 3.3 VALU：向量计算单元

VALU（Vector Arithmetic Logic Unit）负责不同 Lane 的向量运算。

典型工作包括：

- 加法、乘法、FMA。
- 比较和数据转换。
- 逐元素激活。
- 部分索引与向量地址计算。

例如，LSTM 中的逐元素 Gate 更新、Sigmoid/Tanh 相关计算，可能涉及向量指令及特殊数学函数指令。

具体使用哪种执行单元取决于编译器生成的目标 ISA。

### 3.4 特殊函数计算单元

除普通的标量和向量运算外，GPU 还可能提供处理特殊数学运算的功能单元或专门指令。

例如：

- exp
- log
- reciprocal
- rsqrt

这些操作常见于 Softmax、Sigmoid、Tanh 和归一化算子。

在 LSTM 中，Gate 计算需要 Sigmoid/Tanh；在 Attention 中，Softmax 需要指数运算。

所以即使一个模型的大部分 FLOPs 来自矩阵乘法，特殊数学运算也可能影响某些 Kernel 的实际性能。

### 3.5 MMAC：矩阵乘加计算能力

MMAC（Matrix Multiply-Accumulate）是海光 DCU 资料中涉及的矩阵计算技术。

从计算功能上，可以理解为：


$$D=A\times B+C$$


普通 VALU 主要面向逐元素向量运算，而 MMAC 类指令面向矩阵片段的高吞吐乘加。

二者最重要的区别不是单纯的计算精度，而是**计算组织方式与数据布局**。

普通 FMA 通常面向每个 Lane 的标量或向量元素。

矩阵指令则可能要求整个 Wavefront 按特定 Layout 提供输入矩阵片段，并将输出分布到多个 Lane 的寄存器中。

### 主例用到了哪些路径？

<MicroarchDiagram kind="compute" />

| 功能 | 主要职责 | 在主例中如何理解 |
|---|---|---|
| Scheduler | 在可发射的 wave 中选择指令 | 确认输入就绪后，向量加法才消费加载结果 |
| SALU 与 SGPR | 处理可按 wave 表示的标量值、运算与控制 | Kernel 参数、统一地址或循环状态有机会走标量路径，由编译器决定 |
| VALU 与 VGPR | 处理每个有效 lane 各自的数据 | 各 lane 读取自己的两份输入并执行 FP32 加法 |
| VMEM | 发出向量内存操作 | 读取 A、B，并把 C 写回全局地址 |
| MMAC 指令路径 | 同一个 wave 按指令约定协作执行矩阵乘加 | 向量加法用普通 VALU。E07 用单独微基准观察 MMAC |

在 E07 里，BW100 支持的示例指令 `v_mmac_f32_16x16x16_f16` 计算一个 `16×16×16` 的矩阵乘加，FP16 输入、FP32 累加。wave 中的 lane 按指令规定分别持有矩阵片段。寄存器内独立累加链的吞吐和实际 GEMM 的耗时分别由 E07、E09 测量。[MMAC 指令实验](../labs/lab09-mmac-builtin-and-valu.md) · [GEMM 实验](../labs/lab07-triton-gemm-vs-rocblas.md)

## 四个容易混淆的量

| 量 | 回答的问题 |
|---|---|
| Wave 的逻辑宽度 | 这个目标上的一个 wave 有多少个逻辑 lane？本例为 64。 |
| SIMD 的物理执行宽度 | 对应指令路径一次能处理多少个物理 lane？具体组织需要对应架构资料。 |
| 指令依赖延迟 | 上一条结果多久可被下一条依赖指令使用？ |
| 指令吞吐 | 稳态有足够独立工作时，单位时间能完成多少操作？ |

::: details 复述题 · 32 个 lane 进 if，32 个 lane 进 else，时间会翻倍吗？

先检查编译生成了什么。若同一 wave 的两条路径确实分别执行，就要考虑两边的指令量、掩码和访存。E03 的短分支还观察到了 `v_cndmask_b32` 谓词选择，两类源码写法获得近似耗时。用源码里有几路分支，无法直接推导耗时倍率。

:::

[继续 1.3 · 追踪一个操作数](../03-memory/memory-hierarchy.md)

## 进阶阅读

::: details 查看 SIMD、执行掩码与 MMAC 指令细节

### 二、Thread、Wavefront、Block 与 CU 的关系

#### 2.3 SIMD：真正执行向量指令的硬件资源

SIMD（Single Instruction, Multiple Data）是一种并行执行方式。

它通过同一条指令控制多个物理 Lane，对不同数据执行相同运算。

Wavefront 和 SIMD 的区别是：

**Wavefront 是逻辑线程执行组，SIMD 是物理执行资源。**

例如，早期部分 GCN 架构采用：

- 一个 Wavefront 包含 64 个 Thread。
- 一个 SIMD 具有 16 个物理执行 Lane。

那么一条覆盖全部 64 个 Lane 的普通向量指令，可能需要分成四批执行。

```text
Wave64
│
├── 第 1 批：Lane 0～15
├── 第 2 批：Lane 16～31
├── 第 3 批：Lane 32～47
└── 第 4 批：Lane 48～63
```

但不同代际 SIMD 宽度和指令执行方式可能变化，而且矩阵指令不一定遵循普通 VALU 指令的执行规律。

因此需要区分：

1. Wavefront 的逻辑线程数量。
2. SIMD 的物理执行宽度。
3. 单条指令的执行延迟。
4. 指令流水线的吞吐能力。

这四个参数并不是同一个概念。

#### 2.5 一个 Block 如何映射到 CU？

假设：

```cpp
kernel<<<grid_size, 256>>>(...);
```

在 Wave64 的 GPU 上，每个 Block 包含：


256/64=4\ \text{Wavefront}


GPU 将 Block 分配到有可用资源的 CU。

一个可能的执行过程是：

```text
Block 0
    ↓
分配到 CU 3
    ↓
申请该 Block 的寄存器与 LDS
    ↓
创建 4 个 Wavefront
    ↓
由 CU 内调度器安排执行
    ↓
各 Wavefront 执行指令
    ↓
Block 完成，释放硬件资源
```

四个 Wavefront 在创建时通常会按轮询（Round-Robin）方式分配到 CU 内的不同 SIMD 单元（例如在每 CU 4 个 SIMD 的架构中，4 个 Wavefront 往往刚好分散到 4 个 SIMD 上），且一旦分配完成，Wavefront 在其整个生命周期内绑定在该 SIMD 上执行，不会在 SIMD 之间迁移。但这并不意味着它们每个周期都同时执行——实际发射节奏受硬件指令发射口、Wave 就绪状态和执行资源竞争影响。

另外需要注意：`__launch_bounds__(256)` 只是给编译器的**资源上限提示**（用于限制每个线程的寄存器用量以保底 Occupancy），并不决定实际运行时的 Block Size；真正决定每个 Block 线程数的是调用侧 `kernel<<<grid, block>>>` 中的 `block` 参数。

#### 2.6 SIMT 分支发散与执行掩码（EXEC / VCC / SCC）

同一个 Wavefront 内的 64 个 Lane 共享同一条指令流（PC），但实际代码中经常出现 `if / else`。硬件如何让同一条指令流表现出“不同线程走不同分支”的效果？核心依赖 **EXEC（Execution Mask，64-bit 执行掩码寄存器）**：

- `EXEC` 的第 `i` 位对应第 `i` 个 Lane 当前是否处于活跃（Active）状态。
- 当某条向量指令执行时，只有 `EXEC` 中对应 bit 为 `1` 的 Lane 会真正写回结果和产生访存副作用；bit 为 `0` 的 Lane 虽然在同一个物理周期占用了 SIMD 槽位，但其结果被屏蔽丢弃。

根据条件是否全 Wave 一致，分支在硬件上分为两类，性能影响截然不同：

1. **Wave-uniform 分支（标量分支，零发散惩罚）**：
   如果判断条件对整个 Wavefront 的 64 个 Lane 都相同（例如基于 `blockIdx.x`、Kernel 参数或循环次数 `k < K`），编译器会将其编译为标量比较，结果写入 **SCC（Scalar Condition Code）**，再通过标量跳转指令（如 `s_cbranch_scc0/1`）直接跳过未命中的分支块，完全不浪费执行周期。

2. **Lane-divergent 分支（向量分支，串行化执行）**：
   如果判断条件依赖 `threadIdx.x` 或每个线程各自的数据（例如边界检查 `if (idx < N)`、`if (x > 0)`），编译器会将其编译为向量比较，结果写入 64-bit 的 **VCC（Vector Condition Code）**，随后硬件保存当前 `EXEC` 并用 `VCC` 掩码先执行 `if` 分支；完成后再翻转 `EXEC` 执行 `else` 分支；最后恢复进入 `if` 前的 `EXEC`。

```text
分支发散时的硬件执行轨迹：
原始 EXEC = 1111...1111 (64 lanes)
    │
    ├── 向量比较 → 生成 VCC (例如前 32 位为 1，后 32 位为 0)
    ├── 执行 if 路径：EXEC = 原始 EXEC & VCC   (后 32 个 Lane 被屏蔽空转)
    ├── 执行 else 路径：EXEC = 原始 EXEC & ~VCC (前 32 个 Lane 被屏蔽空转)
    └── 汇合点：恢复原始 EXEC = 1111...1111
```

如果目标 ISA 确实用执行掩码分别执行两条路径，实际成本取决于两边的指令、访存与参与的 lane。短分支也可能被编译成谓词选择。参见 [E03](../labs/lab03-branch-divergence.md)。

---

### MMAC 指令深入

#### MMAC 的关键组成

**1. 指令形状**

例如某种矩阵指令的抽象计算形状：


M\times N\times K


其中：

- M：输出矩阵的行数。
- N：输出矩阵的列数。
- K：矩阵乘法中的归约维度。

实际支持的 M/N/K、数据类型和累加方式取决于目标芯片的 ISA。

**2. 操作数布局**

假设一条指令需要 Wavefront 协同完成一个矩阵 Tile。

那么各 Lane 可能分别保存 A、B 矩阵的不同部分：

```text
Wavefront
│
├── Lane 0：A/B 的部分元素
├── Lane 1：A/B 的部分元素
├── ...
└── Lane 63：A/B 的部分元素
             ↓
       MMAC 矩阵乘加
             ↓
       C / D 矩阵片段
       分布在各 Lane 寄存器中
```

这也解释了为什么矩阵指令的操作数 Layout 非常重要。

即使数学上是同一组矩阵数据，如果 Lane 分布和指令要求不匹配，也需要额外的数据重排或寄存器操作。

**3. 指令延迟**

从指令发射到结果可以被依赖指令使用，需要一定时间。

**4. 指令吞吐**

硬件能够以什么速度持续发射或完成独立矩阵指令。

延迟和吞吐是不同概念。

一条指令可能延迟较高，但只要有足够的独立指令和流水线能力，矩阵计算单元仍然可以获得较高吞吐。

#### MMAC 与 ISA、编译器的关系

```text
HIP / Triton Kernel
       ↓
编译器中间表示
       ↓
目标架构代码生成
       ↓
MMAC / VALU / LDS / VMEM 等指令
       ↓
硬件执行
```

因此算子优化不仅要看 C++ 或 Triton 源码，还需要分析最终是否生成了预期的矩阵指令。

其中，MMAC 的具体指令形式、寄存器布局及执行资源组织，需要使用对应海光 DCU 的正式 ISA 资料确认，不能直接照搬 AMD MFMA 的参数。

---

:::
