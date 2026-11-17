# 海光 DCU 微架构与 Kernel 执行机制学习笔记

## 一、GPU 的整体硬件结构与执行层级

理解 DCU 微架构，首先要区分两个概念：

- **软件执行模型：** 程序员如何组织线程、划分计算任务。
- **硬件执行架构：** GPU 如何分配计算资源、调度线程、执行指令和搬运数据。

二者相互关联，但不是简单的一一对应关系。

### 1.1 软件执行模型

在 HIP 编程中，一次 Kernel 启动会形成一个 Grid，Grid 包含多个 Block（Workgroup），每个 Block 内包含多个 Thread（Work-item）。

以 Wavefront Size 为 64、Block Size 为 256 为例：

```text
Kernel Launch
│
└── Grid
    │
    ├── Block 0（256 Threads）
    │   ├── Wavefront 0（Thread 0～63）
    │   ├── Wavefront 1（Thread 64～127）
    │   ├── Wavefront 2（Thread 128～191）
    │   └── Wavefront 3（Thread 192～255）
    │
    ├── Block 1（256 Threads）
    │   └── 4 个 Wavefront
    │
    └── Block N
        └── 4 个 Wavefront
```

各层级的职责如下：

| 概念 | 主要作用 | 与硬件的关系 |
|---|---|---|
| Grid | 组织一次 Kernel 的全部计算任务 | 分布到整张 GPU 的多个 CU |
| Block / Workgroup | 组织可以共享数据并同步的线程 | 一个 Block 被分配到一个 CU |
| Wavefront | 一组共享指令执行流的线程 | GPU 的重要调度与执行单位 |
| Thread / Work-item | 负责具体数据元素的计算 | 对应 Wavefront 中的逻辑 Lane |
| Lane | Thread 在当前 Wavefront 内的位置 | 用于指令执行、数据布局及跨 Lane 通信 |

需要注意，Grid、Block、Wavefront、Thread 是描述程序执行的概念，并不是四种独立的物理计算单元。

### 1.2 DCU 的物理硬件组织

以早期 z100 类微架构为例，可以建立如下整体认知：

```text
DCU GPU
│
├── 命令处理与任务分发
│   ├── 接收 Kernel Dispatch
│   ├── 管理计算任务
│   └── 将 Workgroup 分配给 CU
│
├── Compute Unit 0
│   │
│   ├── 调度与控制单元
│   │   ├── Wavefront Scheduler
│   │   ├── 指令发射与依赖管理
│   │   └── 分支与执行掩码控制
│   │
│   ├── Scalar 执行资源
│   │   ├── SALU
│   │   └── SGPR
│   │
│   ├── Vector 执行资源
│   │   ├── SIMD 0 + VGPR
│   │   ├── SIMD 1 + VGPR
│   │   ├── SIMD 2 + VGPR
│   │   └── SIMD 3 + VGPR
│   │
│   ├── 矩阵计算资源
│   │   └── MMAC / 矩阵乘加指令路径
│   │
│   ├── LDS（片上共享存储）
│   │
│   └── 访存接口与近端 Cache
│
├── Compute Unit 1
│   └── ...
│
├── Compute Unit N
│   └── ...
│
├── GPU 片上互连与共享 L2 Cache
│
└── Memory Controller
    └── HBM / 全局显存
```

这是一张**逻辑功能结构图**，不是具体芯片的物理布局图。

例如矩阵计算资源可能与 SIMD/向量执行资源存在复杂的共享关系；L1、L2 的实际组织与共享范围也因芯片代际而不同。

图中 z100 的部分特性，包括每 CU 四组 SIMD、Wave64 和 CU 内共享 LDS，可以作为学习起点，但不能直接推断所有 BW 系列采用完全相同的组织方式。

### 1.3 从 PyTorch 到 DCU 硬件的软件栈链路

参考 `hello-rocm` 中“解密 AI 加速器（decode-ai-accelerator）”的视角，在实际大模型推理/训练中，一条算子调用是如何穿透软件栈到达上述物理硬件的？

```text
PyTorch / vLLM / SGLang（计算图与调度）
        │
        ├── 通用数学库：rocBLAS / hipBLAS / MIOpen
        ├── 高级融合算子库：lightop / aiter / deepgemm
        └── DSL 编译器：Triton（Python → TTIR → TTGIR → LLVM IR → AMDGCN ISA）
                ↓
        HIP / DTK Runtime（hipLaunchKernelGGL / Stream / Event）
                ↓
        HSA Runtime（用户态队列 AQL Packet、内存管理与信号量）
                ↓
        Kernel Driver（KFD / 硬件驱动提交门铃 Doorbell）
                ↓
        DCU Command Processor (CP) → 分发 Workgroup 到各个 CU
```

理解这条链路有三个直接价值：

1. **定位“到底跑了谁的 Kernel”**：在 PyTorch 中调用一次线性层或 RMSNorm，底层可能走 rocBLAS（`Cijk_...`）、Triton 生成的 `.hsaco`、或者 `lightop`/`aiter`/`deepgemm` 的手写/模板 Kernel，用 `torch.profiler` 或 `rocprof` 才能看清真实落点。
2. **理解 Triton 与手写 HIP 的映射关系**：Triton 的 `tl.load` / `tl.dot` / `num_warps` / `num_stages` 并不是魔法，最终都会被 LLVM 后端编译为同样受 **VGPR、LDS、`s_waitcnt` 和 MMAC** 约束的设备端 ISA。
3. **追踪环境依赖**：通过 `ldd` 或 Python 包路径检查 `torch` 链接的 `libamdhip64.so` / `libhsa-runtime64.so`，可以快速排查版本错配或算子回退问题。

---

## 二、Thread、Wavefront、Block 与 CU 的关系

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

### 2.3 SIMD：真正执行向量指令的硬件资源

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

### 2.5 一个 Block 如何映射到 CU？

假设：

```cpp
kernel<<<grid_size, 256>>>(...);
```

在 Wave64 的 GPU 上，每个 Block 包含：

\[
256/64=4\ \text{Wavefront}
\]

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

### 2.6 SIMT 分支发散与执行掩码（EXEC / VCC / SCC）

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

因此，当一个 Wavefront 内部发生严重分支发散时，两条分支路径会**串行执行**，总指令周期近似等于两段分支之和。在编写边界检查、Masked Attention 或 MoE 路由 Kernel 时，尽量让同一 Wavefront 内的线程走相同分支，是减少有效算力浪费的重要手段。

---

## 三、CU 内部有哪些计算与调度单元？

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

\[
D=A\times B+C
\]

普通 VALU 主要面向逐元素向量运算，而 MMAC 类指令面向矩阵片段的高吞吐乘加。

二者最重要的区别不是单纯的计算精度，而是**计算组织方式与数据布局**。

普通 FMA 通常面向每个 Lane 的标量或向量元素。

矩阵指令则可能要求整个 Wavefront 按特定 Layout 提供输入矩阵片段，并将输出分布到多个 Lane 的寄存器中。

#### MMAC 的关键组成

**1. 指令形状**

例如某种矩阵指令的抽象计算形状：

\[
M\times N\times K
\]

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

## 四、VGPR、SGPR、LDS、Cache、HBM 的关系

GPU 的另一个核心问题是数据如何存储、访问和复用。

可以把这些存储资源分为两类：

**软件显式使用的存储资源：** 寄存器、LDS、全局内存。

**硬件缓存与访存体系：** L1、L2、Memory Controller 等。

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

实际 Bank 映射与地址、访问宽度和硬件规则有关。在常见的 AMD/DCU 架构中，LDS 通常划分为 **32 个 Bank，每个 Bank 宽度为 4 字节（32-bit / 1 DWORD）**：

\[
\text{Bank ID} = \left\lfloor \frac{\text{Byte Address}}{4} \right\rfloor \bmod 32
\]

理解这个公式能直接解释许多常见现象：

- **32-bit 访存（如 `ds_read_b32`）：** 相邻线程访问连续的 `float`（步长 4B）时，依次落入 Bank 0~31，无冲突；但如果步长是 32 的倍数（例如对 `float tile[32][32]` 按列读取），所有线程落入同一个 Bank，引发 **32-way Bank Conflict**，吞吐骤降至 1/32。
- **Broadcast（广播）：** 多个线程读取**完全相同的地址**时，硬件通常能合并为单次读取并广播给所有请求线程，不构成 Bank Conflict。
- **宽指令访存（`ds_read_b64` / `ds_read_b128`）：** 单个线程一次读取 8B 或 16B 时会同时占用 2 个或 4 个连续 Bank；此时一个 Wavefront 往往拆成多个子周期（例如每 16 个或 8 个 Lane 一批）服务，只要同一批次内没有多个 Lane 撞到同一 Bank 即可无冲突。

常见的 LDS 优化方法包括：

- Padding（例如将 `__shared__ float tile[32][32]` 改为 `tile[32][33]`，错开列方向的 Bank 对齐）。
- Swizzle Layout（通过异或 `row ^ col` 重排存储索引，在不浪费额外 LDS 容量的前提下消除冲突，在 FlashAttention / GEMM 中极为常用）。
- 调整数据排列方式与向量化读写粒度。
- 改变线程与数据的对应关系。
- 减少不必要的 LDS 访问（能用寄存器 Shuffle 解决的就不进 LDS）。

这也是为什么两个数学运算完全相同的 GEMM Kernel，可能仅因 LDS Layout 不同就产生明显性能差异。

### 4.5 Cache：L1、L2 与全局内存合并访问（Coalescing）

GPU Cache 主要用于降低对更远端存储的访问成本。

一般可以理解为：

- **Vector L1（TCP）：** 靠近计算单元（通常 CU 私有或紧邻 CU），服务局部向量访存请求；**需要注意的是，不同 CU 的 L1 Cache 之间通常不自动维护硬件一致性**，因此跨 Block 通信（如全局原子操作、Persistent Kernel 标志位轮询）需要通过绕过 L1（如 `glc` / `slc` 缓存策略或 `atomic` 指令）直接与 L2 交互。
- **L2（TCC）：** 覆盖更大范围，由多个 CU 共享，是所有全局访存和原子操作的一级一致性汇聚点。
- **HBM：** 保存模型权重、激活、KV Cache 和其他全局数据。

但不同 DCU 型号中 L1/L2 的实际结构、容量、Cache Line 大小和一致性机制需要通过实测或芯片手册分别确认。

LDS 和 Cache 的区别是：

**LDS 主要由程序显式管理，Cache 主要由硬件缓存机制管理。**

使用 LDS 不代表所有数据都必须经过 L1/L2，也不代表所有 Kernel 都需要先将数据复制到 LDS。

#### 合并访存（Memory Coalescing）与向量化访存

理解全局内存（HBM / L2 / L1）性能的关键在于 **Memory Coalescing**：

- 当一个 Wavefront（64 个 Lane）执行一次全局加载指令时，硬件内存控制器会将这 64 个 Lane 的请求地址按 **Cache Line 粒度（常见为 64B 或 128B）** 进行合并。
- **合并访问（Coalesced）：** 如果 Lane 0~63 访问连续的 64 个 `float`（共 256 字节），硬件只需发起极少数连续的 Cache Line 事务即可满足整个 Wavefront，带宽利用率极高。
- **非合并/跨步访问（Uncoalesced / Strided）：** 如果相邻 Lane 访问步长很大的地址（例如按列访问行主序大矩阵），64 个请求可能分散到 64 个不同的 Cache Line 中，导致有效数据只占实际搬运字节的几分之一，带宽被严重浪费。
- **向量化访存（`float4` / `dwordx4`）：** 让单个线程一次加载 128-bit（4 个 `float`），不仅能将访存指令数减少为 1/4、降低指令发射压力，还能更容易凑满连续的高效内存事务。

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

### 4.7 VMEM 与数据搬运单元

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

#### 异步访存与 `s_waitcnt` 硬件同步机制

理解计算与访存重叠（Overlap）的关键在于：**GPU 的访存指令（VMEM、LDS、SMEM）通常是异步发射的**。

当 Wavefront 发射一条 `global_load` 或 `ds_read` 指令后，硬件并不会立刻阻塞该 Wavefront，而是增加对应的未完成访存计数器，并允许 Wavefront 继续发射后续**不依赖该数据**的独立指令（如地址计算、上一轮 Tile 的 MMAC 计算）。直到真正需要消费该寄存器结果的指令之前，编译器会插入 `s_waitcnt` 指令进行显式等待：

- `vmcnt(k)`：等待未完成的 **VMEM**（全局显存/L1/L2）指令数量降到 `k` 及以下。
- `lgkmcnt(k)`：等待未完成的 **LDS / SMEM** 指令数量降到 `k` 及以下。

```text
Software Pipelining（双缓冲重叠）在指令流上的本质：
1. 发射第 k+1 轮的 global_load  → vmcnt 变为 1（不阻塞）
2. 执行第 k 轮的 MMAC 计算       → 趁数据在路上时占满矩阵单元！
3. s_waitcnt vmcnt(0)           → 此时数据大概率已到达，等待时间极短
4. 进入第 k+1 轮计算
```

如果在 Profiler 中看到某个 Kernel 有大量的 `Waitcnt Stall`，往往意味着 `vmcnt` 或 `lgkmcnt` 阻塞时间过长——要么是驻留 Wavefront 太少（TLP 不足），要么是单 Wave 内部独立指令不够（ILP / 双缓冲预取不足）。

具体采用什么路径，需要查看目标 ISA 和实际生成的指令。

### 4.8 Wavefront 内跨 Lane 通信（Shuffle / DPP）与 LDS 的取舍

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

## 五、Kernel 从启动到执行完成的完整过程

下面把前面所有硬件组件串起来。

假设需要执行一个 GEMM Kernel。

### 阶段一：CPU 发起 Kernel

CPU 通过 HIP Runtime 发起计算任务。

```text
CPU / HIP Runtime
       ↓
Kernel Launch
       ↓
GPU 命令处理
       ↓
创建 Grid / Workgroup 任务
```

GPU 并不需要每执行一个 Thread 都让 CPU 参与调度。

CPU 提交 Kernel 后，GPU 内部负责后续任务分配和执行。

### 阶段二：Workgroup 分配到 CU

GPU 根据 CU 可用资源分配 Workgroup。

需要考虑：

- 可用 Wavefront 驻留槽位。
- VGPR 资源。
- SGPR 资源。
- LDS 容量。
- Workgroup 数量限制。

资源不足时，部分 Workgroup 需要等待。

### 阶段三：Wavefront 建立执行上下文

一个 Workgroup 被拆分成多个 Wavefront。

硬件为其建立执行所需的寄存器和控制状态。

Wavefront 进入可调度状态后，调度器根据指令依赖及执行资源情况选择执行。

### 阶段四：指令被分配到不同执行路径

例如一个 GEMM Kernel 可能依次涉及：

```text
SALU
  └── 地址和循环控制

VMEM
  └── 从全局内存加载矩阵数据

LDS
  └── 数据共享与复用

VALU
  └── 数据转换、重排、逐元素处理

MMAC
  └── 矩阵乘加计算

VMEM
  └── 写回输出
```

这些操作并不一定完全串行。

经过合理的 Kernel 设计，数据加载、矩阵计算和部分独立指令可以通过流水线方式重叠。

### 阶段五：同步和结果写回

如果多个 Wavefront 需要共享数据，必须保证数据写入完成后，消费者才能读取。

例如：

```cpp
// 多个线程写入共享数据
shared_buf[threadIdx.x] = value;

__syncthreads();

// 同一个 Block 的线程读取共享数据
float x = shared_buf[other_idx];
```

对于更复杂的异步搬运及矩阵指令，还可能需要额外的等待与同步机制。

`__syncthreads()` 是 Block 级同步，并不是所有 GPU 或所有 Block 的全局同步。

### 阶段六：释放资源

当 Block 完成后，其寄存器、LDS 和驻留上下文资源被释放。

GPU 可以继续调度等待执行的其他 Workgroup。

整个过程的关键在于：

**GPU 不仅需要计算单元足够快，还需要调度器不断提供可执行任务，访存系统及时提供数据，寄存器和 LDS 能够支持足够的数据复用与并行度。**

---

## 六、Occupancy、流水线与延迟隐藏

### 6.1 Occupancy 是什么？

Occupancy 用于衡量 GPU 上可驻留的 Wavefront 数量与硬件上限之间的关系。

例如某个 CU 最多可以驻留 40 个 Wavefront，而当前 Kernel 实际驻留 20 个：

\[
Occupancy=20/40=50\%
\]

这是一个简化示例。

实际还需要区分：

- 理论 Occupancy。
- 实际驻留 Wave 数。
- 活跃执行比例。
- 执行单元利用率。

这些指标不能混为一谈。

### 6.2 为什么寄存器和 LDS 会影响 Occupancy？

一个 CU 能够同时驻留多少个 Block，受到多个资源限制。

可以近似理解为：

\[
N_{\text{Blocks}}
\leq
\min
\left(
N_{\text{LDS}},
N_{\text{VGPR}},
N_{\text{SGPR}},
N_{\text{Wave}},
N_{\text{BlockSlots}}
\right)
\]

实际资源分配是按**硬件分配粒度（Allocation Granularity）**进行的，而不是连续线性分配。例如在典型的 GCN/DCU 架构上：

- **VGPR 按 SIMD 划分且按块分配**（常见粒度如 4 个或 8 个 VGPR 为一块，总池子例如每 SIMD 256 个 VGPR）。假设每 SIMD 最多驻留 10 个 Wavefront，那么只要单线程 VGPR 超过 25（向上取整到粒度 28 或 32），单 SIMD 能驻留的 Wavefront 数就会瞬间从 10 掉到 8 或 7；若超过 64，就最多只能驻留 4 个。
- **SGPR 与 LDS 同样存在分配粒度**（如 SGPR 按 8/16 个对齐，LDS 按 256B/512B 对齐）。
- 这种现象被称为 **Occupancy 悬崖效应（Cliff Effect）**：有时仅仅因为多定义了一个临时变量导致 VGPR 从 32 变成 33，Occupancy 就可能直接掉一个台阶（例如从 50% 掉到 37.5%）。因此，使用 `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` 可以强制编译器把 VGPR 压在某个阈值以内，保住目标驻留 Wave 数。

例如，在具有 64 KB LDS 的 CU 上：

- 一个 Block 使用 16 KB，LDS 容量最多支持 4 个这样的 Block。
- 一个 Block 使用 32 KB，最多支持 2 个。
- 一个 Block 使用 48 KB，最多支持 1 个。

这些只是 LDS 单项资源的上限。

如果寄存器或 Wavefront 数量先达到限制，实际驻留 Block 数可能更低。

### 6.3 为什么高 Occupancy 不一定更快？

假设一个 GEMM Kernel：

方案 A：

- 每个 Thread 使用较少寄存器。
- Occupancy 较高。
- 但是矩阵 Tile 小，数据重复加载较多。

方案 B：

- 每个 Thread 使用更多寄存器。
- Occupancy 较低。
- 但矩阵 Tile 更大，数据复用更好，MMAC 执行效率更高。

方案 B 仍然可能更快。

因此 Occupancy 只是资源与延迟隐藏能力的一个指标，不能直接代表最终性能。

### 6.4 延迟隐藏与指令流水

假设 Wave 0 正在等待一次 HBM Load。

调度器可以在资源允许时执行其他 Wavefront。

这属于 Thread-level Parallelism（TLP）带来的延迟隐藏。

另外，一个 Wavefront 内部也可能存在多条彼此独立的指令，通过流水线执行形成 Instruction-level Parallelism（ILP）。

对于 GEMM 和 MMAC 优化，需要同时关注：

- 是否有足够的 Wavefront 隐藏访存延迟。
- 是否有足够独立的矩阵指令隐藏计算延迟。
- 数据加载能否提前进行。
- LDS 和寄存器是否提供充分的数据复用。
- 资源使用是否影响后续任务的驻留。

### 6.5 Roofline 模型：定量区分算力瓶颈与访存瓶颈

在优化任何一个 Kernel 之前，先做一次 Roofline 定量估算，可以避免方向性错误（比如对一个 Memory-bound 算子盲目调高计算指令并行度）：

- **计算强度（Arithmetic Intensity, AI）：**

\[
\text{AI} = \frac{\text{总浮点运算次数 (FLOPs)}}{\text{从 HBM 读写的总字节数 (Bytes)}}
\]

- **硬件拐点（Ridge Point）：**

\[
\text{Ridge Point} = \frac{\text{目标精度峰值算力 (TFLOPS)}}{\text{HBM 实测峰值带宽 (TB/s)}}
\]

- 当算子的 $\text{AI} < \text{Ridge Point}$ 时，算子处于 **Memory-bound（访存受限）** 区间，理论上限由 $\text{AI} \times \text{HBM 带宽}$ 决定（例如向量加法、RMSNorm、Batch=1 的 Decode GEMV / LSTM 单步）；此时优化的核心是合并访存、向量化加载、算子融合（减少写回再读出）以及提高数据复用。
- 当算子的 $\text{AI} > \text{Ridge Point}$ 时，算子处于 **Compute-bound（算力受限）** 区间，理论上限由峰值算力决定（例如大矩阵 GEMM、Prefill Attention）；此时优化的核心是提高 MMAC 利用率、指令流水（`s_waitcnt` 重叠）、消除 LDS Bank Conflict 和避免寄存器 Spill。

---

## 七、结合 LSTM Persistent MMAC Kernel 理解硬件协同

这一部分结合之前的 H128 LSTM 项目进行理解。

需要说明：以下保留你原笔记中的线程配置、数据结构和计算思路，但尚未重新检查完整的 adaptive_lstm_hip.cu 源码，因此具体的 Batch 与 Wave 分工映射仍应以代码为准。

### 7.1 LSTM 的计算结构

LSTM 当前时间步依赖：

- 当前输入 \(x_t\)。
- 上一时间步 Hidden State \(h_{t-1}\)。
- 上一时间步 Cell State \(c_{t-1}\)。

主要矩阵计算为：

\[
G_t=x_tW_{ih}^{T}+h_{t-1}W_{hh}^{T}+b
\]

可以拆成两个部分。

**Input Projection**

\[
x_tW_{ih}^{T}
\]

在输入序列已知的情况下，各时间步的这部分矩阵计算不依赖先前的 Hidden State，因此可以提前批量计算。

**Recurrent Projection**

\[
h_{t-1}W_{hh}^{T}
\]

该部分依赖上一时间步的 Hidden State，必须遵循时间递推关系。

### 7.2 LSTM 为什么有四个 Gate？

对于 Hidden Size = H 的 LSTM：

\[
G_t\in \mathbb{R}^{4H}
\]

它包含四个 Gate 的输入：

- Input Gate。
- Forget Gate。
- Cell Candidate。
- Output Gate。

随后计算：

\[
i_t=\sigma(g_i)
\]

\[
f_t=\sigma(g_f)
\]

\[
\tilde c_t=\tanh(g_c)
\]

\[
o_t=\sigma(g_o)
\]

\[
c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t
\]

\[
h_t=o_t\odot\tanh(c_t)
\]

这里的 \(\odot\) 表示逐元素乘法。

以 H128 为例：

- Hidden State 长度：128。
- Cell State 长度：128。
- Recurrent 输入矩阵长度：128。
- 四个 Gate 的总输出长度：512。

因此，Recurrent Projection 可以表示为：

\[
[1,128]\times[128,512]\rightarrow[1,512]
\]

具体矩阵是否带转置，应按照代码中的权重存储方式确认。

### 7.3 为什么使用 MMAC？

Recurrent Projection 的主要计算是矩阵乘法。

使用普通向量 FMA 虽然可以实现，但可能无法充分利用专门的矩阵计算能力。

MMAC 可以将矩阵计算组织为多个 Tile，由 Wavefront 协同执行。

例如，假设目标输出分块宽度为 16：

- 单个 Gate 的 128 个输出可以划分为 8 个 16 列 Tile。
- 四个 Gate 的 512 个输出总共对应 32 个 16 列 Tile。

这是你原笔记中需要补充的一个重要细节：

**H128 划分为 8 个 16 列 Tile，描述的是 128 维范围；如果讨论四个 Gate 的完整 Recurrent 输出，则总维度是 512，需要考虑全部 32 个输出 Tile。**

实际 Kernel 可能通过 Gate 融合、Batch 合并或者更大的 Wave Tile 调整计算组织，不能只根据 Hidden Size 推断整个循环次数。

### 7.4 为什么选择 256 个 Thread？

你之前的项目使用：

```cpp
__global__ void __launch_bounds__(256)
adaptive_lstm_mmac_packed_kernel(...)
```

并以每 Block 256 个线程启动。

在 Wave64 模型下：

\[
256=4\times64
\]

所以一个 Block 包含四个 Wavefront。

如果每个 Wavefront 负责一个 16 列输出 Tile，那么四个 Wavefront 在一个阶段可以覆盖四个 Tile。

例如对于某一 Gate 的 128 维输出：

```text
Wave 0 → Tile 0：0～15
Wave 1 → Tile 1：16～31
Wave 2 → Tile 2：32～47
Wave 3 → Tile 3：48～63

下一轮：

Wave 0 → Tile 4：64～79
Wave 1 → Tile 5：80～95
Wave 2 → Tile 6：96～111
Wave 3 → Tile 7：112～127
```

但实际 Kernel 还可能同时处理多个 Batch 样本。

因此需要通过线程索引和输出地址计算，明确以下映射：

\[
(\text{Block},\text{Wave},\text{Lane})
\rightarrow
(\text{Batch},\text{Gate},\text{Output Tile})
\]

这一步应该成为后续分析该 Kernel 的重点。

### 7.5 为什么使用 LDS 保存 Hidden 与 Cell？

你原来记录的 LDS 数组包括：

```cpp
__shared__ half h_buf[2][...];
__shared__ float cell_lds[...];
__shared__ float recur_h[...];
```

可以分别理解为：

**h_buf：** 保存 Hidden State，双缓冲可以让上一时间步的输入与当前时间步的输出使用不同存储区域，避免覆盖仍需要读取的数据。

**cell_lds：** 保存 Cell State，用于下一时间步的更新。

**recur_h：** 保存 Recurrent 计算过程中需要跨线程共享或继续使用的中间结果。

这些数组的具体大小、数据布局以及读写顺序，决定 LDS 占用和同步开销。

### 7.6 Persistent Kernel 如何工作？

传统分步实现可以是：

```text
时间步 t=0
启动 Kernel
计算 h0、c0
写回全局内存

时间步 t=1
启动 Kernel
读取 h0、c0
计算 h1、c1
写回全局内存

时间步 t=2
...
```

而 Persistent Kernel 可以把整个时间循环留在一个长期运行的 Kernel 中：

```text
Kernel 启动
    │
    ├── Block 分配到 CU
    │
    ├── Wavefront 创建并驻留
    │
    ├── LDS 初始化 Hidden / Cell
    │
    └── 时间循环
         │
         ├── t=0
         │   ├── MMAC 计算 Recurrent
         │   ├── Gate 计算与状态更新
         │   └── 写回 LDS
         │
         ├── t=1
         │   ├── 从 LDS 读取上一时刻状态
         │   ├── MMAC 计算 Recurrent
         │   ├── Gate 计算与状态更新
         │   └── 写回 LDS
         │
         └── ...
    │
    └── 最终结果写回 Global Memory
```

Persistent Kernel 的主要潜在收益包括：

1. 减少逐时间步 Kernel Launch 开销。
2. 在同一个 Kernel 中融合 Recurrent、Gate 和状态更新。
3. 在 LDS 或寄存器中复用 Hidden/Cell 等中间状态。
4. 减少不必要的全局显存读写。
5. 减少重复的调度、初始化及部分数据准备成本。

但 Persistent Kernel 也有代价。

如果每个 Block 的 LDS 或寄存器占用过大，会降低 CU 上的并发驻留数量。

如果 Block 数量不足，还可能无法充分利用所有 CU。

因此 Persistent 方案并不是一定比非 Persistent 方案更快。

### 7.7 LSTM Kernel 中各组件如何协同？

可以把一次时间步计算拆分为：

**第一步：Scheduler 调度 Wavefront**

四个 Wavefront 根据 Kernel 的计算任务执行各自的矩阵 Tile。

**第二步：VMEM/LDS 提供计算数据**

从 LDS 读取上一时间步 Hidden State，加载或复用 Recurrent Weight。

**第三步：VGPR 保存矩阵片段**

输入数据按照 MMAC 所需 Layout 分布到寄存器。

**第四步：MMAC 执行 Recurrent Projection**

利用矩阵乘加指令计算对应的 Gate 输入片段。

**第五步：VALU 与特殊数学指令完成 Gate 更新**

执行 Sigmoid、Tanh、逐元素乘法与加法等操作。

**第六步：LDS 保存新的 Hidden/Cell**

通过必要的同步，确保下一个时间步读取到正确数据。

**第七步：继续下一个时间步**

Wavefront 保持在同一个 Kernel 中执行，直至序列处理完成。

### 7.8 LSTM Persistent Kernel 的优化重点

| 硬件组件 | 需要分析的内容 | 可能的优化方向 |
|---|---|---|
| Scheduler / Wavefront | 4 个 Wave 的工作划分与等待情况 | 调整 Wave 数和任务映射 |
| MMAC | 矩阵指令利用率与依赖关系 | 优化矩阵 Tile、提高指令并行度 |
| VGPR | 矩阵操作数和累加结果占用 | 控制寄存器压力、避免 Spill |
| LDS | Hidden/Cell/临时结果存储 | 优化布局、减少重复访问与同步 |
| VMEM | 权重加载和结果写回 | 提高数据复用、改善访存模式 |
| VALU / 特殊函数 | Gate、Sigmoid、Tanh | 计算融合、减少冗余操作 |
| CU | Block 驻留与整体利用率 | 平衡 Persistent 状态规模和并行度 |

---

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