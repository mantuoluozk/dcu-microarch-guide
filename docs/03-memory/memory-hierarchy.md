# 第 3 章：存储体系、合并访存与异步等待机制

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
