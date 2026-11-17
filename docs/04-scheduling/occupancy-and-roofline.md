# 第 4 章：Kernel 生命周期、Occupancy 与 Roofline 模型

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
