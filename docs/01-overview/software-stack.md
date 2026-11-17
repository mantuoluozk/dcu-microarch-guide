# 第 1 章：DCU 整体架构、执行模型与软件栈链路

!!! tip "小白速读：先用大白话理解这一章"
    把 GPU 想象成一家**大型物流+工厂园区**：**Grid** 是整张生产订单，**Block（Workgroup）** 是一个车间小组，**CU** 是物理车间，**Wavefront（64 个线程）** 是车间里必须“喊同一个口令齐步走”的 64 人方阵。我们在 Python 里写一行 `torch.matmul`，会依次穿过 PyTorch → 算子库/Triton → HIP → HSA 队列，最后由 DCU 的总调度台（CP）把一个个车间小组派发到 64 个 CU 里干活。

## 一、GPU 的整体硬件结构与执行层级

理解 DCU 微架构，首先要区分两个概念：

- **软件执行模型：** 程序员如何组织线程、划分计算任务。
- **硬件执行架构：** GPU 如何分配计算资源、调度线程、执行指令和搬运数据。

二者相互关联，但不是简单的一一对应关系。

### 1.1 软件执行模型

![从软件执行层级到 CU 的计算资源映射（图文笔记图 1）](../assets/images/dcu_note_fig1.png)

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
