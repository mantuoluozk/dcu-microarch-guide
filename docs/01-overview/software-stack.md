---
outline: 2
---

<script setup>
import MicroarchDiagram from "../.vitepress/theme/MicroarchDiagram.vue"
</script>

# 1.1 · 从 Kernel 启动到 CU

CPU 启动一个 Kernel 后，DCU 会把这次计算的 Block 分派到计算单元（CU）执行。本节用一次向量加法说明任务是怎么拆分和分派的。

::: info 先认识这几个词

**Kernel** 是提交给 DCU 执行的函数。**Grid** 是一次提交的全部任务，**Block** 是其中一组能共享数据和同步的线程。**Wavefront**（简称 wave） 是设备执行线程时采用的分组。**CU** 是承载这些任务的硬件计算单元。

:::

## 1. 软件任务与硬件位置

| 名称 | 怎么理解 |
|---|---|
| Grid | 一次 Kernel 启动包含的全部 Block。 |
| Block / Workgroup | 一组能协作的线程。在本教程讨论的 BW100 上，一个 Block 的 wave 在同一个 CU 上执行。 |
| Wavefront / HIP warp | 一组共享指令流的线程。BW100 实测为 Wave64。 |
| Thread / Work-item | 同一个概念的两种叫法，指执行一份 Kernel 代码的逻辑线程。 |
| Lane | 一个线程在所属 wave 内的位置，编号从 0 开始。 |
| CU（Compute Unit，计算单元） | DCU 上负责承载和执行 Block 的硬件单元，包含调度、计算和存储资源。 |

在 Wave64 下，同一个 Block 内的线程 0 和线程 64 分别是 **Wave 0 的 Lane 0** 和 **Wave 1 的 Lane 0**。这是两个不同的线程，lane 编号只在各自的 wave 内有效。软件层级与硬件映射可对照 [HIP 编程模型](https://rocm.docs.amd.com/projects/HIP/en/docs-6.3.0/understand/programming_model.html) 和 [BW100 实测基线](../baselines/bw100-gfx936.md)。

## 2. 启动参数与向量加法

例子计算 `C[i] = A[i] + B[i]`，共 1000 个元素，一个线程负责一个元素。启动配置如下：

```cpp
// dim3(4) 是 Block 数，dim3(256) 是每个 Block 的线程数。
// 两个 0 表示不额外申请动态 LDS、使用默认 stream。
hipLaunchKernelGGL(tutorial_add, dim3(4), dim3(256), 0, 0,
                   device_a, device_b, device_c, 1000);
```

**LDS** 是 Block 内的线程可以共享的片上存储。这个例子没有线程间数据交换，因此不使用 LDS。**Stream** 是 Host 用来提交有序设备工作的队列视图。

| 数量 | 怎么得到 | 本例 |
|---|---|---|
| Grid 中的 Block 数 | 启动时传入，Kernel 内可用 `gridDim.x` 查看 | 4 |
| 每 Block 的线程数 | 启动时传入，Kernel 内可用 `blockDim.x` 查看 | 256 |
| 每 wave 的 lane 数 | 取决于目标设备及执行模式，可查询 `warpSize` | 64 |
| 每 Block 的 wave 数 | 线程总数除以 wave 宽度，向上取整 | `ceil(256 / 64) = 4` |

::: tip 可以设的是 Block 数和线程数

在 BW100 上，普通 HIP launch 没有单独设置 wave 数或 wave 宽度的参数。把每 Block 的线程数从 256 改为 128，可把每 Block 的 wave 数从 4 改为 2。

:::

本例每线程算一个元素，所以 `ceil(1000 / 256) = 4` 个 Block 就够。Block 3 对应索引 768–1023，最后一个 wave 覆盖 960–1023，其中 40 个有效，另外 24 个被边界检查挡住。

```cpp
// 将 Block 的起点和线程在 Block 内的偏移相加。
const int i = blockIdx.x * blockDim.x + threadIdx.x;
// 边界检查挡住多启动的线程。
if (i < count) {
    c[i] = a[i] + b[i];
}
```

<MicroarchDiagram kind="mapping" />

「驻留」指 Block 已分配到某个 CU，并占用执行所需的资源。同驻一个 CU 的两个 Block 仍各自拥有 Block 内的同步范围。即使 BW100 有 64 个 CU，这个例子也只提交了 4 个 Block。

## 3. 从 Host 提交到设备完成 {#一次-launch-怎么结束}

1. **提交。** Host 是运行在 CPU 上的主机端。它通过 HIP Runtime 指定 Kernel、启动配置和参数。普通 launch 可以先返回给 Host，设备继续执行。
2. **分发。** 设备侧把 Block 交给资源可用的 CU，CU 为驻留的 wave 和 Block 分配所需资源。
3. **执行。** CU 从驻留的 wave 中选择就绪指令发射。向量加法里的有效线程读取 A、B，做加法，写回 C。
4. **完成。** Block 执行结束后释放资源。Host 可用同步调用或事件观察已提交工作何时完成。

::: details 进阶阅读 · 软件栈怎样接上设备

应用可以直接用 HIP 提交手写 Kernel，也可以经由 PyTorch 等框架选择算子库（如 rocBLAS、lightop）或由 Triton 编译的 Kernel。Triton 编译会经过 TTIR、TTGIR 和 LLVM IR 等中间表示，再生成目标设备代码。常见的 DTK/HSA 路径中，Runtime 将 Kernel 派发信息写入设备队列中的 AQL 包，再用 doorbell 通知设备有新工作。驱动提供设备访问、队列和内存映射所需的支持。具体算子实际走了哪套实现，可以用 [Profiling 工具](../profiling/01-toolchain-overview.md) 查看派发记录。

:::

### 硬件结构参考 {#_1-2-dcu-的物理硬件组织}

命令处理器（Command Processor，CP）接收设备队列中的派发命令，设备再将 Block 分到各 CU。早期 z100 类架构可作为理解设备功能分工的参考：

```text
命令处理与任务分发
  └─ 多组 CU：wave 调度、标量/向量执行、寄存器、LDS、近端缓存
       └─ 片上互连与共享 L2
            └─ 显存控制器与 HBM
```

这张图描述各部分的职责。具体 SIMD 组织、缓存共享范围和矩阵执行资源随目标而变。下一节拆开看 [CU 内的执行路径](../02-compute/cu-and-simt.md)。

## 4. 动手实战 · 跑通第一个向量加法 {#动手运行}

**ISA**（Instruction Set Architecture，指令集架构）规定设备认识哪些指令、这些指令如何工作。编译器把 Kernel 翻译成目标 DCU 能执行的机器指令，`-save-temps` 保留人可读的设备端 `.s` 汇编文件。比如一行数组加法可能包含两次加载、等待、加法和写回。[1.3 的真实汇编片段](../03-memory/memory-hierarchy.md#跟一条加载指令)

以下命令在 **装有 DTK 且能访问 DCU 的 Linux 终端**里执行，假设当前目录是教程仓库的根目录，也就是能看到 `experiments/` 的位置。先运行 `hipcc --version` 确认编译器可用，再用 `rocminfo` 找到设备条目。教程命令采用 BW100 的 `gfx936`，其他卡应当填自己的目标，并按实际 wave 宽度理解结果。[实验环境检查](../labs/lab-setup.md)

```bash
cd experiments
hipcc -O3 -std=c++17 --offload-arch=gfx936 -save-temps vector_add_tutorial.hip -o tutorial_add
./tutorial_add
```

BW100 实机输出：

```text
BW100 gfx936:sramecc+:xnack- | 4 blocks x 256 threads | wave size 64 | 1000 outputs verified, 24 untouched
```

程序在 Host 上逐项检查了 1000 个有效结果，并确认输出末尾 24 个预留位置保持原值。

第一次阅读源码时，可以只找这几处。先找复制 A、B 的两行 `hipMemcpy(..., hipMemcpyHostToDevice)`，再找 `hipLaunchKernelGGL`，最后找 `hipMemcpyDeviceToHost`，看结果怎样回到 CPU。没有设备时，可以用上面的输出跟着手算，后续几节也提供相应的实测片段。

::: details 查看带注释的完整 HIP 示例

<<< ../../experiments/vector_add_tutorial.hip{cpp}

:::

## 5. 思考题

仍在 BW100 上处理 1000 个元素，每线程算一个元素。把每 Block 的线程数改成 **128**，最少要启动几个 Block？每 Block 有多少个 wave，最后一个 wave 有多少个有效线程？

::: details 查看答案

`ceil(1000 / 128) = 8`，需要 **8 个 Block**。`128 / 64 = 2`，每 Block 有 **2 个 wave**。最后一个 Block 从 `7 × 128 = 896` 开始，有 104 个有效线程，其中最后一个 wave 从索引 960 开始，有 **40 个有效线程**。

:::

[继续 1.2 · CU 怎样执行指令](../02-compute/cu-and-simt.md)
