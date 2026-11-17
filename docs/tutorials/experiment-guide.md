# 海光 DCU 微架构实操实验手册

本目录包含 6 个轻量级、可直接用 `hipcc` 编译运行的微基准实验（Microbenchmarks），用于把《[学习总结.md](../architecture/dcu-microarch.md)》中的微架构概念变成手头 DCU 机器上的真实数据与汇编观察。

---

## 快速开始（在 DCU Linux 机器上）

将 `experiments/` 目录同步到 DCU 机器后执行：

```bash
cd experiments
make all      # 编译全部 6 个实验（如需要指定架构可加 ARCH=--offload-arch=gfx906 等）
make run      # 依次运行全部实验
make isa      # 生成并导出关键实验的设备端汇编 (.s) 到 isa_dump/
```

如果机器上安装了 `rocprof` / `rocprofv2`（或海光对应的 profiler 工具），也可以直接抓取硬件计数器，例如：

```bash
rocprof --stats ./03_lds_bank_conflict
```

---

## 实验清单与观察要点

### 实验 1：硬件拓扑探测与 ISA 反汇编观察（`01_device_info_and_isa.hip`）

- **对应章节：** 第 1、2、4、6 章
- **实操目的：**
  1. 获取当前 DCU 的真实 CU 数量、`warpSize`（确认是否 Wave64）、每 CU 最大驻留线程/Wave 数、LDS 容量、显存位宽与理论带宽。
  2. 运行 `make isa` 后，打开 `isa_dump/` 中对应 GPU 架构的 `.s` 汇编文件，搜索 `sample_fma_kernel`：
     - 查看元数据中的 `.vgpr_count`、`.sgpr_count`、`.lds_size`。
     - 观察异步加载 `global_load_dword` 与后续 `s_waitcnt vmcnt(0)` 的位置。
     - 观察 `v_fma_f32`（VALU）与 `s_cbranch`（SALU 分支）如何分工。

### 实验 2：全局内存合并访存与向量化带宽（`02_coalescing_and_vec.hip`）

- **对应章节：** 4.5 节（Memory Coalescing）、4.6 节（HBM 带宽）、6.5 节（Roofline 基准）
- **实操目的：**
  1. 测出当前 DCU 的 **HBM 实测峰值带宽**（`Coalesced float4`），这就是你做 Roofline 分析时的硬件带宽上限。
  2. 对比 `float`（32-bit）与 `float4`（128-bit）的带宽与耗时差异，理解向量化访存如何减少指令发射压力。
  3. 观察 `stride = 2, 4, 8, 16, 32` 时的带宽断崖，直观理解 Cache Line 合并机制失效后的带宽浪费。

### 实验 3：LDS Bank Conflict 与 Padding / Swizzle 消除（`03_lds_bank_conflict.hip`）

- **对应章节：** 4.4 节（LDS Bank Conflict）
- **实操目的：**
  1. 用高频 LDS 读写循环隔离 HBM 干扰，实测 32-way Bank Conflict（列访问 `tile[tx][r]`）相对于无冲突（行访问 `tile[r][tx]`）的延迟惩罚倍数。
  2. 验证两种工业界常用消除手段的效果：
     - **Padding（`[32][33]`）：** 用少量额外 LDS 空间换取无冲突。
     - **XOR Swizzle（`[tx][r ^ tx]`）：** 零额外 LDS 空间消除冲突（FlashAttention / 高性能 GEMM 标配）。

### 实验 4：SIMT 分支发散与 EXEC 掩码开销（`04_branch_divergence.hip`）

- **对应章节：** 2.6 节（SIMT 分支发散与 EXEC / VCC / SCC）
- **实操目的：**
  1. 对比 `Wave-uniform 分支` 与 `Baseline`：体会基于 `SCC` 的标量分支几乎零额外开销。
  2. 对比 `2-way` / `4-way Lane-divergent 分支`：观察同一 Wavefront 内部分支串行化执行导致的耗时成倍增长。
  3. 结合 `make isa` 产出的 `04_test` 汇编，搜索 `s_cbranch_scc` 与 `s_and_saveexec_b64` / `v_cmp_`，亲眼看到两种分支在 ISA 层面的本质区别。

### 实验 5：Occupancy 悬崖效应与 ILP 延迟隐藏（`05_occupancy_and_ilp.hip`）

- **对应章节：** 6.1 ~ 6.4 节（Occupancy、TLP vs ILP）
- **实操目的：**
  1. 通过动态 LDS（0 KB → 16 KB → 32 KB → 48 KB）人为压低每 CU 可驻留的 Block/Wave 数量，观察 `hipOccupancyMaxActiveBlocksPerMultiprocessor` 返回的驻留台阶变化。
  2. 对比 `Low-ILP`（单路串行依赖）与 `High-ILP`（4 路独立加载累加）：
     - 在 `Low-ILP` 下，Occupancy 降到 1 Block/CU 后耗时明显恶化（TLP 不足，无法隐藏访存延迟）。
     - 在 `High-ILP` 下，即使 Occupancy 同样被压低，性能仍能保持坚挺（单 Wave 内部 ILP 扛住了延迟）。

### 实验 6：Wavefront 内 Shuffle Reduction vs LDS Reduction（`06_shuffle_vs_lds.hip`）

- **对应章节：** 4.8 节（Shuffle / DPP vs LDS）、第 8 章（RMSNorm / Softmax）
- **实操目的：**
  1. 对比纯 LDS 树状规约（多次 `__syncthreads()`）与 `__shfl_down` 寄存器规约 + 单次 LDS 汇总的性能差异。
  2. 查看 `isa_dump/` 中 `06_test` 的汇编，对比两个 Kernel 中 `s_barrier` 与 `ds_read`/`ds_write`/`ds_bpermute`（或 DPP）指令数量的变化。

---

---

## 高级算子实验（`experiments/advanced/`：Triton / 软件栈追踪 / 专项算子库）

结合 `hello-rocm` 的 `decode-ai-accelerator` 视角与海光 DCU 算子生态（`triton-das`、`lightop`、`aiter-das`、`deepgemm-das`），在 `experiments/advanced/` 下新增了 3 个 Python 高级实验：

```bash
cd experiments/advanced
python 09_hygon_ops_and_stack.py
python 07_triton_fused_rmsnorm.py
python 08_triton_gemm_mmac.py
```

### 实验 7：Triton Fused RMSNorm vs PyTorch（`07_triton_fused_rmsnorm.py`）
- **对应章节：** 4.8 节（Wave 内规约）、6.5 节（Memory-bound Roofline 融合）
- **实操目的：**
  1. 对比 PyTorch 拆解实现（多次 HBM 读写）与 Triton 单 Kernel 融合（单次读入 + 片上规约 + 单次写回）的真实有效带宽（GB/s）。
  2. 自动导出设备端汇编 `07_rmsnorm_isa.s`，统计 `load_dword`、`s_waitcnt`、`s_barrier` 与 `bpermute/dpp` 指令数。

### 实验 8：Triton GEMM 与 `tl.dot` → MMAC/MFMA 映射（`08_triton_gemm_mmac.py`）
- **对应章节：** 3.5 节（MMAC）、4.5 节（L2 Swizzle）、4.7 节（`s_waitcnt` 流水重叠）
- **实操目的：**
  1. 对比 `torch.matmul`（rocBLAS/hipBLAS）与不同 `(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)` 配置下的 Triton GEMM TFLOPS。
  2. 观察 `num_stages`（Software Pipelining）如何影响 LDS 占用字节数、`vgpr_count`、`s_waitcnt` 与矩阵指令数量。

### 实验 9：软件栈链路追踪与海光算子库探测（`09_hygon_ops_and_stack.py`）
- **对应章节：** 1.3 节（PyTorch → Triton/算子库 → HIP/DTK → HSA → GPU CP）
- **实操目的：**
  1. 自动读取 `/proc/self/maps`，追踪当前 PyTorch 实际加载的 `libamdhip64.so`、`libhsa-runtime64.so`、`librocblas.so` 等底层动态库路径。
  2. 自动探测当前容器是否已安装 `triton`、`lightop`、`aiter`、`deep_gemm` 并列出可用算子接口。
  3. 用 `torch.profiler` 打印真实派发到底层的 Kernel 名称与耗时占比，作为后续替换/对比 `lightop`/`aiter`/`deepgemm` 算子的基准。

---

## 进阶建议：把实验结果写回你的学习笔记

跑完上述实验后，建议在《[学习总结.md](../architecture/dcu-microarch.md)》第 10 章的表格中填入你这台 DCU 的**实测数据**：

1. 实测 CU 数、Wavefront Size、每 CU 最大 Wave 数、HBM 实测峰值带宽（GB/s）。
2. 32-way LDS Bank Conflict 的实测减速比，以及 Padding / Swizzle 的恢复比例。
3. 结合你的 **H128 LSTM Persistent MMAC Kernel**，用 `--save-temps` 导出一份它的 `.s` 汇编，检查它的 `.vgpr_count`、是否生成了预期的矩阵指令（MMAC/MFMA），以及是否存在寄存器 Spill（搜索 `SCRATCH` 或 `buffer_load/store.*offen`）。
