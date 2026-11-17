# 海光 DCU（BW100）微架构实战教程与实验记录

> **适用对象：** 希望从“会调 PyTorch/HIP API”进阶到“看懂硬件资源、ISA 汇编、Profiler 指标并独立调优高性能算子”的开发者。  
> **配套源码：** 本地 [experiments/](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/)（已同步至容器 `zk-migraphx` 的 `/exp/dcu_microarch_exp`）  
> **理论参考：** [学习总结.md](../architecture/dcu-microarch.md)  
> **文档性质：** 实战教程 + 持续实验日志（每次在 DCU 上跑出新实验后，都会在此处追加新章节与实测记录）

---

## 第 0 课：实验环境准备与软件栈全景（先搞清“代码怎么跑到卡上”）

在写任何高性能算子之前，第一步不是盲目写 Kernel，而是像 `hello-rocm`（`decode-ai-accelerator`）中那样，先摸清两件事：
1. **你的 Python/C++ 代码经过了哪几层软件栈才到达 DCU？**
2. **你手头这张卡的物理家底是什么？**

### 0.1 一键探测软件栈与已安装算子库

在容器内运行 [09_hygon_ops_and_stack.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/09_hygon_ops_and_stack.py)：

```bash
cd /exp/dcu_microarch_exp/advanced
python 09_hygon_ops_and_stack.py
```

该脚本会做三件事：
1. 读取 `/proc/self/maps`，列出 PyTorch 进程实际加载的底层 `.so` 动态库（类似 `ldd` 追踪调用链）。
2. 检查当前环境是否安装了 `triton`、`lightop`、`aiter`、`deep_gemm`、`vllm` 等算子/推理库。
3. 用 `torch.profiler` 抓一次 `torch.matmul` + Elementwise，看底层到底派发了谁的 Kernel。

**BW100 容器（`zk-migraphx`）实测软件栈：**
- **上层框架：** `torch 2.5.1`（`torch.version.hip = 6.3.26093`）、`vllm 0.11.0`
- **算子库与 DSL：** `triton 3.1.0`（后端为 `triton/backends/hcu/`，`arch='gfx936'`）、`lightop 0.6.0`（`aiter` / `deep_gemm` 未安装）
- **DTK 数学库：** `DTK-26.04`（`librocblas.so.4.3`、`libhipblas.so.2.3`、`libhipblaslt.so.0.10`、`libMIOpen.so.1.0`、`librccl.so.1.0`）
- **HSA 运行时：** `/opt/hyhal/lib/libhsa-runtime64.so.1.11.0`（负责用户态 AQL 队列与驱动通信）

### 0.2 一键探测硬件规格

运行 [01_device_info_and_isa.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/01_device_info_and_isa.hip)（`make 01_device_info_and_isa && ./01_device_info_and_isa`）：

| 硬件参数 | BW100 实测值 | 写算子时为什么必须关心它？ |
|---|---|---|
| 架构代号 (`gcnArchName`) | `gfx936:sramecc+:xnack-` | 编译指定 `--offload-arch=gfx936` 可避免生成多余架构 Fatbin |
| CU 数量 | `64` | Grid 中的 Block 数至少要是 64 的整数倍（如 128/256），才能铺满整卡 |
| Wavefront 宽度 (`warpSize`) | `64` | 最小调度与执行单元是 **Wave64**，Block Size 应为 64 的倍数 |
| 每 CU 最大驻留 Wave 数 | `40`（`2560` 线程） | 计算理论 Occupancy 的分母 |
| LDS 容量 (`sharedMemPerBlock`) | `64 KB` | 若单 Block 用 >32 KB LDS，每 CU 最多只能驻留 1 个 Block |
| L2 Cache 容量 | `8192 KB` (`8 MB`) | 超过 8 MB 的工作集一定会打穿到 HBM |
| HBM 显存与位宽 | `63.98 GB`, `4096-bit @ 1800 MHz` | 理论峰值约 `1843.2 GB/s`（实测极限见第 1 课） |

> **如何计算本机的 Roofline 拐点（Ridge Point）？**  
> 拐点 = 峰值算力 ÷ 实测 HBM 带宽。由下文第 1 课与第 7 课实测：  
> $256.80\text{ TFLOPS} \div 1.365\text{ TB/s} \approx \mathbf{188.1\text{ FLOPs/Byte}}$。  
> 记住这个数字：**凡是计算强度低于 188 的算子（如 Norm、Softmax、Elementwise、Decode GEMV），优化的唯一核心就是“少读写 HBM、合并访存、做算子融合”**。

---

## 第 1 课：全局内存合并访存与向量化（为什么 `float4` 快 2.5 倍？）

### 1.1 核心原理
在 Wave64 架构上，64 个线程同时执行一次 `global_load`。硬件内存控制器会把这 64 个地址按 **Cache Line（如 64B / 128B）** 合并成尽量少的内存事务：
- **合并访问（Coalesced）：** 相邻线程访问相邻地址，一次事务喂饱整个 Wavefront。
- **跨步访问（Strided）：** 相邻线程地址间隔很大，每个线程都要单独拉一条 Cache Line，有效数据只占搬运量的几分之一。
- **向量化（`float4` / 128-bit）：** 单个线程一次发射 `global_load_dwordx4`（16 字节），指令数直接减为 `1/4`，大幅减轻指令发射队列压力。

### 1.2 动手实验与 BW100 实测数据
代码见 [02_coalescing_and_vec.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/02_coalescing_and_vec.hip)（拷贝两块 256 MB 显存，确保穿透 8 MB L2，直接打满 HBM）：

```bash
./02_coalescing_and_vec
```

| 访问模式 | 耗时 (ms) | 实测带宽 (GB/s) | 结论 |
|---|---:|---:|---|
| `Coalesced float (32-bit)` | `1.0124` | `530.29` | 即使完全合并，32-bit 访存指令发射密度也喂不饱 HBM |
| **`Coalesced float4 (128-bit)`** | **`0.3933`** | **`1364.93`** | **实测 HBM 带宽天花板！比 `float` 快 `2.57x`** |
| `Strided float (stride=2)` | `1.3159` | `408.00` | Cache Line 有效利用率减半 |
| `Strided float (stride=4)` | `2.5414` | `211.25` | 带宽继续腰斩 |
| `Strided float (stride=8)` | `5.2079` | `103.09` | 严重浪费 HBM 带宽 |
| `Strided float (stride=16)` | `10.3145` | `52.05` | 接近 1/10 |
| `Strided float (stride=32)` | `15.4136` | `34.83` | 暴跌至 `float4` 的 **2.6%** |

> **实战口诀：** 写任何访存型算子（如 RMSNorm、Transpose、Embedding），**第一件事就是用 `float4` / `int4`（128-bit）向量化读写，并确保沿着 `threadIdx.x` 连续维度对齐**。

---

## 第 2 课：LDS Bank Conflict 与零开销消除（Padding vs XOR Swizzle）

### 2.1 核心原理
LDS（64 KB/Block）通常划分为 **32 个 Bank × 4 字节宽度**，地址映射公式为：
\[
\text{Bank ID} = \left\lfloor \frac{\text{Byte Address}}{4} \right\rfloor \bmod 32
\]
如果一个 `32×32` 的 `__shared__ float tile[32][32]`：
- **按行读 `tile[r][tx]`：** `tx=0..31` 依次命中 Bank `0..31` → **0 冲突**。
- **按列读 `tile[tx][r]`：** 同一行相邻元素步长为 32 个 `float`，所以所有 `tx` 全都撞在同一个 Bank `r` 上 → **32-way Bank Conflict（硬件被迫串行服务）**。

### 2.2 两种工业界消除方法
1. **Padding（`tile[32][33]`）：** 把列步长从 32 改成 33，$\gcd(33,32)=1$，缺点是多占约 3% LDS。
2. **XOR Swizzle（`tile[tx][r ^ tx]`）：** 用异或重排物理列号，**不浪费 1 字节 LDS**，是 FlashAttention / 高性能 GEMM 的标准写法。

### 2.3 动手实验与 BW100 实测数据
代码见 [03_lds_bank_conflict.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/03_lds_bank_conflict.hip)（Kernel 内做 2048 轮 LDS 读改写，放大冲突代价）：

```bash
./03_lds_bank_conflict
```

| LDS 访问模式 | 耗时 (ms) | 相对无冲突 |
|---|---:|---:|
| `0. Row-wise (No conflict)` | `0.6578` | `1.00x` |
| `1. Col-wise (32-way Bank Conflict)` | **`12.9097`** | **慢 `19.62x`** |
| `2. Col-wise + Padding [32][33]` | `0.6769` | 恢复 **`19.07x`** |
| `3. Col-wise + XOR Swizzle [tx][r^tx]` | `0.6751` | 恢复 **`19.12x`**（零额外空间） |

---

## 第 3 课：SIMT 分支发散与编译器的“偷懒”优化（If-Conversion）

### 3.1 核心原理
Wave64 内 64 个 Lane 共享一条 PC：
- **Wave-uniform 分支：** 64 个 Lane 走同一边，走 `SCC` 标量跳转（`s_cbranch_scc`），零额外开销。
- **Lane-divergent 分支：** 一半 Lane 进 `if`、一半进 `else`，硬件必须用 64-bit `EXEC` 掩码先屏蔽一半 Lane 跑 `if`，再翻转 `EXEC` 跑 `else`，两路**串行执行**。

### 3.2 动手实验与 BW100 实测数据（附带一个重要发现）
代码见 [04_branch_divergence.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/04_branch_divergence.hip)：

| 分支模式 | 耗时 (ms) | 现象拆解 |
|---|---:|---|
| `0. No branch (Baseline)` | `1.6253` | 基准：纯 FMA 循环 |
| `1. Wave-uniform branch` | `4.0450` | 与 2-way 一样慢！因为分支体只有 1 条 FMA，编译器直接做了 **Predicated If-Conversion**（两路都算再用 `v_cndmask` 选择） |
| `2. 2-way Lane-divergent` | `4.0451` | 同上，触发 `v_cndmask` 化简 |
| `3. 4-way Lane-divergent` | **`11.2303`** | 4 路分支无法用单条 `v_cndmask` 化简，**触发真实 `EXEC` 掩码串行化（慢 `6.91x`）** |

> **教程提示：** 这正好说明了“不能只凭源码猜硬件行为”。当分支体很短时，编译器会自动做 `v_cndmask` 谓词化消除分支；只有分支体较重或多路分支时才会走 `EXEC` 压栈串行。

---

## 第 4 课：Occupancy 悬崖与 TLP / ILP 延迟隐藏

### 4.1 核心原理
GPU 靠两套机制掩盖几百个周期的 HBM 访存延迟：
- **TLP（线程级并行 / Occupancy）：** Wave 0 等 HBM 时，调度器切去执行 Wave 1~31。
- **ILP（指令级并行）：** 单个 Wave 内部连续发射多条互不依赖的 `global_load`，在 `s_waitcnt` 前一起等。

### 4.2 动手实验与 BW100 实测数据
代码见 [05_occupancy_and_ilp.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/05_occupancy_and_ilp.hip)（通过动态 LDS 从 `0 → 48 KB` 强制压低每 CU 驻留 Wave 数）：

| 动态 LDS | 驻留 Block/CU | 驻留 Wave/CU | `Low-ILP` (ms) | `High-ILP` (ms) |
|---|---:|---:|---:|---:|
| `0 KB` | `8` | `32` (`80%`) | **`0.0386`** | `0.0484` |
| `16 KB` | `4` | `16` (`40%`) | `0.0491` | `0.0501` |
| `32 KB` | `2` | `8` (`20%`) | `0.0806` | **`0.0779`** |
| `48 KB` | `1` | `4` (`10%`) | `0.1045` | `0.1218` |

**怎么读这组数据？**
1. 驻留 Wave 从 `32` 降到 `4` 时，`Low-ILP` 耗时从 `0.0386 ms` 恶化到 `0.1045 ms`（**2.71x 退化**），说明单路串行依赖在低 Occupancy 下完全暴露了 HBM 延迟。
2. 在 `8 Waves/CU`（32 KB LDS）时，`High-ILP` 反超了 `Low-ILP`（`0.0779` vs `0.0806 ms`），但在 0 KB / 48 KB 时 `High-ILP` 略慢，因为该脚本中 4 路索引使用了额外的 `% n` 取模 VALU 指令。

---

## 第 5 课：Wavefront 内 Shuffle 规约 vs LDS 规约

### 5.1 动手实验与 BW100 实测数据
代码见 [06_shuffle_vs_lds.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/06_shuffle_vs_lds.hip)（`8192 × 4096` 按行求和）：
- `Pure LDS + Barriers`（8 次 `__syncthreads()`）：`0.1357 ms`
- `Wave Shuffle (__shfl_down) + 1 LDS`（仅 1 次 `__syncthreads()`）：**`0.1286 ms`**（快 `1.06x`）

因为 `8192×4096`（128 MB）本身受限于 HBM 读入时间，省去 7 次 Barrier 带来了约 `7 us` 的纯同步收益。

---

## 第 6 课：用 Triton 与 `lightop` 实战 Memory-bound 算子融合（RMSNorm）

### 6.1 为什么 PyTorch 原生写 RMSNorm 这么慢？
直接写 `(x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)) * w` 时，PyTorch 会依次启动多个独立 Kernel：`mul` → `mean` → `add` → `rsqrt` → `mul` → `mul`，中间结果反复在 **VGPR ↔ HBM** 之间来回搬运。

### 6.2 动手实验：PyTorch vs `lightop` vs Triton
代码见 [07_triton_fused_rmsnorm.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/07_triton_fused_rmsnorm.py) 与 [10_lightop_vs_triton_pytorch.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/10_lightop_vs_triton_pytorch.py)（测试规模 `8192 × 4096` FP16）：

| 实现方式 | 平均耗时 (ms) | 有效 HBM 带宽 (GB/s) | 加速比 | 底层真相 |
|---|---:|---:|---:|---|
| `PyTorch Unfused` | `1.9925` | `67.36` | `1.00x` | 多次 HBM 读写往返 |
| `lightop.fused_rms_norm_contiguous` | `0.1139` | `1178.54` | **`17.49x`** | `torch.profiler` 捕获底层派发 `fused_rms_kernel_contiguous` |
| **`Triton Fused`** | **`0.1037`** | **`1293.82`** | **`19.21x`** | **达到第 1 课实测 HBM 极限带宽（1364.93 GB/s）的 `94.8%`！** |

### 6.3 学会看 Triton 导出的底层 ISA（`07_rmsnorm_isa.s`）
脚本自动打印了 Triton 编译元数据与 AMDGCN 汇编统计：
- `shared=32`：整个 Block（`num_warps=8`）只用了 **32 字节** LDS 来汇总 8 个 Wave 的部分和。
- `permute/dpp=1`、`s_barrier=2`：Wave 内部全部通过寄存器级跨 Lane 指令完成规约，几乎达到物理带宽极限。

---

## 第 7 课：Triton GEMM（`tl.dot` → 矩阵指令）与 `rocBLAS` 差距拆解

### 7.1 动手实验与 BW100 实测数据
代码见 [08_triton_gemm_mmac.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/08_triton_gemm_mmac.py)（`4096 × 4096 × 4096` FP16 矩阵乘）：

| 实现 / 配置 | 耗时 (ms) | 算力 (TFLOPS) | LDS (B) | VGPR / SGPR | 矩阵指令数 | `s_waitcnt` |
|---|---:|---:|---:|---:|---:|---:|
| **`torch.matmul` (rocBLAS)** | **`0.535`** | **`256.80`** | - | - | - | - |
| Triton (`BM=128, BN=128, BK=32, warps=4, stages=1`) | `1.416` | **`97.08`** | `8704` | `143 / 27` | `32` | `27` |
| Triton (`BM=128, BN=128, BK=32, warps=4, stages=2`) | `1.500` | `91.61` | `16384` | `160 / 36` | `64` | `32` |
| Triton (`BM=128, BN=128, BK=64, warps=8, stages=2`) | `2.465` | `55.76` | `32768` | `128 / 44` | `64` | `56` |
| Triton (`BM=128, BN=256, BK=32, warps=8, stages=2`) | `2.566` | `53.56` | `24576` | `128 / 44` | `64` | `65` |

### 7.2 为什么 `stages=2` 反而变慢？为什么 `rocBLAS` 能到 `256.80 TFLOPS`？
1. **Occupancy 被 VGPR 和 LDS 压垮：** `stages=2` 虽然想做双缓冲流水线，但导致 VGPR 从 `143` 涨到 `160`、LDS 从 `8.7 KB` 翻倍到 `16 KB`，大幅压缩了每 CU 可驻留的 Wavefront 数量；而 `BK=64` 更是把 LDS 撑到 `32 KB`，算力直接腰斩到 `55.76 TFLOPS`。
2. **`rocBLAS` 的启示：** 通过 `torch.profiler` 看到 `torch.matmul` 派发的 Tensile Kernel 名为 `Cijk_Ailk_Bljk_HBH_MT256x256x16_SE_AMAS3_BW_DTL0_BL1...`——注意里面的 **`MT256x256x16`**（$K$ 维分块是 **16** 而不是 32/64），小 $K$ 步长能大幅节省 LDS 与寄存器，把更多资源留给大 $M/N$ Tile 和更高的驻留 Wave 数。

---

## 第 8 课（待续）：后续实验路线图与持续记录区

后续我们将按以下路线继续在 `zk-migraphx` 中做实验，并把每次的实测结果与分析直接追加到本教程中：

- [ ] **实验 11（Triton GEMM 调优实战）：** 参考 `MT256x256x16`，测试 `BK=16`、`kpack`、`waves_per_eu`、`matrix_instr_nonkdim` 以及 `triton.autotune`，看能否把 Triton GEMM 从 `97 TFLOPS` 提升到 `180+ TFLOPS`。
- [ ] **实验 12（`lightop` 专项算子拆解）：** 对比 `lightop.gemm_w8a8_asm`、`lightop.m_grouped_bf16_gemm_nt_masked`（MoE Grouped GEMM）、`lightop.fuse_rmsnorm_rope_quant_qkv` 与常规实现的加速比及 `torch.profiler` 调度轨迹。
- [ ] **实验 13（`s_waitcnt` 与 Persistent Kernel 实践）：** 结合你原有的 **H128 LSTM Persistent MMAC Kernel**，用 `hipcc --save-temps` 导出 `.s` 汇编，检查 `.vgpr_count`、是否存在寄存器 Spill（`SCRATCH`），并对比 Persistent vs 非 Persistent 的真实收益边界。
- [x] **实验 14（`hipprof` 硬件 PMC 计数器实测）：** 已完成！详见下文“第 8 课”。

---

## 第 8 课（实验 14）：用 `hipprof` 抓取真实硬件 PMC 计数器——把瓶颈抓个现行

在海光 DTK（`dtk-26.04`）中，对应的底层硬件性能计数器工具是 `/opt/dtk/bin/hipprof`（而不是上游的 `rocprof`）。

### 8.1 如何用 `hipprof` 抓硬件计数器？

最实用的命令是 `--pmc --pmc-type 3`（导出完整 CSV 表格），配合我们编写的解析脚本 [parse_hipprof_pmc.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/parse_hipprof_pmc.py)：

```bash
cd /exp/dcu_microarch_exp
mkdir -p prof_out

# 1. 对任意 HIP 可执行文件或 Python 脚本抓 PMC
hipprof --pmc --pmc-type 3 -o prof_out/exp03 ./03_lds_bank_conflict
hipprof --pmc --pmc-type 3 -o prof_out/exp02 ./02_coalescing_and_vec
hipprof --pmc --pmc-type 3 -o prof_out/exp08 python3 advanced/08_triton_gemm_mmac.py

# 2. 解析并汇总关键硬件计数器
python3 parse_hipprof_pmc.py prof_out/exp03.csv
```

> **你必须认识的 6 个核心 PMC 指标：**
> - `scr`（Scratch Bytes）：只要 `> 0`，说明发生了**寄存器溢出（Register Spill）**，必须立刻警惕。
> - `SQ_INSTS_VMEM_RD / WR`：向量全局访存指令数（向量化 `float4` 后这里会精确降为 `1/4`）。
> - `SQ_LDS_BANK_CONFLICT`：LDS Bank 冲突计数。
> - `SQ_WAIT_INST_LDS`：Wavefront 因为等 LDS 返回而停顿的周期数。
> - `TCC_HIT / TCC_MISS`：L2 Cache（TCC）的命中与未命中次数（32 个通道求和）。
> - `TCP_TCP_TA_DATA_STALL_CYCLES`：向量 L1 / 纹理单元数据停顿周期数。

---

### 8.2 实测一：用硬件计数器验证实验 3（LDS Bank Conflict）

运行 `python3 parse_hipprof_pmc.py prof_out/exp03.csv` 得到：

| Kernel 模式 | LDS (B) | `SQ_INSTS_LDS` | **`SQ_LDS_BANK_CONFLICT`** | **`SQ_WAIT_INST_LDS`** | 耗时 (us) |
|---|---:|---:|---:|---:|---:|
| `0. Row-wise (无冲突)` | `4096` | `16,781,312` | **`0`** | `160,742,378` | `657.67` |
| `1. Col-wise (32-way 冲突)` | `4096` | `16,781,312` | **`1,040,187,392`** | **`2,944,249,187`** | `12,908.98` |
| `2. Padding [32][33]` | `4224` | `16,781,312` | **`0`** | `99,406,587` | `676.26` |
| `3. XOR Swizzle` | **`4096`** | `16,781,312` | **`0`** | `94,050,631` | `674.75` |

- **铁证如山：** Mode 1 执行了完全相同的 `1678 万` 次 LDS 指令，但硬件计数器抓到了超过 **10.4 亿次 `SQ_LDS_BANK_CONFLICT`**，导致 LDS 等待周期暴涨 **18.3 倍**（29.4 亿周期）；而 XOR Swizzle 在**零额外 LDS（`4096B`）**下把 `SQ_LDS_BANK_CONFLICT` 直接清零！

---

### 8.3 实测二：用硬件计数器验证实验 2（Coalescing 与 `float4`）

运行 `python3 parse_hipprof_pmc.py prof_out/exp02.csv` 得到：

| Kernel | `VMEM_RD` 指令数 | `VMEM_WR` 指令数 | `TCC_MISS` (L2 Miss) | `TCP_TA_DATA_STALL` | 耗时 (us) |
|---|---:|---:|---:|---:|---:|
| `copy_float_coalesced` | `1,048,576` | `1,048,576` | `6,291,461` | `7,070,939` | `1011.68` |
| `copy_float4_coalesced` | **`262,144`（精确 1/4）** | **`262,144`（精确 1/4）** | `6,291,461`（相同） | `10,351,276` | **`391.04`** |
| `copy_float_strided (stride=32)` | `1,048,576` | `1,048,576` | **`134,217,736`（暴涨 21.3x）** | **`519,544,084`（暴涨 73x）** | `15,422.14` |

- **铁证如山：**
  1. `float4` 与 `float` 的 L2 Miss 次数一模一样（都是 `6,291,461` 次，因为搬运的都是 512 MB HBM 数据），但 `float4` 的访存指令数精确降为 `1/4`，所以带宽直接翻倍到 `1364.93 GB/s`。
  2. `stride=32` 的指令数和 `float` 完全一样，但非合并访问导致 L2 Miss 暴涨 **21.3 倍**、L1/TA Stall 暴涨 **73 倍**（5.19 亿周期）！

---

### 8.4 实测三：用 PMC 彻底破案——为什么第 7 课的 Triton GEMM 跑不过 `rocBLAS`？

运行 `python3 parse_hipprof_pmc.py prof_out/exp08.csv` 后，我们直接拿到了 `rocBLAS`（`Cijk_..._MT256x256x16`）与 Triton `gemm_kernel`（最后一组配置 `BM=128, BN=256, BK=32`）的硬件级对账单：

| 关键硬件指标 | `rocBLAS` (`MT256x256x16`) | Triton `gemm_kernel` | 差距诊断 |
|---|---:|---:|---|
| 平均耗时 | **`530.78 us`** | `2556.83 us` | 慢 `4.82x` |
| VGPR / SGPR | `212 / 112` | `128 / 48` | - |
| **寄存器溢出 (`Scratch`)** | **`0 B`** | **`96 B`（发生 Spill！）** | **Triton 产生了 96B/thread 栈溢出** |
| 全局读指令 (`SQ_INSTS_VMEM_RD`) | **`1,048,576`** | `8,429,568` | Triton 多发射了 **8x** 全局加载指令 |
| LDS 指令 (`SQ_INSTS_LDS`) | **`4,194,304`** | `21,266,432` | Triton 多做了 **5x** LDS 读写 |
| **`SQ_LDS_BANK_CONFLICT`** | `8,388,608` | **`34,078,720`** | Triton LDS 冲突是 `rocBLAS` 的 **4.06x** |
| **`SQ_WAIT_INST_LDS`** | **`3,934,304`** | **`99,050,034`** | **Triton 有高达 9900 万周期卡在等 LDS（25.2x）！** |
| L2 命中率 (`TCC_HIT %`) | `81.82%` | **`91.17%`** | 证明 Triton 的 `GROUP_M=8` L2 Swizzle 非常有效 |

**破案结论：** Triton GEMM 的 L2 命中率甚至高于 `rocBLAS`（`91.17%`），它慢的真正元凶不是外部显存，而是内部**高达 9900 万周期的 `SQ_WAIT_INST_LDS`（5 倍 LDS 指令 + 4 倍 Bank Conflict）**、**8 倍 `VMEM_RD` 指令**以及 **`96 B` 寄存器 Spill**。这为我们后续做 **实验 11（调优 Triton GEMM）** 指明了精准方向。
