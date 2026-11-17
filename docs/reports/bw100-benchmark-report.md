# 海光 BW100 DCU 微架构实测与算子实验完整报告

> **实验日期：** 2026-11-17  
> **实验环境：** `10.17.28.67` / 容器 `zk-migraphx`（容器内路径 `/exp/dcu_microarch_exp`，宿主机路径 `/data1/zk/migraphx_work/training_20261112/dcu_microarch_exp`）  
> **核心目标：** 将《[学习总结.md](../architecture/dcu-microarch.md)》中的微架构理论（CU、Wave64、Coalescing、LDS Bank Conflict、EXEC 掩码、Occupancy、Roofline、MMAC/MFMA、Software Pipelining）在真实 BW100 DCU 硬件上通过 **HIP 微基准 + Triton DSL + `lightop` + `torch.profiler`** 逐一验证、量化并形成可复现的工程结论。

---

## 一、实验环境与 BW100 硬件画像

### 1.1 软件栈链路探测过程（实验 9）

结合 `hello-rocm` 中“解密 AI 加速器（`decode-ai-accelerator`）”的软件栈追踪思路，通过 [09_hygon_ops_and_stack.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/09_hygon_ops_and_stack.py) 读取 `/proc/self/maps` 与 Python 模块元数据，确认当前容器内的完整调用栈如下：

```text
PyTorch 2.5.1 (torch.version.hip = 6.3.26093)
   ├── 通用计算库：DTK-26.04
   │     ├── librocblas.so.4.3
   │     ├── libhipblas.so.2.3 / libhipblaslt.so.0.10
   │     ├── libMIOpen.so.1.0 / libMIOpen-recommend.so
   │     └── librccl.so.1.0
   ├── DSL / 专项算子库：
   │     ├── triton 3.1.0（后端：triton/backends/hcu/，arch='gfx936'）
   │     ├── lightop 0.6.0（内置融合 Norm、MoE、W8A8/W4A8 GEMM 等）
   │     └── vllm 0.11.0（注：当前容器未安装 aiter / deep_gemm / sglang）
   └── 底层驱动与运行时：
         └── HSA Runtime：/opt/hyhal/lib/libhsa-runtime64.so.1.11.0
```

### 1.2 BW100 硬件规格实测（实验 1）

编译并运行 [01_device_info_and_isa.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/01_device_info_and_isa.hip)（`hipcc -O3 -std=c++17`），通过 `hipGetDeviceProperties` 读取到的设备参数如下：

| 硬件属性 | 实测值 | 微架构含义 |
|---|---|---|
| 设备名称 / 架构代号 | `BW100` / `gfx936:sramecc+:xnack-` | 海光 BW 系列 DCU，开启 SRAM ECC、关闭 XNACK |
| Compute Unit (CU) 数量 | `64` | 整卡共有 64 个物理 CU |
| `warpSize`（Wavefront 宽度） | `64` | 标准 Wave64 执行模型 |
| 每 CU 最大驻留线程 / Wave 数 | `2560` 线程 / `40` Waves | 满 Occupancy 时每 CU 可同时驻留 40 个 Wavefront |
| 每 Block 最大线程数 | `1024` | 单 Block 最多 16 个 Wave64 |
| LDS 容量（`sharedMemPerBlock`） | `64 KB` | 每个 Block/CU 可用片上共享内存上限 |
| 每 Block 寄存器池（32-bit） | `196608` | 极充裕的标量/向量寄存器资源池 |
| L2 Cache 容量 | `8192 KB` (`8 MB`) | 全卡多 CU 共享 L2；超过 8 MB 的工作集将穿透至 HBM |
| 全局显存（HBM） | `63.98 GB`（`4096-bit` @ `1800 MHz`） | 理论峰值带宽约 `1843.2 GB/s` |
| 核心频率（`clockRate`） | `1300.00 MHz` | 计算单元标称主频 |

> **实测 Roofline 拐点（Ridge Point）估算：**  
> 由实验 2 实测 HBM 有效带宽 $\approx 1364.93\text{ GB/s}$（$1.365\text{ TB/s}$），实验 8 `torch.matmul` 实测 FP16 算力 $\approx 256.80\text{ TFLOPS}$，因此 BW100 在 FP16 下的实测 Roofline 拐点约为：
> \[
> \text{Ridge Point} \approx \frac{256.80\text{ TFLOPS}}{1.365\text{ TB/s}} \approx 188.1\text{ FLOPs/Byte}
> \]
> 即：**计算强度 $< 188\text{ FLOPs/Byte}$ 的算子（如 RMSNorm、Elementwise、Decode 阶段小 Batch GEMV）绝对受限于 HBM 带宽**；只有高计算强度的算子（如大矩阵 GEMM、Prefill Attention）才会触及算力天花板。

---

## 二、HIP 底层微架构实验过程、结果与分析

所有 HIP 微基准实验均通过 `cd /exp/dcu_microarch_exp && make run` 统一编译运行，计时采用 `hipEventElapsedTime`（预热 5 次，正式测量 20 次取均值）。

### 2.1 实验 1：基础 FMA 与编译目标观察
- **文件：** [01_device_info_and_isa.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/01_device_info_and_isa.hip)
- **过程与结果：** 对 $2^{20}$ 个 `float` 执行 `alpha * a[i] + b[i]`，`c[0] = 6.50` 校验通过，平均耗时 `0.0229 ms`。
- **工程观察：** 未传 `--offload-arch` 时，`hipcc` 默认同时为 `gfx906 / gfx926 / gfx928 / gfx936 / gfx938` 生成多目标胖二进制（Fatbin）。针对本机单卡实验，后续建议加 `ARCH=--offload-arch=gfx936` 可大幅缩短编译时间。

---

### 2.2 实验 2：全局内存合并访存（Coalescing）与向量化访存

- **文件：** [02_coalescing_and_vec.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/02_coalescing_and_vec.hip)
- **实验设计：** 分配两块各 **256 MB**（$2^{26}$ 个 `float`，远超 BW100 的 8 MB L2 Cache，确保请求全部打到 HBM）的缓冲区做拷贝（总访存量 = 读 + 写 = 512 MB），对比连续 `float`（32-bit/lane）、连续 `float4`（128-bit/lane）以及步长为 `2 / 4 / 8 / 16 / 32` 的跨步访问。
- **实测结果：**

| 访问模式 | 平均耗时 (ms) | 有效带宽 (GB/s) | 相对 `float4` 带宽比例 |
|---|---:|---:|---:|
| `Coalesced float (32-bit)` | `1.0124` | `530.29` | `38.8%` |
| **`Coalesced float4 (128-bit)`** | **`0.3933`** | **`1364.93`** | **`100.0%`（实测峰值）** |
| `Strided float (stride=2)` | `1.3159` | `408.00` | `29.9%` |
| `Strided float (stride=4)` | `2.5414` | `211.25` | `15.5%` |
| `Strided float (stride=8)` | `5.2079` | `103.09` | `7.6%` |
| `Strided float (stride=16)` | `10.3145` | `52.05` | `3.8%` |
| `Strided float (stride=32)` | `15.4136` | `34.83` | `2.6%` |

- **微架构分析：**
  1. **为什么 `float4` 比 `float` 快 `2.57x`？** 同样的 512 MB 数据，`float4` 让每条 `global_load_dwordx4` 指令搬运 128-bit，访存指令数与指令发射队列压力直接降为 `1/4`，显著提高了每个周期向 HBM 控制器投递的字节密度。
  2. **为什么 `stride` 从 2 到 32 呈线性雪崩？** 当步长为 $S$ 时，同一个 Wavefront 内相邻 Lane 访问的字节间隔为 $4S$ 字节。在硬件按 Cache Line（如 64B / 128B）合并事务的机制下，步长越大，单个 Cache Line 内被当前 Wavefront 实际消费的字节占比越低（从 `stride=4` 到 `stride=16` 每次步长翻倍，带宽几乎精确减半：`211.25 → 103.09 → 52.05 GB/s`），大量 HBM 带宽浪费在搬运相邻未命中的无效字节上。

---

### 2.3 实验 3：LDS Bank Conflict 与 Padding / Swizzle 消除

- **文件：** [03_lds_bank_conflict.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/03_lds_bank_conflict.hip)
- **实验设计：** 申请 `32×32` 的 `__shared__ float` Tile，每个 Block 1024 线程（Grid 256 个 Block），在 Kernel 内部连续做 **2048 轮** LDS 读-改-写（彻底隔离 HBM 延迟），对比 4 种布局：
  - Mode 0：行访问 `tile_std[r][tx]`（相邻 `tx` 落入相邻 Bank，0 冲突）
  - Mode 1：列访问 `tile_std[tx][r]`（`tx=0..31` 全部命中同一 Bank，32-way 冲突）
  - Mode 2：Padding `tile_pad[32][33]` 列访问（列步长变为 33，$\gcd(33, 32)=1$）
  - Mode 3：XOR Swizzle `tile_std[tx][r ^ tx]`（零额外内存开销）
- **实测结果：**

| LDS 访问模式 | 平均耗时 (ms) | 相对无冲突倍率 |
|---|---:|---:|
| `0. Row-wise (No conflict)` | `0.6578` | `1.00x` |
| `1. Col-wise (32-way Bank Conflict)` | **`12.9097`** | **`19.62x` 减速** |
| `2. Col-wise + Padding [32][33]` | `0.6769` | `1.03x`（较冲突加速 **19.07x**） |
| `3. Col-wise + XOR Swizzle [tx][r^tx]` | `0.6751` | `1.03x`（较冲突加速 **19.12x**） |

- **微架构分析：**
  1. 实测 32-way Bank Conflict 带来了接近 **20 倍** 的延迟惩罚，与理论的 32 个 Bank 串行化高度吻合（部分被高循环并发和指令开销摊薄）。
  2. **Padding vs Swizzle：** 二者都能把性能恢复到无冲突水平，但 `Padding` 会额外占用 LDS 容量（对 LDS 本就紧张的 GEMM / FlashAttention 可能触发 Occupancy 下降），而 **XOR Swizzle 不增加 1 字节 LDS 占用**，仅用一条极低成本的 `v_xor_b32` 地址计算即可完全消除冲突。
  3. **编译/运行时提示观察：** 运行时捕获到提示 `Launch params (1024, 1, 1) are larger than launch bounds (256)`——这验证了 DTK/HIP 编译器在未显式标注 `__launch_bounds__(1024)` 时默认按 `256` 线程估算寄存器预算的机制。

---

### 2.4 实验 4：SIMT 分支发散与 EXEC 掩码机制

- **文件：** [04_branch_divergence.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/04_branch_divergence.hip)
- **实验设计：** 在 4096 轮循环中对比：无分支（Mode 0）、Wave-uniform 分支 `(threadIdx.x / warpSize) & 1`（Mode 1）、2-way Lane-divergent `threadIdx.x & 1`（Mode 2）、4-way Lane-divergent `threadIdx.x & 3`（Mode 3）。
- **实测结果：**

| 分支模式 | 平均耗时 (ms) | 相对 Baseline 倍率 |
|---|---:|---:|
| `0. No branch (Baseline)` | `1.6253` | `1.00x` |
| `1. Wave-uniform branch` | `4.0450` | `2.49x` |
| `2. 2-way Lane-divergent` | `4.0451` | `2.49x` |
| `3. 4-way Lane-divergent` | **`11.2303`** | **`6.91x`**（相对 2-way 为 `2.78x`） |

- **微架构分析（一个非常值得记录的编译器行为）：**
  - 为什么 Mode 1（Wave-uniform）和 Mode 2（2-way）耗时几乎完全一致（`4.0450 ms` vs `4.0451 ms`）？
  - 原因有二：(1) 两个分支内部只有 **1 条 FMA**，LLVM 后端对短分支会自动做 **Predicated If-Conversion**（把 `if/else` 消去，改成两条分支都计算再用 `v_cndmask_b32` 选结果），所以 Mode 1 和 Mode 2 实际执行了相同的指令数；(2) `warpSize` 在 HIP 中若被当作运行时变量做整数除法 `threadIdx.x / warpSize`，编译器无法静态证明其 Wave-uniform，仍会走向量路径。
  - 而到了 Mode 3（4-way 分支），分支体变复杂后编译器无法继续用单条 `v_cndmask` 化简，**触发了真实的 `EXEC` 掩码压栈与多路串行执行**，耗时立刻跳升到 `11.2303 ms`（比 2-way 慢 `2.78x`）。

---

### 2.5 实验 5：Occupancy 悬崖与 TLP / ILP 延迟隐藏

- **文件：** [05_occupancy_and_ilp.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/05_occupancy_and_ilp.hip)
- **实验设计：** 固定 Block=256（4 Waves/Block），通过动态申请 `0 / 16 / 32 / 48 KB` LDS 人为限制每 CU 驻留的 Block 数，对比 `Low-ILP`（单路串行加载累加）与 `High-ILP`（4 路独立加载累加）在 64 MB 显存上的耗时。
- **实测结果：**

| 动态 LDS 占用 | 每 CU 驻留 Block 数 | 每 CU 驻留 Wave 数 | `Low-ILP` 耗时 (ms) | `High-ILP` 耗时 (ms) |
|---|---:|---:|---:|---:|
| `0 KB` | `8` | `32` (`80%`) | **`0.0386`** | `0.0484` |
| `16 KB` | `4` | `16` (`40%`) | `0.0491` | `0.0501` |
| `32 KB` | `2` | `8` (`20%`) | `0.0806` | **`0.0779`** |
| `48 KB` | `1` | `4` (`10%`) | `0.1045` | `0.1218` |

- **微架构分析：**
  1. **精确的硬件资源台阶：** `64 KB` LDS 下，随着单 Block LDS 从 `0 → 16 → 32 → 48 KB`，每 CU 驻留 Wave 数呈精确的阶梯下降 `32 → 16 → 8 → 4`。
  2. **TLP 延迟隐藏失效：** 在 `Low-ILP` 模式下，当驻留 Wave 数从 `32` 降到 `4` 时，调度器找不到足够的就绪 Wavefront 来掩盖 HBM 访问延迟，耗时从 `0.0386 ms` 恶化到 `0.1045 ms`（**退化 2.71x**）。
  3. **ILP 的代价与甜点：** 在 `8 Waves/CU`（32 KB LDS）时，`High-ILP`（`0.0779 ms`）反超了 `Low-ILP`（`0.0806 ms`）；但在 0 KB 和 48 KB 时 `High-ILP` 略慢，是因为该微基准中 `High-ILP` 使用了 4 组取模索引 `(idx + k*stride) % n`，引入了额外的整数除法/取模 VALU 指令与寄存器开销。

---

### 2.6 实验 6：Wavefront Shuffle vs LDS 规约

- **文件：** [06_shuffle_vs_lds.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/06_shuffle_vs_lds.hip)
- **实验设计：** 对 `8192 × 4096` 的矩阵做按行求和规约，对比纯 LDS + 8 次 `__syncthreads()` 树状规约 vs `__shfl_down` Wave 内寄存器规约 + 1 次 LDS 跨 Wave 汇总。
- **实测结果：**
  - `Pure LDS + Barriers`：`0.1357 ms`（`out[0] = 4096.0`）
  - `Wave Shuffle + 1 LDS`：**`0.1286 ms`**（`out[0] = 4096.0`，快 `1.06x`）
- **分析：** 对 `8192 × 4096`（128 MB）的数据量而言，整体耗时主要受限于从 HBM 读入 128 MB 数据的带宽（按 1365 GB/s 算理论下限约 `0.094 ms`），因此省去 7 次 `__syncthreads()` 带来了约 `7 us`（`6%`）的净收益；在序列更短（如 `COLS=256/512`）或循环内多次规约时，Shuffle 的优势会更明显。

---

## 三、高级算子实验（Triton / `lightop` / `rocBLAS`）

### 3.1 实验 7 & 10：Memory-bound 算子融合对比（PyTorch vs `lightop` vs Triton RMSNorm）

- **文件：** [07_triton_fused_rmsnorm.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/07_triton_fused_rmsnorm.py)、[10_lightop_vs_triton_pytorch.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/10_lightop_vs_triton_pytorch.py)
- **测试负载：** `(8192, 4096)` FP16 输入的 RMSNorm（最低理论 HBM 读写量：读 X + 写 Y = `2 × 8192 × 4096 × 2B = 134.2 MB`）。
- **实测结果：**

| 实现方式 | 平均耗时 (ms) | 有效 HBM 带宽 (GB/s) | 相对 PyTorch 加速比 | 底层派发 Kernel / 关键特征 |
|---|---:|---:|---:|---|
| `PyTorch Unfused` | `1.9925` | `67.36` | `1.00x` | 拆分为 `aten::mul`、`aten::mean`、`rsqrt` 等多个独立 Kernel，反复读写 HBM |
| `lightop.fused_rms_norm_contiguous` | `0.1139` | `1178.54` | **`17.49x`** | `torch.profiler` 捕获底层为 `fused_rms_kernel_contiguous`（单次 `111.2 us`） |
| **`Triton Fused`** | **`0.1037`** | **`1293.82`** | **`19.21x`** | **达到实测 HBM 峰值（1364.93 GB/s）的 `94.8%`** |

- **Triton 编译元数据与 AMDGCN ISA 剖析（`07_rmsnorm_isa.s`）：**
  - 编译元数据：`arch='gfx936'`, `backend_name='hip'`, `num_warps=8`, `warp_size=64`, **`shared=32`**（仅用 32 字节 LDS 做跨 8 个 Wave 的汇总）。
  - ISA 指令统计（总计 436 行汇编）：`load_dword=3`、`s_waitcnt=18`、`s_barrier=2`、`permute/dpp=1`（通过寄存器级跨 Lane 通信完成 Wave 内规约，彻底消除了多轮 LDS 同步）。

---

### 3.2 实验 8：Compute-bound 矩阵映射（Triton GEMM `tl.dot` vs `rocBLAS`）

- **文件：** [08_triton_gemm_mmac.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/08_triton_gemm_mmac.py)
- **测试负载：** `4096 × 4096 × 4096` FP16 矩阵乘（总计算量 $2 \times 4096^3 = 137.44\text{ GFLOPs}$）。
- **实测结果：**

| 实现 / 配置 | 平均耗时 (ms) | 实测算力 (TFLOPS) | LDS 占用 (B) | VGPR / SGPR | 矩阵指令数 (`matrix_ops`) | `s_waitcnt` 次数 |
|---|---:|---:|---:|---:|---:|---:|
| **`torch.matmul` (rocBLAS)** | **`0.535`** | **`256.80`** | - | - | - | - |
| Triton (`BM=128, BN=128, BK=32, warps=4, stages=1`) | `1.416` | **`97.08`** | `8704` | `143 / 27` | `32` | `27` |
| Triton (`BM=128, BN=128, BK=32, warps=4, stages=2`) | `1.500` | `91.61` | `16384` | `160 / 36` | `64` | `32` |
| Triton (`BM=128, BN=128, BK=64, warps=8, stages=2`) | `2.465` | `55.76` | `32768` | `128 / 44` | `64` | `56` |
| Triton (`BM=128, BN=256, BK=32, warps=8, stages=2`) | `2.566` | `53.56` | `24576` | `128 / 44` | `64` | `65` |

- **深度微架构分析：**
  1. **`rocBLAS` 为什么能跑到 `256.80 TFLOPS`？** `torch.profiler` 捕获到 `torch.matmul` 底层调用的是 Tensile 针对 BW100 手工/自动调优的汇编 Kernel：`Cijk_Ailk_Bljk_HBH_MT256x256x16_SE_AMAS3_BW_DTL0_BL1...`，其 Macro Tile 为 **`256×256×16`**，精准匹配了 BW100 的矩阵指令流、LDS Bank 无冲突布局与寄存器分配。
  2. **为什么 Triton `stages=2` 反而比 `stages=1` 慢（`91.61` vs `97.08 TFLOPS`）？**
     - 当 `num_stages` 从 `1` 增至 `2` 时，Triton 为了在 LDS 做双缓冲流水线，将 **LDS 占用从 `8704 B` 翻倍到 `16384 B`**，同时 **VGPR 从 `143` 膨胀到 `160`**、**SGPR 从 `27` 升到 `36`**。
     - 高 VGPR（160）严重挤压了单 CU 上可并发驻留的 Wavefront 数量（Occupancy 下降），导致增加的流水线深度收益被并发度损失抵消。
  3. **为什么 `BK=64, warps=8` 跌至 `55.76 TFLOPS`？** LDS 暴涨到 `32768 B`（32 KB，直接把每 CU 驻留 Block 上限锁死在 2 个甚至 1 个），且 `s_waitcnt` 次数翻倍到 `56` 次，流水线停顿加剧。
  4. **如何进一步优化 Triton GEMM：** 从 `rocBLAS` 的 `MT256x256x16` 可以看出，BW100 上更适合尝试 **`BK=16`（减小单步 LDS 与 VGPR 压力）+ `kpack` / `waves_per_eu` 调优**，配合 `triton.autotune` 搜索更优的 Tile 组合。

---

## 四、总结：核心微架构规律速查表

| 微架构机制 | BW100 实测关键数字 | 算子优化第一准则 |
|---|---|---|
| **HBM 合并与向量化** | `float4` `1364.93 GB/s` vs `float` `530.29 GB/s` vs `stride=32` `34.83 GB/s` | Memory-bound 算子务必保证相邻 Lane 连续访问，优先用 128-bit 向量化加载 |
| **LDS Bank Conflict** | 32-way 冲突慢 **`19.62x`**；Padding/Swizzle 恢复 **`19.1x`** | 跨行/跨列访问 LDS Tile 时，优先用 XOR Swizzle（零额外容量）或 Padding |
| **SIMT 分支发散** | 4-way 向量分支慢 **`6.91x`**（相对 Baseline） | 尽量把条件判断做成 Wave-uniform，避免同一 Wave 内走多路复杂分支 |
| **Occupancy vs 寄存器/LDS** | LDS 0→48KB 使 Wave/CU `32→4`，Low-ILP 慢 **`2.71x`**；Triton `vgpr=160` 拖累 GEMM | 不要盲目堆大 `BLOCK_K` 或 `num_stages`，时刻盯紧 VGPR 和 LDS 是否触发 Occupancy 悬崖 |
| **算子融合（Roofline）** | 拐点 `~188 FLOPs/Byte`；RMSNorm 融合提速 **`19.2x`** | 对所有低计算强度算子，优先通过 Triton 或 `lightop` 融合消除中间 HBM 读写 |

---

## 附录：实验文件索引

- 基础 HIP 微基准（`experiments/`）：
  - [01_device_info_and_isa.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/01_device_info_and_isa.hip)
  - [02_coalescing_and_vec.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/02_coalescing_and_vec.hip)
  - [03_lds_bank_conflict.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/03_lds_bank_conflict.hip)
  - [04_branch_divergence.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/04_branch_divergence.hip)
  - [05_occupancy_and_ilp.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/05_occupancy_and_ilp.hip)
  - [06_shuffle_vs_lds.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/06_shuffle_vs_lds.hip)
- 高级算子与软件栈实验（`experiments/advanced/`）：
  - [07_triton_fused_rmsnorm.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/07_triton_fused_rmsnorm.py)
  - [08_triton_gemm_mmac.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/08_triton_gemm_mmac.py)
  - [09_hygon_ops_and_stack.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/09_hygon_ops_and_stack.py)
  - [10_lightop_vs_triton_pytorch.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/10_lightop_vs_triton_pytorch.py)
