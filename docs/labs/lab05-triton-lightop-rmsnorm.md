# 实验 5：Memory-bound 算子融合实战（PyTorch vs `lightop` vs Triton RMSNorm）

> **源码：** [07_triton_fused_rmsnorm.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/07_triton_fused_rmsnorm.py) & [10_lightop_vs_triton_pytorch.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/10_lightop_vs_triton_pytorch.py)  
> **关联章节：** [第 4 章：Roofline 模型](../04-scheduling/occupancy-and-roofline.md)

---

## 1. 实验背景

RMSNorm 的计算强度极低（远低于 BW100 的 Roofline 拐点 `~188.1 FLOPs/Byte`），属于典型的 **Memory-bound** 算子：
- **PyTorch 拆解实现：** `mul` → `mean` → `add` → `rsqrt` → `mul` → `mul`，每个小算子都必须把中间张量写回 HBM 再由下一个 Kernel 读出。
- **算子融合（Triton / `lightop`）：** 仅从 HBM 读一次输入 $X$、在 VGPR 与 Wavefront 内完成规约、再一次性写回输出 $Y$（`2 × 8192 × 4096 × 2B = 134.2 MB`）。

---

## 2. BW100 实测性能对比（`8192 × 4096` FP16）

| 实现方式 | 平均耗时 (ms) | 有效 HBM 带宽 (GB/s) | 加速比 | `torch.profiler` / 编译特征 |
|---|---:|---:|---:|---|
| `PyTorch Unfused` | `1.9925` | `67.36` | `1.00x` | 多个独立 Kernel 反复读写 HBM |
| `lightop.fused_rms_norm_contiguous` | `0.1139` | `1178.54` | **`17.49x`** | 底层派发 `fused_rms_kernel_contiguous`（`111.2 us`） |
| **`Triton Fused`** | **`0.1037`** | **`1293.82`** | **`19.21x`** | **达到实验 1 实测 HBM 极限（`1364.93 GB/s`）的 `94.8%`** |

---

## 3. Triton 编译产物与 AMDGCN ISA 剖析

`07_triton_fused_rmsnorm.py` 自动打印的 `KernelMetadata` 与 `07_rmsnorm_isa.s` 汇编统计如下：

- **Target：** `backend_name='hip'`, `arch='gfx936'`, `warp_size=64`, `num_warps=8`
- **LDS 占用：** `shared=32`（仅用 **32 字节** LDS 做跨 8 个 Wavefront 的部分和汇总）
- **ISA 统计（436 行）：** `load_dword=3`、`s_waitcnt=18`、`s_barrier=2`、`permute/dpp=1`（Wave 内部完全通过寄存器级跨 Lane 指令完成规约）
