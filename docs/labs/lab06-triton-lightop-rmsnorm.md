# 实验 6：Memory-bound 算子融合实战（PyTorch vs `lightop` vs Triton RMSNorm）

> **源码：** [07_triton_fused_rmsnorm.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/07_triton_fused_rmsnorm.py) & [10_lightop_vs_triton_pytorch.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/10_lightop_vs_triton_pytorch.py)  
> **关联理论：** [第 4 章：Roofline 模型](../04-scheduling/occupancy-and-roofline.md)

---

## 1. 实验原理：为什么 RMSNorm 的核心是“减少 HBM 往返”？

由实验 1 与实验 7 算出的 BW100 FP16 Roofline 拐点约为 `188.1 FLOPs/Byte`。而 RMSNorm 每个元素只做约 4~5 次浮点运算却要读写 4 字节（FP16 读+写），计算强度仅约 **1.25 FLOPs/Byte**，处于极度 **Memory-bound** 区间：
- **PyTorch 拆解实现：** `mul` → `mean` → `add` → `rsqrt` → `mul` → `mul`，每一步都要把中间结果写回 HBM，再由下一个 Kernel 重新读入。
- **Triton / `lightop` 融合实现：** 只从 HBM 读 1 次输入 `X` 进 VGPR，在片上寄存器与 Wavefront 内完成平方和规约，再一次性写回输出 `Y`（总 HBM 理论流量仅 `2 × 8192 × 4096 × 2B = 134.2 MB`）。

---

## 2. 完整源码与逐段解读

```python title="experiments/advanced/07_triton_fused_rmsnorm.py"
import time
import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x_row, stride_y_row,
    N: tl.constexpr, eps: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)                  # 每个 Program (Block) 负责矩阵的一行
    cols = tl.arange(0, BLOCK_N)            # 0..4095（覆盖整行特征维度）
    mask = cols < N

    x_ptrs = X_ptr + row * stride_x_row + cols
    w_ptrs = W_ptr + cols
    y_ptrs = Y_ptr + row * stride_y_row + cols

    # 1. 从 HBM 一次性加载整行到 VGPR 并转为 FP32 累加精度
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    # 2. 全部在寄存器 / Wavefront 内完成规约（不回写 HBM）
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    y = x * rstd * w

    # 3. 仅写回一次结果到 HBM
    tl.store(y_ptrs, y.to(Y_ptr.dtype.element_ty), mask=mask)
```

对比 `lightop` 的调用与 `torch.profiler` 抓包（[10_lightop_vs_triton_pytorch.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/10_lightop_vs_triton_pytorch.py)）：

```python title="experiments/advanced/10_lightop_vs_triton_pytorch.py (核心对比片段)"
y_lop = lightop.fused_rms_norm_contiguous(out.clone(), x, w, eps)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
) as prof:
    lightop.fused_rms_norm_contiguous(out, x, w, eps)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
```

---

## 3. BW100 实测性能对比（`8192 × 4096` FP16）

| 实现方式 | 平均耗时 (ms) | 有效 HBM 带宽 (GB/s) | 加速比 | 底层派发 Kernel / 特征 |
|---|---:|---:|---:|---|
| `PyTorch Unfused` | `1.9925` | `67.36` | `1.00x` | 多个独立 Kernel 反复读写 HBM |
| `lightop.fused_rms_norm_contiguous` | `0.1139` | `1178.54` | **`17.49x`** | `torch.profiler` 捕获 `fused_rms_kernel_contiguous`（`111.2 us`） |
| **`Triton Fused`** | **`0.1037`** | **`1293.82`** | **`19.21x`** | **达到实验 1 实测 HBM 极限（`1364.93 GB/s`）的 `94.8%`** |

---

## 4. Triton 编译元数据与 AMDGCN ISA 剖析

`07_triton_fused_rmsnorm.py` 自动导出了 `KernelMetadata` 与汇编文件 `07_rmsnorm_isa.s`：

```text
KernelMetadata(..., arch='gfx936', backend_name='hip', num_warps=8, warp_size=64, shared=32, ...)
[ISA Summary (amdgcn)] total_lines=436, load_dword=3, s_waitcnt=18, s_barrier=2, permute/dpp=1
```

**怎么读这份 ISA 摘要？**
1. **`shared=32`：** 整个 Block 有 `num_warps=8`（512 线程），却只用了 **32 字节 LDS**（正好 8 个 `float`），因为每个 Wave64 内部全部通过寄存器级跨 Lane 指令（`permute/dpp`）规约，只把 8 个部分和放进 LDS 汇总。
2. **仅 `2` 次 `s_barrier`：** 相比传统纯 LDS 树状规约省去了大量同步停顿，使有效带宽直接贴近物理 HBM 天花板。
