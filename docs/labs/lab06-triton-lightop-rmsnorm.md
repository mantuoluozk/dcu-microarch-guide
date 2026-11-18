# E08 · Memory-bound 算子融合实战（PyTorch vs `lightop` vs Triton RMSNorm）

**研究问题**　融合能让当前形状的 RMSNorm 提速多少？

历史 BW100 记录中，8192×4096 的 FP16 输入上，Triton 融合比所测 PyTorch 非融合表达式快约 19.21 倍。

[适用环境](../baselines/bw100-gfx936.md) · [复现命令](#复现命令)

## BW100 结果

| 实现方式 | 平均耗时 (ms) | 有效带宽 (GB/s) | 加速比 | 底层派发 Kernel / 特征 |
|---|---:|---:|---:|---|
| `PyTorch Unfused` | `1.9925` | `67.36` | `1.00x` | 多个独立 Kernel 反复读写 HBM |
| `lightop.fused_rms_norm_contiguous` | `0.1139` | `1178.54` | **`17.49x`** | `torch.profiler` 捕获 `fused_rms_kernel_contiguous`（`111.2 us`） |
| **`Triton Fused`** | **`0.1037`** | **`1293.82`** | **`19.21x`** | 按张量最小读写量折算 |

## 复现命令

```bash
cd /exp/dcu_microarch_exp/advanced
python3 07_triton_fused_rmsnorm.py
python3 10_lightop_vs_triton_pytorch.py
```

> **源码：** [07_triton_fused_rmsnorm.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/07_triton_fused_rmsnorm.py) & [10_lightop_vs_triton_pytorch.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/10_lightop_vs_triton_pytorch.py)  
> **关联理论：** [第 4 章：Roofline 模型](../04-scheduling/occupancy-and-roofline.md)

## 实验原理：为什么 RMSNorm 的核心是“减少 HBM 往返”？

按最少 FP16 读写估算，每元素约 4–5 FLOPs / 4 B。这个口径下计算强度约为 1.25 FLOPs/B，可从访存和融合着手验证。

- **PyTorch 拆解实现：** `mul` → `mean` → `add` → `rsqrt` → `mul` → `mul`，每一步都要把中间结果写回 HBM，再由下一个 Kernel 重新读入。
- **Triton / `lightop` 融合实现：** 只从 HBM 读 1 次输入 `X` 进 VGPR，在片上寄存器与 Wavefront 内完成平方和规约，再一次性写回输出 `Y`（总 HBM 理论流量仅 `2 × 8192 × 4096 × 2B = 134.2 MB`）。

## Triton 编译元数据与 AMDGCN ISA 剖析

`07_triton_fused_rmsnorm.py` 自动导出了 `KernelMetadata` 与汇编文件 `07_rmsnorm_isa.s`：

```text
KernelMetadata(..., arch='gfx936', backend_name='hip', num_warps=8, warp_size=64, shared=32, ...)
[ISA Summary (amdgcn)] total_lines=436, load_dword=3, s_waitcnt=18, s_barrier=2, permute/dpp=1
```

**怎么读这份 ISA 摘要？**
1. **`shared=32`：** 整个 Block 有 `num_warps=8`（512 线程），却只用了 **32 字节 LDS**（正好 8 个 `float`），因为每个 Wave64 内部全部通过寄存器级跨 Lane 指令（`permute/dpp`）规约，只把 8 个部分和放进 LDS 汇总。
2. **仅 `2` 次 `s_barrier`：** 相比传统纯 LDS 树状规约省去了大量同步停顿，减少了这份实现里的同步开销。

::: details 查看源码与实现细节

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

:::

::: details 查看原始终端输出

```text
$ python3 07_triton_fused_rmsnorm.py
=== Exp 7: Triton Fused RMSNorm (8192x4096, torch.float16) ===
  Max diff vs PyTorch   : 0.003906
  PyTorch Unfused       :   1.9920 ms | Effective BW =   67.38 GB/s
  Triton Fused          :   0.1038 ms | Effective BW = 1293.03 GB/s
  Speedup               : 19.19x

[Triton Metadata] KernelMetadata(..., arch='gfx936', backend_name='hip', ..., num_warps=8, ..., shared=32, ..., warp_size=64)

[ISA Summary (amdgcn)] total_lines=436, load_dword=3, s_waitcnt=18, s_barrier=2, permute/dpp=1
  Full ISA saved to 07_rmsnorm_isa.s
```

```text
$ python3 10_lightop_vs_triton_pytorch.py
=== Exp 10: PyTorch vs Triton vs lightop RMSNorm (8192x4096, torch.float16) ===
  Max diff (Triton  vs PyTorch): 0.001953
  Max diff (lightop vs PyTorch): 0.001953
  PyTorch Unfused         :   1.9925 ms | Effective BW =   67.36 GB/s
  Triton Fused            :   0.1037 ms | Effective BW = 1293.82 GB/s
  lightop fused_rms_norm  :   0.1139 ms | Effective BW = 1178.54 GB/s
```

:::
