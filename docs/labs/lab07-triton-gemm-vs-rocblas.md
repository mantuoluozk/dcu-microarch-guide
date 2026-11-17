# 实验 7：Triton GEMM（`tl.dot`）vs `rocBLAS` 与 `hipprof` PMC 根因破案

> **源码：** [08_triton_gemm_mmac.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/advanced/08_triton_gemm_mmac.py)  
> **关联理论：** [第 2 章：MMAC 矩阵引擎](../02-compute/cu-and-simt.md) & [第 3 章：`s_waitcnt` 异步流水](../03-memory/memory-hierarchy.md)

---

## 1. 实验原理：从 `tl.dot` 到硬件矩阵指令与 L2 Swizzle

在大矩阵乘 `C = A × B`（`4096 × 4096 × 4096` FP16，总计算量 `137.44 GFLOPs`）中，有两个关键优化直接决定算力上限：

1. **L2 Cache 友好的 Group Ordering（`GROUP_M = 8`）：** 如果按普通行主序派发 Block，同一批执行的 Block 会访问同一行 `A` 却把 `B` 扫穿整个 8 MB L2；按 `GROUP_M = 8` 分组后，相邻 Block 在 $M$ 和 $N$ 方向同时复用 `A` 和 `B` 的 Tile，大幅提升 L2 命中率。
2. **Software Pipelining（`num_stages`）与资源权衡：** `num_stages >= 2` 会在 LDS 中开辟多级缓冲做计算/访存重叠，但也会成倍增加 LDS 占用和 VGPR 压力，可能反过来压低 Occupancy 甚至触发寄存器 Spill。

---

## 2. 完整源码与逐段解读

```python title="experiments/advanced/08_triton_gemm_mmac.py (核心 Kernel 与 ISA 检查)"
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    # 1. L2 Swizzle 分组映射：提升相邻 Block 的 A/B Tile 复用率
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # 2. K 维累加主循环：tl.dot 会被后端降级为 DCU 矩阵乘加指令 (v_mfma / v_mmac)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(C_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
```

---

## 3. BW100 实测算力与静态资源对比

```bash
cd /exp/dcu_microarch_exp
python3 advanced/08_triton_gemm_mmac.py
hipprof --pmc --pmc-type 3 -o prof_out/exp08 python3 advanced/08_triton_gemm_mmac.py
python3 parse_hipprof_pmc.py prof_out/exp08.csv
```

### 原始终端输出

```text
$ python3 advanced/08_triton_gemm_mmac.py
=== Exp 8: Triton GEMM vs torch.matmul (4096x4096x4096, FP16) ===
  torch.matmul (rocBLAS/hipBLAS) :   0.535 ms | 256.80 TFLOPS

  BM=128, BN=128, BK=32, warps=4, stages=1 |   1.416 ms |  97.08 TFLOPS | diff=0.000 | LDS=8704B | vgpr=143, sgpr=27, matrix_ops=32, s_waitcnt=27
  BM=128, BN=128, BK=32, warps=4, stages=2 |   1.500 ms |  91.61 TFLOPS | diff=0.000 | LDS=16384B | vgpr=160, sgpr=36, matrix_ops=64, s_waitcnt=32
  BM=128, BN=128, BK=64, warps=8, stages=2 |   2.465 ms |  55.76 TFLOPS | diff=0.000 | LDS=32768B | vgpr=128, sgpr=44, matrix_ops=64, s_waitcnt=56
  BM=128, BN=256, BK=32, warps=8, stages=2 |   2.566 ms |  53.56 TFLOPS | diff=0.000 | LDS=24576B | vgpr=128, sgpr=44, matrix_ops=64, s_waitcnt=65
```

```text
$ python3 parse_hipprof_pmc.py prof_out/exp08.csv   # 节选 rocBLAS 与 Triton 关键对比
Kernel: Cijk_Ailk_Bljk_HBH_MT256x256x16_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KM... (calls=41)
  Grid=131072 | Block=512 | LDS=65536B | Scratch=0B | VGPR=212 (Accum=0) | SGPR=112
  Avg Kernel Time      :     530.78 us | GRBM_GUI_ACTIVE = 653,439
  Insts (VALU/LDS/Rd/Wr): 17,670,144 / 4,194,304 / 1,048,576 / 262,144
  SQ_LDS_BANK_CONFLICT : 8,388,608 | SQ_WAIT_INST_LDS = 3,934,304
  L2 (TCC) Hit / Miss  : 11,161,361 / 2,479,638 (Hit Rate = 81.82%)

Kernel: gemm_kernel (calls=164)
  Grid=262144 | Block=512 | LDS=24576B | Scratch=96B | VGPR=128 (Accum=0) | SGPR=48
  Avg Kernel Time      :    2556.83 us | GRBM_GUI_ACTIVE = 3,341,603
  Insts (VALU/LDS/Rd/Wr): 42,508,288 / 21,266,432 / 8,429,568 / 126,976
  SQ_LDS_BANK_CONFLICT : 34,078,720 | SQ_WAIT_INST_LDS = 99,050,034
  L2 (TCC) Hit / Miss  : 28,551,829 / 2,765,341 (Hit Rate = 91.17%)
```

### 结构化汇总表

| 实现 / 配置 | 耗时 (ms) | 算力 (TFLOPS) | LDS (B) | VGPR / SGPR | `matrix_ops` | `s_waitcnt` |
|---|---:|---:|---:|---:|---:|---:|
| **`torch.matmul` (rocBLAS)** | **`0.535`** | **`256.80`** | - | - | - | - |
| Triton (`BM=128, BN=128, BK=32, w=4, s=1`) | `1.416` | **`97.08`** | `8704` | `143 / 27` | `32` | `27` |
| Triton (`BM=128, BN=128, BK=32, w=4, s=2`) | `1.500` | `91.61` | `16384` | `160 / 36` | `64` | `32` |
| Triton (`BM=128, BN=128, BK=64, w=8, s=2`) | `2.465` | `55.76` | `32768` | `128 / 44` | `64` | `56` |
| Triton (`BM=128, BN=256, BK=32, w=8, s=2`) | `2.566` | `53.56` | `24576` | `128 / 44` | `64` | `65` |

---

## 4. `hipprof` PMC 硬件计数器破案（为什么 Triton 慢 4.8x？）

通过 `parse_hipprof_pmc.py prof_out/exp08.csv` 对比 `rocBLAS`（`Cijk_Ailk_Bljk_HBH_MT256x256x16...`）与 Triton `gemm_kernel`（`BM=128, BN=256, BK=32`）：

| 关键硬件 PMC 指标 | `rocBLAS` (`MT256x256x16`) | Triton `gemm_kernel` | 诊断结论 |
|---|---:|---:|---|
| 平均耗时 | **`530.78 us`** | `2556.83 us` | 慢 `4.82x` |
| VGPR / SGPR | `212 / 112` | `128 / 48` | - |
| **寄存器溢出 (`Scratch`)** | **`0 B`** | **`96 B`（发生 Spill！）** | **Triton 产生了 96B/thread 显存栈溢出** |
| 全局读指令 (`SQ_INSTS_VMEM_RD`) | **`1,048,576`** | `8,429,568` | Triton 多发射了 **8x** 全局加载指令 |
| LDS 指令 (`SQ_INSTS_LDS`) | **`4,194,304`** | `21,266,432` | Triton 多做了 **5x** LDS 读写 |
| **`SQ_LDS_BANK_CONFLICT`** | `8,388,608` | **`34,078,720`** | Triton LDS 冲突是 `rocBLAS` 的 **4.06x** |
| **`SQ_WAIT_INST_LDS`** | **`3,934,304`** | **`99,050,034`** | **Triton 有 9900 万周期卡在等 LDS（25.2x）！** |
| L2 命中率 (`TCC_HIT %`) | `81.82%` | **`91.17%`** | `GROUP_M=8` 的 L2 Swizzle 非常有效 |

**破案总结：**
1. **外部访存并不是瓶颈：** Triton 的 L2 命中率（`91.17%`）甚至高于 `rocBLAS`。
2. **三大真实元凶：** **`96 B` 寄存器 Spill**、**8 倍 `VMEM_RD` 指令**，以及 **高达 9900 万周期的 `SQ_WAIT_INST_LDS`（5 倍 LDS 指令 + 4 倍 Bank Conflict）**。
3. **下一步调优方向（见路线图实验 8）：** 参考 `rocBLAS` 的 `MT256x256x16`，在 BW100 上改用 **`BK=16`（降低 LDS 与寄存器压力）+ `kpack` / `waves_per_eu` 调优**，并用 `hipprof --pmc` 验证 `Scratch=0B` 与 `SQ_WAIT_INST_LDS` 的消除效果。
