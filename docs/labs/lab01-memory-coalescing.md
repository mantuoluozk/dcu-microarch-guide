# 实验 1：全局内存合并访存与 `float4` 向量化

> **源码：** [02_coalescing_and_vec.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/02_coalescing_and_vec.hip)  
> **关联章节：** [第 3 章：存储体系与合并访存](../03-memory/memory-hierarchy.md)

---

## 1. 实验设计

分配两块各 **256 MB**（$2^{26}$ 个 `float`，远超 BW100 的 8 MB L2 Cache）的显存缓冲区做拷贝（总 HBM 读写量 = 512 MB），对比：
1. 连续合并 `float`（32-bit/lane）
2. 连续合并 `float4`（128-bit/lane）
3. 跨步非合并 `float`（`stride = 2, 4, 8, 16, 32`）

```bash
./02_coalescing_and_vec
hipprof --pmc --pmc-type 3 -o prof_out/exp02 ./02_coalescing_and_vec
python3 parse_hipprof_pmc.py prof_out/exp02.csv
```

---

## 2. BW100 实测性能与 PMC 硬件计数器

| 访问模式 | 平均耗时 (ms) | 有效带宽 (GB/s) | `VMEM_RD` 指令 | `VMEM_WR` 指令 | `TCC_MISS` (L2 Miss) | `TCP_TA_DATA_STALL` |
|---|---:|---:|---:|---:|---:|---:|
| `Coalesced float (32-bit)` | `1.0124` | `530.29` | `1,048,576` | `1,048,576` | `6,291,461` | `7,070,939` |
| **`Coalesced float4 (128-bit)`** | **`0.3933`** | **`1364.93`** | **`262,144`** | **`262,144`** | `6,291,461` | `10,351,276` |
| `Strided float (stride=2)` | `1.3159` | `408.00` | - | - | - | - |
| `Strided float (stride=4)` | `2.5414` | `211.25` | - | - | - | - |
| `Strided float (stride=8)` | `5.2079` | `103.09` | - | - | - | - |
| `Strided float (stride=16)` | `10.3145` | `52.05` | - | - | - | - |
| `Strided float (stride=32)` | `15.4136` | `34.83` | `1,048,576` | `1,048,576` | **`134,217,736`** | **`519,544,084`** |

---

## 3. 微架构根因分析

1. **`float4` 提速 `2.57x` 的物理原因：**
   `float4` 与 `float` 的 L2 Miss 次数完全相同（`6,291,461`），说明外部搬运的总数据量完全一致；但 `float4` 的 `VMEM_RD/WR` 指令数精确降为 **`1/4`**，大幅降低了指令发射瓶颈，实测带宽达到 **`1364.93 GB/s`**（BW100 HBM 实测上限）。
2. **`stride=32` 暴跌至 `2.6%` 的物理原因：**
   `stride=32` 发射了与合并 `float` 完全相同的 `1,048,576` 条访存指令，但因为跨步打散了 Cache Line 合并，`TCC_MISS` 暴涨 **21.3 倍**（1.34 亿次），`TCP_TA_DATA_STALL` 暴涨 **73 倍**（5.19 亿周期）。
