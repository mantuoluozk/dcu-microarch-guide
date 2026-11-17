# 实验 1：全局内存合并访存与 `float4` 向量化

> **源码：** [02_coalescing_and_vec.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/02_coalescing_and_vec.hip)  
> **关联理论：** [第 3 章：存储体系与合并访存](../03-memory/memory-hierarchy.md)

---

## 1. 实验原理：为什么访问步长与位宽会决定生死？

当 Wavefront（64 个 Lane）执行一条 `global_load` 指令时，硬件内存控制器会把 64 个 Lane 请求的地址按 **Cache Line（常见 64B / 128B）** 合并成尽量少的内存事务：

- **Coalesced（合并访存）：** 相邻 Lane 访问相邻地址，64 个 `float`（256 字节）只需极少数连续 Cache Line 事务即可满足。
- **Strided（跨步访存）：** 当步长 $S$ 增大时，相邻 Lane 地址间隔变为 $4S$ 字节，64 个请求被迫分散到更多 Cache Line 中，大量搬运回来的相邻字节根本不被当前 Wave 使用，造成**有效带宽雪崩**。
- **Vectorized（`float4` 128-bit 访存）：** 每个 Lane 一次发射 128-bit（4 个 `float`），在搬运相同总字节数时，访存指令数精确降为 `1/4`，大幅缓解指令发射队列瓶颈。

---

## 2. 完整源码与逐段解读

```cpp title="experiments/02_coalescing_and_vec.hip"
#include "common.h"

// 1. 连续合并 32-bit 访存：每个线程读写 1 个 float
__global__ void copy_float_coalesced(const float* __restrict__ src,
                                     float* __restrict__ dst,
                                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// 2. 连续合并 128-bit 向量化访存：每个线程用 float4 一次读写 4 个 float
__global__ void copy_float4_coalesced(const float4* __restrict__ src,
                                      float4* __restrict__ dst,
                                      int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        dst[idx] = src[idx];
    }
}

// 3. 跨步非合并访存：通过位与取模保证所有线程依然合法访存且总读写量不变
__global__ void copy_float_strided(const float* __restrict__ src,
                                   float* __restrict__ dst,
                                   int n,
                                   int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int s_idx = static_cast<int>((static_cast<long long>(idx) * stride) & (n - 1));
        dst[s_idx] = src[s_idx];
    }
}

int main() {
    // 64M floats = 256 MB/缓冲区，远超 BW100 的 8 MB L2 Cache，确保穿透到 HBM
    constexpr int N = 1 << 26;
    constexpr size_t bytes = static_cast<size_t>(N) * sizeof(float);

    float *d_src = nullptr, *d_dst = nullptr;
    HIP_CHECK(hipMalloc(&d_src, bytes));
    HIP_CHECK(hipMalloc(&d_dst, bytes));
    HIP_CHECK(hipMemset(d_src, 1, bytes));

    constexpr int block_size = 256;
    const int grid_f1 = (N + block_size - 1) / block_size;
    const int n4 = N / 4;
    const int grid_f4 = (n4 + block_size - 1) / block_size;

    auto report = [&](const char* name, float ms) {
        double gbs = (2.0 * bytes / 1e9) / (ms / 1e3); // 读 + 写 = 2 * 256 MB
        std::printf("  %-26s : %8.4f ms | %8.2f GB/s\n", name, ms, gbs);
    };

    report("Coalesced float  (32-bit)", benchmark_kernel([&]() {
        hipLaunchKernelGGL(copy_float_coalesced, dim3(grid_f1), dim3(block_size), 0, 0, d_src, d_dst, N);
    }));
    report("Coalesced float4 (128-bit)", benchmark_kernel([&]() {
        hipLaunchKernelGGL(copy_float4_coalesced, dim3(grid_f4), dim3(block_size), 0, 0,
                           reinterpret_cast<const float4*>(d_src),
                           reinterpret_cast<float4*>(d_dst), n4);
    }));

    const int strides[] = {2, 4, 8, 16, 32};
    for (int s : strides) {
        char label[64];
        std::snprintf(label, sizeof(label), "Strided float (stride=%d)", s);
        report(label, benchmark_kernel([&]() {
            hipLaunchKernelGGL(copy_float_strided, dim3(grid_f1), dim3(block_size), 0, 0, d_src, d_dst, N, s);
        }));
    }

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));
    return 0;
}
```

### 关键代码设计细节
- **为什么分配 `256 MB`（$2^{26}$ 个 `float`）？** BW100 的 L2 Cache 为 `8 MB`。只有工作集远大于 `8 MB` 时，预热后的测试才不会命中 L2，测出的才是真实的 **HBM 带宽**。
- **`__restrict__` 修饰符的作用：** 告诉编译器 `src` 和 `dst` 指针不重叠，允许编译器大胆重排访存指令并生成只读/向量化加载路径。
- **`(idx * stride) & (n - 1)` 技巧：** 由于 `n` 是 2 的幂，用按位与取代 `% n` 避免引入昂贵的整数除法指令，同时保证总访存指令数与 `copy_float_coalesced` 完全一致（都是 `N` 次读 + `N` 次写），做到**单变量控制**。

---

## 3. 复现命令

```bash
cd /exp/dcu_microarch_exp
make 02_coalescing_and_vec
./02_coalescing_and_vec

# 抓取硬件 PMC 计数器
hipprof --pmc --pmc-type 3 -o prof_out/exp02 ./02_coalescing_and_vec
python3 parse_hipprof_pmc.py prof_out/exp02.csv
```

---

## 4. BW100 原始输出日志与 PMC 证据

### 原始终端输出

```text
$ ./02_coalescing_and_vec
=== Exp 2: Global Memory Coalescing & Vectorization (256MB copy) ===
  Coalesced float  (32-bit)  :   1.0124 ms |   530.29 GB/s
  Coalesced float4 (128-bit) :   0.3933 ms |  1364.93 GB/s
  Strided float (stride=2)   :   1.3159 ms |   408.00 GB/s
  Strided float (stride=4)   :   2.5414 ms |   211.25 GB/s
  Strided float (stride=8)   :   5.2079 ms |   103.09 GB/s
  Strided float (stride=16)  :  10.3145 ms |    52.05 GB/s
  Strided float (stride=32)  :  15.4136 ms |    34.83 GB/s
```

```text
$ python3 parse_hipprof_pmc.py prof_out/exp02.csv
=== PMC Summary: prof_out/exp02.csv (3 unique kernels) ===

Kernel: _Z20copy_float_coalescedPKfPfi (calls=25)
  Grid=67108864 | Block=256 | LDS=0B | Scratch=0B | VGPR=4 (Accum=0) | SGPR=16
  Avg Kernel Time      :    1011.68 us | GRBM_GUI_ACTIVE = 1,330,892
  Insts (VALU/LDS/Rd/Wr): 10,485,760 / 0 / 1,048,576 / 1,048,576
  SQ_LDS_BANK_CONFLICT : 0 | SQ_WAIT_INST_LDS = 0
  L2 (TCC) Hit / Miss  : 195 / 6,291,461 (Hit Rate = 0.00%)
  TCP_TA_DATA_STALL    : 7,070,939

Kernel: _Z21copy_float4_coalescedPK15HIP_vector_typeIfLj4EEPS0_i (calls=25)
  Grid=16777216 | Block=256 | LDS=0B | Scratch=0B | VGPR=8 (Accum=0) | SGPR=16
  Avg Kernel Time      :     391.04 us | GRBM_GUI_ACTIVE = 524,295
  Insts (VALU/LDS/Rd/Wr): 2,621,440 / 0 / 262,144 / 262,144
  SQ_LDS_BANK_CONFLICT : 0 | SQ_WAIT_INST_LDS = 0
  L2 (TCC) Hit / Miss  : 195 / 6,291,461 (Hit Rate = 0.00%)
  TCP_TA_DATA_STALL    : 10,351,276

Kernel: _Z18copy_float_stridedPKfPfii (calls=125)
  Grid=67108864 | Block=256 | LDS=0B | Scratch=0B | VGPR=4 (Accum=0) | SGPR=16
  Avg Kernel Time      :   15422.14 us | GRBM_GUI_ACTIVE = 20,072,775
  Insts (VALU/LDS/Rd/Wr): 12,582,912 / 0 / 1,048,576 / 1,048,576
  SQ_LDS_BANK_CONFLICT : 0 | SQ_WAIT_INST_LDS = 0
  L2 (TCC) Hit / Miss  : 234 / 134,217,736 (Hit Rate = 0.00%)
  TCP_TA_DATA_STALL    : 519,544,084
```

### 结构化汇总表

| 访问模式 | 平均耗时 (ms) | 有效带宽 (GB/s) | `SQ_INSTS_VMEM_RD` | `SQ_INSTS_VMEM_WR` | `TCC_MISS` (L2 Miss) | `TCP_TA_DATA_STALL` |
|---|---:|---:|---:|---:|---:|---:|
| `Coalesced float (32-bit)` | `1.0124` | `530.29` | `1,048,576` | `1,048,576` | `6,291,461` | `7,070,939` |
| **`Coalesced float4 (128-bit)`** | **`0.3933`** | **`1364.93`** | **`262,144`（精确 1/4）** | **`262,144`（精确 1/4）** | `6,291,461` | `10,351,276` |
| `Strided float (stride=2)` | `1.3159` | `408.00` | - | - | - | - |
| `Strided float (stride=4)` | `2.5414` | `211.25` | - | - | - | - |
| `Strided float (stride=8)` | `5.2079` | `103.09` | - | - | - | - |
| `Strided float (stride=16)` | `10.3145` | `52.05` | - | - | - | - |
| `Strided float (stride=32)` | `15.4136` | `34.83` | `1,048,576` | `1,048,576` | **`134,217,736`（21.3x）** | **`519,544,084`（73x）** |

---

## 5. 现象拆解与工程结论

1. **`float4` 提速 `2.57x` 的硬件真相：** 看 `TCC_MISS` 列，`float4` 和 `float` 都是 `6,291,461` 次（搬运了完全相同的 512 MB HBM 数据）；但 `float4` 把 `VMEM_RD` 和 `VMEM_WR` 指令数精确压到了 **`1/4`**，直接释放了指令发射瓶颈，测出了 BW100 的 **HBM 实测峰值带宽 `1364.93 GB/s`**。
2. **`stride=32` 暴跌至 `2.6%` 的硬件真相：** 同样发射 `1,048,576` 条访存指令，非合并访问让 L2 Miss 暴涨 **21.3 倍**（1.34 亿次），L1/TA 数据停顿周期暴涨 **73 倍**（5.19 亿周期）。
3. **实战准则：** 在编写任何 Memory-bound 算子时，必须确保 `threadIdx.x` 对应张量的最内层连续维度（Stride=1），并优先使用 `float4` / `int4`（128-bit）向量化读写。
