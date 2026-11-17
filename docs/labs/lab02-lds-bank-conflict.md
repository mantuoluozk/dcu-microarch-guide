# 实验 2：LDS Bank Conflict 与 Padding / XOR Swizzle 消除

> **源码：** [03_lds_bank_conflict.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/03_lds_bank_conflict.hip)  
> **关联理论：** [第 3 章：LDS 与 Bank Conflict](../03-memory/memory-hierarchy.md)

---

## 1. 实验原理：32-Bank 映射公式与冲突产生机制

DCU 片上的 LDS（64 KB/Block）通常划分为 **32 个 Bank，每个 Bank 宽度为 4 字节（32-bit）**：


\text{Bank ID} = \left\lfloor \frac{\text{Byte Address}}{4} \right\rfloor \bmod 32


对于 `__shared__ float tile[32][32]`：

- **行访问 `tile[r][tx]`：** 当 `tx = 0..31` 时，字节偏移依次增加 4 字节，恰好落在 Bank `0..31`，**0 冲突**。
- **列访问 `tile[tx][r]`：** 相邻 `tx` 之间的元素间隔为 `32` 个 `float`（128 字节），即 $(x + 32) \bmod 32 = x \bmod 32$，导致 32 个线程全部撞在同一个 Bank `r` 上，触发 **32-way Bank Conflict**。
- **Padding 消除（`[32][33]`）：** 把行宽改为 33，列步长变为 33，$\gcd(33, 32) = 1$，32 个线程错开落入全部 32 个 Bank（代价是多占 ~3% LDS）。
- **XOR Swizzle 消除（`tile[tx][r ^ tx]`）：** 对同一行内任意固定 `r`，当 `tx` 遍历 `0..31` 时，`r ^ tx` 同样是 `0..31` 的双射排列！**不浪费 1 字节 LDS 容量，完美清零冲突**。

---

## 2. 完整源码与逐段解读

```cpp title="experiments/03_lds_bank_conflict.hip"
#include "common.h"

constexpr int TILE = 32;
constexpr int LDS_ITERS = 2048;

template <int MODE>
__global__ void lds_bank_conflict_kernel(float* __restrict__ out) {
    // MODE 0: 行访问（无冲突）
    // MODE 1: 列访问（32-way Bank Conflict）
    // MODE 2: 列访问 + Padding [32][33]（无冲突，多占 128B LDS）
    // MODE 3: 列访问 + XOR Swizzle [tx][r ^ tx]（无冲突，零额外 LDS 开销）
    __shared__ float tile_std[TILE][TILE];
    __shared__ float tile_pad[TILE][TILE + 1];

    int tx = threadIdx.x % TILE; // 0..31
    int ty = threadIdx.x / TILE; // 0..31 (Block=1024)

    if constexpr (MODE == 2) {
        tile_pad[ty][tx] = static_cast<float>(threadIdx.x) * 0.001f;
    } else {
        tile_std[ty][tx] = static_cast<float>(threadIdx.x) * 0.001f;
    }
    __syncthreads();

    float acc = 0.0f;
    // 通过数据依赖链与写回防止编译器把 LDS 循环优化掉
    for (int i = 0; i < LDS_ITERS; ++i) {
        int r = (ty + i) & (TILE - 1);
        float v = 0.0f;
        if constexpr (MODE == 0) {
            v = tile_std[r][tx];
            tile_std[r][tx] = v + 1e-6f;
        } else if constexpr (MODE == 1) {
            v = tile_std[tx][r];
            tile_std[tx][r] = v + 1e-6f;
        } else if constexpr (MODE == 2) {
            v = tile_pad[tx][r];
            tile_pad[tx][r] = v + 1e-6f;
        } else if constexpr (MODE == 3) {
            v = tile_std[tx][r ^ tx];
            tile_std[tx][r ^ tx] = v + 1e-6f;
        }
        acc += v;
    }

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    out[gid] = acc;
}
```

### 关键代码设计细节
- **为什么在 Kernel 内循环 `2048` 次 LDS 读-改-写？** 单次 LDS 访问只有几十个周期，如果只访问一次，耗时全被 Kernel Launch 开销掩盖。循环 2048 次且不访问 HBM，可以把**纯粹的 LDS 冲突代价**放大到毫秒级。
- **为什么需要 `tile_std[...] = v + 1e-6f`？** 如果只读不写，编译器可能把循环不变量外提；读-改-写形成真依赖链，迫使硬件每一轮都真实发射 `ds_read_b32` 和 `ds_write_b32`。
- **为什么这里触发了编译器警告？** 本 Kernel 以 `block_size = 1024` 启动但未写 `__launch_bounds__(1024)`，运行时提示 `Launch params (1024, 1, 1) are larger than launch bounds (256)`，实际工程中应显式标注 `__launch_bounds__`。

---

## 3. 复现命令与 BW100 实测数据

```bash
cd /exp/dcu_microarch_exp
make 03_lds_bank_conflict && ./03_lds_bank_conflict
hipprof --pmc --pmc-type 3 -o prof_out/exp03 ./03_lds_bank_conflict
python3 parse_hipprof_pmc.py prof_out/exp03.csv
```

### 原始终端输出

```text
$ ./03_lds_bank_conflict
Launch params (1024, 1, 1) are larger than launch bounds (256) for kernel _Z24lds_bank_conflict_kernelILi0EEvPf please add __launch_bounds__ to kernel define or use --gpu-max-threads-per-block recompile program ! 
=== Exp 3: LDS Bank Conflict (32x32 Tile, 2048 LDS r/w iters) ===
  0. Row-wise (No conflict)              :   0.6578 ms
  1. Col-wise (32-way Bank Conflict)     :  12.9097 ms
  2. Col-wise + Padding [32][33]         :   0.6769 ms
  3. Col-wise + XOR Swizzle [tx][r^tx]   :   0.6751 ms
  Slowdown (32-way vs No-conflict)      : 19.62x
  Speedup  (Padding vs Conflict)        : 19.07x
  Speedup  (Swizzle vs Conflict)        : 19.12x
```

```text
$ python3 parse_hipprof_pmc.py prof_out/exp03.csv
=== PMC Summary: prof_out/exp03.csv (4 unique kernels) ===

Kernel: _Z24lds_bank_conflict_kernelILi0EEvPf (calls=25)
  Grid=262144 | Block=1024 | LDS=4096B | Scratch=0B | VGPR=24 (Accum=0) | SGPR=16
  Avg Kernel Time      :     657.67 us | GRBM_GUI_ACTIVE = 870,671
  Insts (VALU/LDS/Rd/Wr): 41,996,288 / 16,781,312 / 0 / 4,096
  SQ_LDS_BANK_CONFLICT : 0 | SQ_WAIT_INST_LDS = 160,742,378

Kernel: _Z24lds_bank_conflict_kernelILi1EEvPf (calls=25)
  Grid=262144 | Block=1024 | LDS=4096B | Scratch=0B | VGPR=24 (Accum=0) | SGPR=16
  Avg Kernel Time      :   12908.98 us | GRBM_GUI_ACTIVE = 16,797,098
  Insts (VALU/LDS/Rd/Wr): 42,000,384 / 16,781,312 / 0 / 4,096
  SQ_LDS_BANK_CONFLICT : 1,040,187,392 | SQ_WAIT_INST_LDS = 2,944,249,187

Kernel: _Z24lds_bank_conflict_kernelILi2EEvPf (calls=25)
  Grid=262144 | Block=1024 | LDS=4224B | Scratch=0B | VGPR=24 (Accum=0) | SGPR=16
  Avg Kernel Time      :     676.26 us | GRBM_GUI_ACTIVE = 894,555
  Insts (VALU/LDS/Rd/Wr): 50,384,896 / 16,781,312 / 0 / 4,096
  SQ_LDS_BANK_CONFLICT : 0 | SQ_WAIT_INST_LDS = 99,406,587

Kernel: _Z24lds_bank_conflict_kernelILi3EEvPf (calls=25)
  Grid=262144 | Block=1024 | LDS=4096B | Scratch=0B | VGPR=16 (Accum=0) | SGPR=16
  Avg Kernel Time      :     674.75 us | GRBM_GUI_ACTIVE = 892,620
  Insts (VALU/LDS/Rd/Wr): 50,388,992 / 16,781,312 / 0 / 4,096
  SQ_LDS_BANK_CONFLICT : 0 | SQ_WAIT_INST_LDS = 94,050,631
```

### 结构化汇总表

| LDS 访问模式 | LDS 占用 | `SQ_INSTS_LDS` | **`SQ_LDS_BANK_CONFLICT`** | **`SQ_WAIT_INST_LDS`** | 平均耗时 (ms) | 相对无冲突 |
|---|---:|---:|---:|---:|---:|---:|
| `0. Row-wise` | `4096 B` | `16,781,312` | **`0`** | `160,742,378` | `0.6578` | `1.00x` |
| `1. Col-wise (32-way)` | `4096 B` | `16,781,312` | **`1,040,187,392`** | **`2,944,249,187`** | **`12.9097`** | **慢 `19.62x`** |
| `2. Padding [32][33]` | `4224 B` | `16,781,312` | **`0`** | `99,406,587` | `0.6769` | 恢复 **`19.07x`** |
| `3. XOR Swizzle` | **`4096 B`** | `16,781,312` | **`0`** | `94,050,631` | **`0.6751`** | 恢复 **`19.12x`** |

---

## 4. 现象拆解与工程结论

1. **10.4 亿次硬件冲突铁证：** Mode 1 执行了完全相同的 `1678 万` 次 LDS 指令，但 `SQ_LDS_BANK_CONFLICT` 暴涨至 **10.4 亿次**，`SQ_WAIT_INST_LDS` 等待周期翻了 **18.3 倍**（29.4 亿周期）。
2. **Swizzle 是最优解：** Padding 多占用了 `128 B` LDS，在 LDS 紧张的算子中可能压低 Occupancy；而 **XOR Swizzle 在零额外 LDS 开销（`4096 B`）下完美清零冲突**，应作为矩阵转置、GEMM 与 FlashAttention 共享内存布局的首选方案。
