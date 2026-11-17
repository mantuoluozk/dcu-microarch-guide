# 实验 9：用 HCU Builtin 直呼 MMAC 矩阵引擎——测出 BW100 算力天花板

> **源码：** [08_mmac_builtin_bench.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/08_mmac_builtin_bench.hip)  
> **关联理论：** [第 2 章：MMAC 矩阵计算能力](../02-compute/cu-and-simt.md)

---

## 1. 小白也能懂的原理：普通“算盘” vs 矩阵“印钞机”

GPU 里有两种截然不同的算术单元：

- **VALU（向量算术单元）：** 像每个人手里拿一把算盘，一条指令让 64 个线程各自算一次 `a * b + c`（FMA）。
- **MMAC（矩阵乘加引擎）：** 像一台专门印矩阵的“大型印钞机”，通过海光原生的 **`__builtin_hcu_mmac_f32_16x16x16_f16`** 内置指令，让整个 Wave64（64 个线程）手拉手协作，**一条指令直接算完一个 `16 × 16 × 16` 的半精度矩阵乘法并累加到 FP32 结果中**（共 $2 \times 16 \times 16 \times 16 = \mathbf{8192\text{ FLOPs}}$）！

本实验直接调用海光 `hcu builtin` 接口，在 BW100 上实测：

1. **VALU FP32 的极限算力**是多少 TFLOPS？
2. **MMAC `16×16×16` FP16 的极限算力**是多少 TFLOPS？
3. **单条 MMAC 指令的依赖延迟**是多少个时钟周期？

---

## 2. 完整源码与逐段解读

```cpp title="experiments/08_mmac_builtin_bench.hip"
#include "common.h"

// 定义 LLVM 向量类型：对应 AMDGCN 中的连续 VGPR 寄存器组
using float4_t = float __attribute__((ext_vector_type(4)));
using half4_t = _Float16 __attribute__((ext_vector_type(4)));

constexpr int ITERS = 4096;

// 1. VALU FP32 峰值测试：每线程开 8 路互不依赖的累加器，打满向量流水线
__global__ void valu_fp32_peak_kernel(float* __restrict__ out, float seed) {
    float a0 = seed + 0.1f, a1 = seed + 0.2f, a2 = seed + 0.3f, a3 = seed + 0.4f;
    float a4 = seed + 0.5f, a5 = seed + 0.6f, a6 = seed + 0.7f, a7 = seed + 0.8f;
    float b = 1.0001f, c = 0.0001f;

    for (int i = 0; i < ITERS; ++i) {
        a0 = a0 * b + c; a1 = a1 * b + c; a2 = a2 * b + c; a3 = a3 * b + c;
        a4 = a4 * b + c; a5 = a5 * b + c; a6 = a6 * b + c; a7 = a7 * b + c;
    }
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    out[gid] = (a0 + a1 + a2 + a3) + (a4 + a5 + a6 + a7);
}

// 2. HCU Builtin MMAC 峰值测试：每 Wave 开 8 路独立的 16x16x16 矩阵累加器
__global__ void mmac_fp16_peak_kernel(float* __restrict__ out, _Float16 seed) {
    half4_t a = {seed, seed, seed, seed};
    half4_t b = {(_Float16)1.0, (_Float16)1.0, (_Float16)1.0, (_Float16)1.0};
    float4_t c0 = {0, 0, 0, 0}, c1 = {0, 0, 0, 0}, c2 = {0, 0, 0, 0}, c3 = {0, 0, 0, 0};
    float4_t c4 = {0, 0, 0, 0}, c5 = {0, 0, 0, 0}, c6 = {0, 0, 0, 0}, c7 = {0, 0, 0, 0};

    for (int i = 0; i < ITERS; ++i) {
        c0 = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c0);
        c1 = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c1);
        c2 = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c2);
        c3 = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c3);
        c4 = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c4);
        c5 = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c5);
        c6 = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c6);
        c7 = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c7);
    }
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float4_t s = (c0 + c1 + c2 + c3) + (c4 + c5 + c6 + c7);
    out[gid] = s[0] + s[1] + s[2] + s[3];
}

// 3. 单条 MMAC 依赖延迟测试：仅用 1 路累加器 c，逼迫硬件等上一条算完才能算下一条
__global__ void mmac_fp16_latency_kernel(float* __restrict__ out,
                                         unsigned long long* __restrict__ out_cycles,
                                         _Float16 seed) {
    half4_t a = {seed, seed, seed, seed};
    half4_t b = {(_Float16)1.0, (_Float16)1.0, (_Float16)1.0, (_Float16)1.0};
    float4_t c = {0, 0, 0, 0};

    unsigned long long t0 = clock64();
    for (int i = 0; i < ITERS; ++i) {
        c = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c);
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) {
        out[0] = c[0];
        out_cycles[0] = t1 - t0;
    }
}
```

### 为什么每个 Lane 传入 `half4_t`（4 个 FP16），返回 `float4_t`（4 个 FP32）？
算一笔账就懂了：

- 输出矩阵大小是 $16 \times 16 = 256$ 个 FP32 元素；
- 一个 Wave64 有 64 个 Lane，正好 $256 \div 64 = \mathbf{4}$ 个 FP32/Lane（即 `float4_t`）！
- 输入矩阵 $A$（$16 \times 16$）和 $B$（$16 \times 16$）各有 256 个 FP16 元素，分摊到 64 个 Lane，正好也是每人拿 $256 \div 64 = \mathbf{4}$ 个 FP16（即 `half4_t`）！

---

## 3. 复现命令、ISA 汇编观察与 BW100 实测数据

```bash
cd /exp/dcu_microarch_exp
hipcc -O3 -std=c++17 --offload-arch=gfx936 08_mmac_builtin_bench.hip -o 08_mmac_builtin_bench
./08_mmac_builtin_bench
```

### 原始终端输出

```text
$ ./08_mmac_builtin_bench
=== Exp 9: VALU vs HCU Builtin MMAC Peak & Latency (BW100, CU=64) ===
  VALU FP32 Peak   :   0.8499 ms |   20.21 TFLOPS
  MMAC 16x16x16 FP16:   3.2970 ms |  333.49 TFLOPS (16.50x vs VALU FP32)
  MMAC 16x16x16 Dependent Latency : 10.02 cycles / instruction
```

### 结构化汇总表

### BW100 实测结果

| 测试项目 | 平均耗时 / 延迟 | 实测算力 (TFLOPS) | 相对 VALU 倍率 |
|---|---|---:|---:|
| `VALU FP32 Peak`（8 路独立） | `0.8499 ms` | **`20.21`** | `1.00x` |
| **`MMAC 16×16×16 FP16`（8 路独立）** | `3.2970 ms` | **`333.49`** | **`16.50x`** |
| `MMAC 16×16×16` 单指令依赖延迟 | **`10.02 cycles / 指令`** | - | - |

### 导出 `.s` 汇编亲眼看看真实指令

用 `hipcc -O3 --offload-arch=gfx936 -save-temps` 导出汇编后，可以看到编译器生成了两条非常关键的底层指令：

1. **VALU 竟然自动打包成了双发射 `v_pk_fma_f32`：**
   ```asm
   v_pk_fma_f32 v[1:2], v[1:2], v[9:10], v[11:12]
   ```
   BW100（`gfx936`）支持 **Packed FP32 FMA**（一条指令同时算 2 个 FP32 FMA），因此实测 VALU FP32 峰值冲到了 **`20.21 TFLOPS`**！

2. **MMAC 直接映射为原生矩阵指令：**
   ```asm
   v_mmac_f32_16x16x16_f16 v[1:4], v[5:6], v[7:8], v[1:4]
   ```
   - `v[5:6]`（2 个 32-bit VGPR = 4 个 FP16）放 $A$；
   - `v[7:8]`（2 个 32-bit VGPR = 4 个 FP16）放 $B$；
   - `v[1:4]`（4 个 32-bit VGPR = 4 个 FP32）既是输入 $C$ 又是输出 $D$；
   - 单条依赖延迟仅 **`10.02 cycles`**，只要写算子时展开 4~8 路独立累加器（ILP）或驻留足够 Wave（TLP），就能轻松掩盖这 10 个周期的流水线延迟，触及 **`333.49 TFLOPS`** 的纯硅片算力极限！
