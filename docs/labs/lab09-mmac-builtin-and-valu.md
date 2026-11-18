# E07 · MMAC 与 VALU 指令吞吐

**研究问题**　用寄存器常驻操作数和独立累加链，在 BW100 上能测到多少指令吞吐？

本次复测的 MMAC 吞吐约为 333 TFLOPS，口径为 FP16 输入、FP32 累加、16×16×16 指令。结果用于与同卡型、同指令形状的微基准对照。[适用环境](../baselines/bw100-gfx936.md)

## BW100 结果

| 路径 | HIP event 耗时 (ms) | 实测吞吐 (TFLOPS) |
|---|---:|---:|
| VALU FP32，8 路独立链 | 0.8499 | 20.21 |
| MMAC FP16 → FP32，8 路独立链 | 3.2970 | 333.49 |

单依赖链结果为 **10.01 clock64 tick/指令**。[下载原始记录与 ISA](/data/bw100-20261118.zip)

## 复现命令

```bash
cd experiments
hipcc -O3 -std=c++17 --offload-arch=gfx936 08_mmac_builtin_bench.hip -o 08_mmac_builtin_bench
./08_mmac_builtin_bench
```

> **源码：** [08_mmac_builtin_bench.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/08_mmac_builtin_bench.hip)
>
> **关联理论：** [第 2 章：MMAC 矩阵计算能力](../02-compute/cu-and-simt.md)

## 小白也能懂的原理：普通“算盘” vs 矩阵“印钞机”

GPU 里有两种截然不同的算术单元：

- **VALU（向量算术单元）：** 像每个人手里拿一把算盘，一条指令让 64 个线程各自算一次 `a * b + c`（FMA）。
- **MMAC（矩阵乘加引擎）：** 像一台专门印矩阵的“大型印钞机”，通过海光原生的 **`__builtin_hcu_mmac_f32_16x16x16_f16`** 内置指令，让整个 Wave64（64 个线程）手拉手协作，**一条指令直接算完一个 `16 × 16 × 16` 的半精度矩阵乘法并累加到 FP32 结果中**（共 $2 \times 16 \times 16 \times 16 = \mathbf{8192\text{ FLOPs}}$）！

本实验直接调用海光 `hcu builtin` 接口，在 BW100 上实测：

1. **VALU FP32 的极限算力**是多少 TFLOPS？
2. **MMAC `16×16×16` FP16 的极限算力**是多少 TFLOPS？
3. **单条 MMAC 指令的依赖延迟**是多少个时钟周期？

## 导出 `.s` 汇编亲眼看看真实指令

用 `hipcc -O3 --offload-arch=gfx936 -save-temps` 导出汇编后，可以看到编译器生成了两条非常关键的底层指令：

1. **VALU 竟然自动生成了打包指令 `v_pk_fma_f32`：**
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
   - 依赖链测得约 10 个 `clock64()` tick/指令，计时包含循环和测量边界开销。

::: details 查看源码

<<< ../../experiments/08_mmac_builtin_bench.hip{cpp}

:::

::: details 查看原始终端输出

<<< ../public/data/bw100-20261118/e07.txt{text}

:::
