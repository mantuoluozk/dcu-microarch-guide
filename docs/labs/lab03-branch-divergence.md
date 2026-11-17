# 实验 3：SIMT 分支发散与编译器 Predicated If-Conversion

> **源码：** [04_branch_divergence.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/04_branch_divergence.hip)  
> **关联章节：** [第 2 章：SIMT 分支发散与执行掩码](../02-compute/cu-and-simt.md)

---

## 1. 实验设计

在 4096 轮循环中对比 4 种分支形态：
- `Mode 0`：无分支（Baseline）
- `Mode 1`：Wave-uniform 条件 `(threadIdx.x / warpSize) & 1`
- `Mode 2`：2-way Lane-divergent 条件 `threadIdx.x & 1`
- `Mode 3`：4-way Lane-divergent 条件 `threadIdx.x & 3`

---

## 2. BW100 实测结果

| 分支模式 | 平均耗时 (ms) | 相对 Baseline 倍率 |
|---|---:|---:|
| `0. No branch (Baseline)` | `1.6253` | `1.00x` |
| `1. Wave-uniform branch` | `4.0450` | `2.49x` |
| `2. 2-way Lane-divergent` | `4.0451` | `2.49x` |
| `3. 4-way Lane-divergent` | **`11.2303`** | **`6.91x`**（相对 2-way 慢 `2.78x`） |

---

## 3. 现象拆解：为什么 Mode 1 和 Mode 2 一样慢？

1. **短分支的 Predicated If-Conversion：** Mode 1 和 Mode 2 的 `if / else` 体内只有 1 条 FMA，LLVM 后端会自动消除跳转，改为**两路都算，再用 `v_cndmask_b32` 选出正确结果**；加上 `warpSize` 在 HIP 中作为运行时变量做除法未能被静态识别为标量，因此二者耗时完全重合（`4.045 ms`）。
2. **多路/重分支的真实 `EXEC` 串行化：** 到 Mode 3（4-way 分支）时，编译器无法再用单条 `v_cndmask` 化简，**触发真实的 `EXEC` 掩码压栈与多路串行执行**，耗时直接飙升至 `11.2303 ms`（`6.91x`）。
