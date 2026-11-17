# 实验 2：LDS Bank Conflict 与 Padding / XOR Swizzle 消除

> **源码：** [03_lds_bank_conflict.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/03_lds_bank_conflict.hip)  
> **关联章节：** [第 3 章：LDS Bank Conflict](../03-memory/memory-hierarchy.md)

---

## 1. 实验设计

在 `__shared__ float tile[32][32]`（以及 Padding 版 `tile[32][33]`）上循环执行 **2048 轮** LDS 读-改-写（彻底屏蔽 HBM 干扰），对比：
- `Mode 0`：行访问 `tile[r][tx]`（相邻 Lane 落入不同 Bank，0 冲突）
- `Mode 1`：列访问 `tile[tx][r]`（相邻 Lane 撞在同一 Bank，32-way 冲突）
- `Mode 2`：列访问 + Padding `tile_pad[32][33]`（步长 33 与 32 互质）
- `Mode 3`：列访问 + XOR Swizzle `tile_std[tx][r ^ tx]`（零额外 LDS 占用）

```bash
./03_lds_bank_conflict
hipprof --pmc --pmc-type 3 -o prof_out/exp03 ./03_lds_bank_conflict
python3 parse_hipprof_pmc.py prof_out/exp03.csv
```

---

## 2. BW100 实测性能与 PMC 硬件计数器

| LDS 访问模式 | LDS 占用 | `SQ_INSTS_LDS` | **`SQ_LDS_BANK_CONFLICT`** | **`SQ_WAIT_INST_LDS`** | 平均耗时 (ms) | 相对无冲突 |
|---|---:|---:|---:|---:|---:|---:|
| `0. Row-wise` | `4096 B` | `16,781,312` | **`0`** | `160,742,378` | `0.6578` | `1.00x` |
| `1. Col-wise (32-way)` | `4096 B` | `16,781,312` | **`1,040,187,392`** | **`2,944,249,187`** | **`12.9097`** | **慢 `19.62x`** |
| `2. Padding [32][33]` | `4224 B` | `16,781,312` | **`0`** | `99,406,587` | `0.6769` | 恢复 **`19.07x`** |
| `3. XOR Swizzle` | **`4096 B`** | `16,781,312` | **`0`** | `94,050,631` | **`0.6751`** | 恢复 **`19.12x`** |

---

## 3. 核心结论

1. 32-way 冲突下，4 个模式执行了完全相同的 `1678 万` 次 LDS 指令，但 Mode 1 触发了 **10.4 亿次 `SQ_LDS_BANK_CONFLICT`**，直接让 LDS 等待周期暴涨 **18.3 倍**（29.4 亿周期）。
2. **Padding vs XOR Swizzle：** Padding 多占用了 `128 B` LDS；而 **XOR Swizzle 在零额外 LDS 占用（`4096 B`）下把 `SQ_LDS_BANK_CONFLICT` 直接清零**，仅消耗极轻量的地址异或指令，是 FlashAttention 和高性能 GEMM 的首选。
3. 编译提示观察：未加 `__launch_bounds__(1024)` 时，HIP 编译器会提示 `Launch params (1024, 1, 1) are larger than launch bounds (256)`，提醒开发者显式标注 `__launch_bounds__`。
