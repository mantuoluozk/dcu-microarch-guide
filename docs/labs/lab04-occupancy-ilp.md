# 实验 4：Occupancy 悬崖与 TLP / ILP 延迟隐藏

> **源码：** [05_occupancy_and_ilp.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/05_occupancy_and_ilp.hip)  
> **关联理论：** [第 4 章：Occupancy、TLP 与 ILP](../04-scheduling/occupancy-and-roofline.md)

---

## 1. 实验原理：没有足够 Wavefront 时，CU 会怎样？

一次 HBM 访问延迟高达数百个时钟周期。DCU 隐藏延迟靠两条腿走路：
1. **TLP（线程级并行 / Occupancy）：** Wave 0 等待 HBM 时，CU 调度器立刻切换执行 Wave 1 ~ Wave N。如果因为 VGPR 或 LDS 占用过大，导致每 CU 只能驻留极少数 Wavefront，调度器就会“无 Wave 可切”，只能干等。
2. **ILP（指令级并行）：** 单个 Wavefront 内部连续发射多条互不依赖的独立加载指令（在 `s_waitcnt` 前一起在途飞行），也能在一定程度上摊薄延迟。

---

## 2. 完整源码与逐段解读

```cpp title="experiments/05_occupancy_and_ilp.hip"
#include "common.h"

// Low-ILP：单路串行依赖加载（高度依赖 TLP / 高 Occupancy 隐藏延迟）
__global__ void low_ilp_kernel(const float* __restrict__ a,
                               float* __restrict__ out,
                               int n,
                               int steps) {
    extern __shared__ char dummy_lds[];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    if (gid >= n) return;

    float acc = 0.0f;
    int idx = gid;
    for (int s = 0; s < steps; ++s) {
        acc += a[idx];
        idx += stride;
        if (idx >= n) idx -= n;
    }
    if (threadIdx.x == 0 && dummy_lds[0] == 123) out[0] = 0.0f; // 防止动态 LDS 被优化掉
    out[gid] = acc;
}

// High-ILP：每轮发射 4 条独立加载（在低 Occupancy 下靠单 Wave 内 ILP 抗延迟）
__global__ void high_ilp_kernel(const float* __restrict__ a,
                                float* __restrict__ out,
                                int n,
                                int steps) {
    extern __shared__ char dummy_lds[];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    if (gid >= n) return;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int idx0 = gid;
    int idx1 = (gid + stride) % n;
    int idx2 = (gid + 2 * stride) % n;
    int idx3 = (gid + 3 * stride) % n;

    for (int s = 0; s < steps; s += 4) {
        float v0 = a[idx0];
        float v1 = a[idx1];
        float v2 = a[idx2];
        float v3 = a[idx3];
        acc0 += v0; acc1 += v1; acc2 += v2; acc3 += v3;
        idx0 = (idx0 + 4 * stride) % n;
        idx1 = (idx1 + 4 * stride) % n;
        idx2 = (idx2 + 4 * stride) % n;
        idx3 = (idx3 + 4 * stride) % n;
    }
    if (threadIdx.x == 0 && dummy_lds[0] == 123) out[0] = 0.0f;
    out[gid] = (acc0 + acc1) + (acc2 + acc3);
}
```

### 关键代码设计细节
- **如何精准控制每 CU 驻留 Wave 数？** 通过 `hipLaunchKernelGGL(..., lds_bytes, ...)` 传入 `0 / 16 / 32 / 48 KB` 动态 LDS，并用 `hipOccupancyMaxActiveBlocksPerMultiprocessor` 查询驱动计算出的真实驻留 Block 上限。
- **`dummy_lds[0] == 123` 的作用：** 避免编译器认为动态共享内存未被使用而优化掉资源声明（实际运行时不会触发写入）。

---

## 3. 复现命令与 BW100 实测数据

```bash
cd /exp/dcu_microarch_exp
make 05_occupancy_and_ilp && ./05_occupancy_and_ilp
```

| 动态 LDS 占用 | 每 CU 驻留 Block 数 | 每 CU 驻留 Wave 数（Occupancy） | `Low-ILP` (ms) | `High-ILP` (ms) |
|---|---:|---:|---:|---:|
| `0 KB` | `8` | `32` (`80%`) | **`0.0386`** | `0.0484` |
| `16 KB` | `4` | `16` (`40%`) | `0.0491` | `0.0501` |
| `32 KB` | `2` | `8` (`20%`) | `0.0806` | **`0.0779`** |
| `48 KB` | `1` | `4` (`10%`) | `0.1045` | `0.1218` |

---

## 4. 现象拆解与工程结论

1. **清晰的 Occupancy 阶梯：** BW100 每 CU 64 KB LDS 下，随着单 Block LDS 从 `0 → 16 → 32 → 48 KB`，驻留 Wave 数按 `32 → 16 → 8 → 4` 阶梯腰斩。
2. **Low-ILP 的延迟暴露：** 当 Wave 数从 `32` 跌到 `4` 时，`Low-ILP` 耗时从 `0.0386 ms` 恶化到 `0.1045 ms`（**2.71x 退化**）。
3. **ILP 的代价：** 在 `8 Waves/CU`（32 KB LDS）甜点处，`High-ILP` 反超 `Low-ILP`（`0.0779` vs `0.0806 ms`）；但在 0 KB 与 48 KB 时略慢，是因为 `High-ILP` 循环中 4 路索引使用了额外的 `% n` 整数取模 VALU 指令。这提醒我们：**提升 ILP 时不能引入过重的索引计算开销**。
