# 实验 8：用“寻宝游戏”测出 L1 / L2 / HBM 真实延迟阶梯

> **源码：** [07_memory_latency_ladder.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/07_memory_latency_ladder.hip)  
> **关联理论：** [第 3 章：存储体系](../03-memory/memory-hierarchy.md)

---

## 1. 小白也能懂的原理：什么是 Pointer-Chasing（指针追逐）？

在前面实验 1 中，我们用成千上万个线程一起读显存，测出来的是**“带宽”**（就像 100 辆卡车一起拉货，每秒能拉多少吨）。但如果我们想知道**“单辆卡车跑一趟来回到底要多久（延迟 Latency）”**，该怎么测？

如果还是派 100 个线程一起读，GPU 调度器会把等待时间藏起来；如果只派 1 个线程顺序读 `a[0], a[1], a[2]...`，硬件预取器又会提前帮你把下一个数据偷渡进缓存。

于是我们用了一个经典技巧——**Pointer-Chasing（寻宝接力赛）**：
- 在数组里随机连成一个大环：`buf[0]` 里存着下一个要去的下标，比如 `48`；去了 `buf[48]` 才看到下一张纸条写着 `12`……
- 每次循环只做一件事：`idx = next_idx[idx]`。
- 因为**不知道下一张纸条写着啥，硬件绝对无法提前预取**，只能老老实实等当前这次访存从内存里返回，才能发起下一次！
- 我们从小到大改变“寻宝地图”的大小（`4 KB → 16 KB → 32 KB → ... → 64 MB`）：
  - 当地图能塞进 **L1 Cache**（随身小兜）时，延迟极低；
  - 一旦超过 L1 塞不下了，延迟会**突然跳上一个台阶**，掉进 **L2 Cache**（楼层公共柜）；
  - 超过阈值掉出片上缓存后，延迟会**再跳上一个台阶**，直达 **HBM 显存**（楼下大仓库）！

---

## 2. 完整源码与逐段解读

```cpp title="experiments/07_memory_latency_ladder.hip"
#include "common.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

// 1. 全局内存指针追逐：只派 1 个线程，消除并发隐藏，暴露最纯粹的往返延迟
__global__ void global_chase_kernel(const int* __restrict__ next_idx,
                                    int* __restrict__ out,
                                    unsigned long long* __restrict__ out_cycles,
                                    int steps) {
    int idx = 0;
    // 先预热一轮：让小工作集的数据提前住进 L1 / L2 Cache
    for (int i = 0; i < steps; ++i) {
        idx = next_idx[idx];
    }

    // 用 GPU 硬件时钟 clock64() 记录纯粹的追逐周期数
    unsigned long long t0 = clock64();
    for (int i = 0; i < steps; ++i) {
        idx = next_idx[idx];
    }
    unsigned long long t1 = clock64();

    out[0] = idx;
    out_cycles[0] = t1 - t0;
}

// 2. LDS 指针追逐：对比片上共享内存的延迟
__global__ void lds_chase_kernel(int* __restrict__ out,
                                 unsigned long long* __restrict__ out_cycles,
                                 int n_elems,
                                 int steps) {
    __shared__ int s_next[4096]; // 16 KB LDS
    for (int i = 0; i < n_elems; ++i) {
        s_next[i] = (i + 17) % n_elems; // 互质步长构建环形链
    }
    __syncthreads();

    int idx = 0;
    for (int i = 0; i < steps; ++i) idx = s_next[idx];
    unsigned long long t0 = clock64();
    for (int i = 0; i < steps; ++i) idx = s_next[idx];
    unsigned long long t1 = clock64();

    out[0] = idx;
    out_cycles[0] = t1 - t0;
}
```

### 代码设计细节（为什么每个跳跃间隔 64 字节？）
在 `main()` 构建随机环时，我们每隔 `LINE_INTS = 16`（即 `16 × 4B = 64 字节`）放一个节点，保证每次跳跃都打在不同的 Cache Line 上，避免同一次 Cache Line 加载顺带命中多个节点。

---

## 3. 复现命令与 BW100 实测延迟阶梯

```bash
cd /exp/dcu_microarch_exp
hipcc -O3 -std=c++17 --offload-arch=gfx936 07_memory_latency_ladder.hip -o 07_memory_latency_ladder
./07_memory_latency_ladder
```

| 测试区域 / 工作集大小 | 实测平均延迟 (Cycles) | 换算绝对时间 (ns @ 1.30 GHz) | 所处存储层级判断 |
|---|---:|---:|---|
| `LDS (16 KB)` | `64.1` | `49.28 ns` | 片上共享内存（LDS） |
| `4 KB` | **`57.1`** | **`43.89 ns`** | **L1 Cache（命中！）** |
| `16 KB` | **`57.1`** | **`43.91 ns`** | **L1 Cache（16 KB 临界点）** |
| `32 KB` | **`216.4`** | **`166.46 ns`** | **跌出 L1 → L2 Cache（延迟跳涨 3.8 倍！）** |
| `64 KB` | `208.4` | `160.29 ns` | L2 Cache |
| `256 KB` | `216.4` | `166.49 ns` | L2 Cache |
| `1024 KB (1 MB)` | **`415.5`** | **`319.64 ns`** | **穿透到全局显存（HBM）** |
| `4096 KB ~ 65536 KB` | `406.7 ~ 417.9` | `312.83 ~ 321.49 ns` | HBM 全局显存稳态延迟 |

---

## 4. 小白也能看懂的核心结论

1. **BW100 的 Vector L1 Cache 容量精准实测为 `16 KB`：** 工作集在 `4 KB` 和 `16 KB` 时延迟稳稳停在 `57.1 cycles`；一跨过 `16 KB` 到 `32 KB`，延迟立刻翻了近 4 倍跳到 `216.4 cycles`！
2. **三层存储的真实时间差距：**
   - **LDS / L1：** 约 `44 ~ 49 ns`（~60 周期）
   - **L2 Cache：** 约 `160 ~ 166 ns`（~215 周期，比 L1 慢 ~3.8 倍）
   - **HBM 显存：** 约 `313 ~ 321 ns`（~407 周期，比 L1 慢 **7 倍以上**）
3. 这也解释了为什么前面实验中我们需要那么多 Wavefront 和软件流水线：单次去 HBM 仓库拿数据要等 400 多个周期，不靠并发和缓存复用，计算单元就会一直干等！
