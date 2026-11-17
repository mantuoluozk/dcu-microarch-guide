# 实验导览：环境探测、ISA 导出与 `hipprof` 工具链

在动手写任何高性能算子之前，第一步是建立**“从 Python/C++ → 编译器 ISA → 硬件计数器”**的观测闭环。本章带你用两段探测代码摸清海光 BW100 的软件栈链路与物理硬件规格，并掌握导出 AMDGCN 汇编与抓取 `hipprof` PMC 计数器的方法。

---

## 1. 软件栈链路探测（`09_hygon_ops_and_stack.py`）

参考 `hello-rocm` 中“解密 AI 加速器”的方法，我们先用脚本追踪 PyTorch 进程实际加载了哪些底层 `.so` 动态库（类似 `ldd`），并检查容器内已安装的 DSL / 专项算子库，再用 `torch.profiler` 观察一次 `matmul` 到底派发了哪个底层 Kernel。

### 完整源码

```python title="experiments/advanced/09_hygon_ops_and_stack.py"
import importlib
import os
import torch


def trace_loaded_libs():
    print("=== 1. Software Stack Probe (PyTorch -> HIP/DTK -> HSA) ===")
    print(f"  torch.__version__ : {torch.__version__}")
    print(f"  torch.version.hip : {getattr(torch.version, 'hip', None)}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  Device 0          : {props.name}")
        print(f"  gcnArchName       : {getattr(props, 'gcnArchName', 'N/A')}")
        print(f"  CU count          : {props.multi_processor_count}")

    maps_path = "/proc/self/maps"
    if os.path.exists(maps_path):
        keywords = ("libamdhip64", "libhsa-runtime64", "librocblas", "libhipblas", "librccl", "libMIOpen")
        found = set()
        with open(maps_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                for kw in keywords:
                    if kw in line and ".so" in line:
                        path = line.strip().split()[-1]
                        found.add((kw, path))
        print("\n  Loaded Runtime Shared Libraries:")
        for kw, path in sorted(found):
            print(f"    {kw:<18s} -> {path}")


def probe_op_libraries():
    print("\n=== 2. HYGON / ROCm Operator Libraries Probe ===")
    candidates = ["triton", "lightop", "aiter", "deep_gemm", "deepgemm", "vllm", "sglang"]
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "unknown")
            loc = getattr(mod, "__file__", "namespace")
            attrs = [a for a in dir(mod) if not a.startswith("_")][:12]
            print(f"  [INSTALLED] {name:<10s} | ver={ver} | path={loc}")
            print(f"              sample exports: {attrs}")
        except ImportError:
            print(f"  [MISSING]   {name}")


def profile_pytorch_kernels():
    print("\n=== 3. torch.profiler Kernel Dispatch Trace ===")
    if not torch.cuda.is_available():
        return
    x = torch.randn(2048, 4096, device="cuda", dtype=torch.float16)
    w = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    _ = torch.matmul(x, w)
    torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        y = torch.matmul(x, w)
        _ = (y * y).mean(dim=-1)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    trace_loaded_libs()
    probe_op_libraries()
    profile_pytorch_kernels()
```

### BW100 容器（`zk-migraphx`）实测输出解读

- **软件栈版本：** `torch 2.5.1`（`torch.version.hip = 6.3.26093`）、`triton 3.1.0`（`triton/backends/hcu/`）、`lightop 0.6.0`、`vllm 0.11.0`。
- **底层共享库：**
  - `DTK-26.04`：`/opt/dtk-26.04/lib/librocblas.so.4.3`、`libhipblas.so.2.3`、`libhipblaslt.so.0.10`、`libMIOpen.so.1.0`、`librccl.so.1.0`
  - `HSA Runtime`：`/opt/hyhal/lib/libhsa-runtime64.so.1.11.0`
- **Kernel 派发真相：** `torch.matmul(2048x4096, 4096x4096)` 底层实际调用的是 Tensile 针对 BW100 调优的汇编 Kernel：`Cijk_Ailk_Bljk_HBH_MT256x256x16_SE_AMAS3_BW_DTL0_BL1...`（`261.12 us`）。

---

## 2. 硬件规格探测与计时框架（`common.h` & `01_device_info_and_isa.hip`）

### 公共计时头文件解读

为了保证所有 HIP 微基准的计时一致且排除冷启动/编译缓存干扰，`experiments/common.h` 封装了 `HIP_CHECK` 宏与 `benchmark_kernel` 模板：先预热 `warmup = 5` 次让硬件频率与指令缓存稳定，再用 `hipEventRecord` 连续跑 `iters = 20` 次取平均毫秒数。

```cpp title="experiments/common.h"
#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call)                                                      \
    do {                                                                     \
        hipError_t err = (call);                                             \
        if (err != hipSuccess) {                                             \
            std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, hipGetErrorString(err));                  \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

template <typename Func>
float benchmark_kernel(Func launch_fn, int warmup = 5, int iters = 20) {
    for (int i = 0; i < warmup; ++i) launch_fn();
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) launch_fn();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return total_ms / static_cast<float>(iters);
}
```

### 硬件属性探测源码

```cpp title="experiments/01_device_info_and_isa.hip"
#include "common.h"
#include <vector>

__global__ void sample_fma_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  float alpha,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = a[idx];
        float vb = b[idx];
        c[idx] = alpha * va + vb;
    }
}

int main() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    for (int dev = 0; dev < count; ++dev) {
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, dev));
        std::printf("=== Device %d: %s ===\n", dev, prop.name);
        std::printf("  gcnArchName                : %s\n", prop.gcnArchName);
        std::printf("  CU count (multiProcessor)  : %d\n", prop.multiProcessorCount);
        std::printf("  warpSize (Wavefront size)  : %d\n", prop.warpSize);
        std::printf("  Max threads / CU           : %d (Max Waves/CU = %d)\n",
                    prop.maxThreadsPerMultiProcessor,
                    prop.maxThreadsPerMultiProcessor / prop.warpSize);
        std::printf("  Max threads / Block        : %d\n", prop.maxThreadsPerBlock);
        std::printf("  LDS (sharedMemPerBlock)    : %zu KB\n", prop.sharedMemPerBlock / 1024);
        std::printf("  Regs / Block (32-bit)      : %d\n", prop.regsPerBlock);
        std::printf("  L2 Cache Size              : %d KB\n", prop.l2CacheSize / 1024);
        std::printf("  Global Mem                 : %.2f GB\n",
                    static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
        std::printf("  Memory Bus Width           : %d-bit\n", prop.memoryBusWidth);
        std::printf("  Memory Clock Rate          : %.2f MHz\n", prop.memoryClockRate / 1000.0);
        std::printf("  Core Clock Rate            : %.2f MHz\n", prop.clockRate / 1000.0);
        double theo_bw_gbs = 2.0 * (prop.memoryClockRate * 1e3) * (prop.memoryBusWidth / 8.0) / 1e9;
        std::printf("  Theoretical Peak HBM BW*   : %.1f GB/s (DDR approx)\n\n", theo_bw_gbs);
    }
    return 0;
}
```

### BW100 实测硬件规格表

| 硬件参数 | BW100 实测值 | 微架构与调优意义 |
|---|---|---|
| 架构代号 (`gcnArchName`) | `gfx936:sramecc+:xnack-` | 编译指定 `ARCH=--offload-arch=gfx936` 可避免多架构胖二进制编译 |
| CU 数量 | `64` | Grid 中的 Block 数至少为 64 的整数倍才能打满全卡 |
| `warpSize` | `64`（Wave64） | Block Size 必须按 64 对齐 |
| 每 CU 最大驻留 Wave 数 | `40`（`2560` 线程） | 理论 Occupancy 的上限分母 |
| LDS 容量 | `64 KB` | 单 Block 用 >32 KB 时每 CU 只能驻留 1 个 Block |
| 每 Block 寄存器池 | `196608`（32-bit） | 决定单 Block 可用的最大 VGPR/SGPR 总额 |
| L2 Cache 容量 | `8192 KB`（`8 MB`） | 超过 8 MB 的工作集将穿透至 HBM |
| HBM 显存与位宽 | `63.98 GB`, `4096-bit @ 1800 MHz` | 理论峰值约 `1843.2 GB/s` |

---

## 3. 导出设备端 ISA 与抓取 `hipprof` PMC

### 导出 AMDGCN ISA 汇编

```bash
cd /exp/dcu_microarch_exp
make isa    # 使用 hipcc -O3 -save-temps 导出 .s 汇编到 isa_dump/
```

在生成的 `.s` 文件中重点检查：
- `.vgpr_count` / `.sgpr_count` / `.lds_size`：静态资源占用。
- `global_load_dword*` 与 `s_waitcnt vmcnt(k)`：异步加载与等待点。
- `s_cbranch_scc*` vs `s_and_saveexec_b64` / `v_cndmask_b32`：分支指令形态。

### 用 `hipprof` 抓取硬件 PMC 计数器

海光 DTK-26.04 提供的硬件计数器工具为 `/opt/dtk/bin/hipprof`，配合 [parse_hipprof_pmc.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/parse_hipprof_pmc.py) 一键汇总每个 Kernel 的硬件计数器：

```bash
mkdir -p prof_out
hipprof --pmc --pmc-type 3 -o prof_out/my_exp ./my_binary
python3 parse_hipprof_pmc.py prof_out/my_exp.csv
```
