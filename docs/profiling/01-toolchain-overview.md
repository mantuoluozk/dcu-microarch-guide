# 2.1 Profiling 工具链全景与软件栈探测

在动手优化任何算子之前，必须先建立**“从上层框架 → 底层共享库 → 硬件规格”**的可观测能力。本节梳理海光 DCU 上常用的性能分析工具链，并给出环境探测脚本。

---

## 1. 海光 DCU 性能分析工具矩阵

| 层级 | 现有工具 | 主要用途 | 状态 |
|---|---|---|---|
| **框架级** | `torch.profiler` | 抓取 PyTorch / vLLM / SGLang 算子调用栈、CPU/GPU 耗时占比与底层实际派发的 Kernel 名称 | 已实战使用 |
| **Kernel / PMC 级** | `/opt/dtk/bin/hipprof` | 抓取 HIP API Trace 与硬件性能计数器（`--pmc`），定位 Spill、Bank Conflict、L2 Miss 与 Stall | 已实战使用 |
| **指令 / 汇编级** | `hipcc -save-temps` / Triton `compiled.asm` | 导出设备端 AMDGCN ISA（`.s`），检查 `.vgpr_count`、`s_waitcnt` 与 `v_mmac` 生成情况 | 已实战使用 |
| **微架构仿真级** | DCUSIM（内部仿真工具） | 指令流水与硬件资源竞争级仿真分析 | 待补充 |
| **自研辅助工具** | `parse_hipprof_pmc.py` 等 | 自动解析 `hipprof` 导出的海量 PMC CSV，按 Kernel 聚合核心瓶颈指标 | 已实战使用 |

---

## 2. 软件栈链路与算子库探测（`09_hygon_ops_and_stack.py`）

通过读取 `/proc/self/maps` 与 Python 模块元数据，快速摸清当前容器加载的 HIP/DTK/HSA 动态库以及已安装的算子库：

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
```

### BW100 实验环境输出

```text
$ python3 advanced/09_hygon_ops_and_stack.py
=== 1. Software Stack Probe (PyTorch -> HIP/DTK -> HSA) ===
  torch.__version__ : 2.5.1
  torch.version.hip : 6.3.26093
  Device 0          : BW100
  gcnArchName       : gfx936:sramecc+:xnack-
  CU count          : 64

  Loaded Runtime Shared Libraries:
    libMIOpen          -> /opt/dtk-26.04/lib/libMIOpen-recommend.so
    libMIOpen          -> /opt/dtk-26.04/lib/libMIOpen.so.1.0
    libhipblas         -> /opt/dtk-26.04/lib/libhipblas.so.2.3
    libhipblas         -> /opt/dtk-26.04/lib/libhipblaslt.so.0.10
    libhsa-runtime64   -> /opt/hyhal/lib/libhsa-runtime64.so.1.11.0
    librccl            -> /opt/dtk-26.04/lib/librccl.so.1.0
    librocblas         -> /opt/dtk-26.04/lib/librocblas.so.4.3

=== 2. HYGON / ROCm Operator Libraries Probe ===
  [INSTALLED] triton     | ver=3.1.0
  [INSTALLED] lightop    | ver=0.6.0
  [INSTALLED] vllm       | ver=0.11.0
```

---

## 3. 硬件基础规格探测（`01_device_info_and_isa.hip`）

运行 [01_device_info_and_isa.hip](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/01_device_info_and_isa.hip) 获取 `hipDeviceProp_t` 全量参数：

```text
$ ./01_device_info_and_isa
=== Device 0: BW100 ===
  gcnArchName                : gfx936:sramecc+:xnack-
  CU count (multiProcessor)  : 64
  warpSize (Wavefront size)  : 64
  Max threads / CU           : 2560 (Max Waves/CU = 40)
  Max threads / Block        : 1024
  LDS (sharedMemPerBlock)    : 64 KB
  Regs / Block (32-bit)      : 196608
  L2 Cache Size              : 8192 KB
  Global Mem                 : 63.98 GB
  Memory Bus Width           : 4096-bit
  Memory Clock Rate          : 1800.00 MHz
  Core Clock Rate            : 1300.00 MHz
  Theoretical Peak HBM BW*   : 1843.2 GB/s (DDR approx)
```
