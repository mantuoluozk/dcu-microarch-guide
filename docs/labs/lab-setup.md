# 实验导览：环境探测、ISA 导出与 `hipprof` 工具链

本章节介绍如何在海光 DCU（BW100 / `gfx936`）环境中编译运行本仓库的全部实验，并抓取设备端汇编（ISA）与硬件性能计数器（PMC）。

---

## 1. 硬件规格与软件栈探测

在容器内运行探测脚本与基础信息程序：

```bash
cd /exp/dcu_microarch_exp
make 01_device_info_and_isa && ./01_device_info_and_isa
python3 advanced/09_hygon_ops_and_stack.py
```

### BW100 实测硬件画像（`01_device_info_and_isa`）

| 硬件参数 | BW100 实测值 | 优化意义 |
|---|---|---|
| 架构代号 (`gcnArchName`) | `gfx936:sramecc+:xnack-` | 编译指定 `--offload-arch=gfx936` 可加快编译 |
| CU 数量 | `64` | Grid Block 数建议为 64 的整数倍 |
| `warpSize` | `64`（Wave64） | Block Size 应为 64 的倍数 |
| 每 CU 最大驻留 Wave 数 | `40`（`2560` 线程） | 理论 Occupancy 的上限分母 |
| LDS 容量 (`sharedMemPerBlock`) | `64 KB` | 单 Block 用 >32 KB 时每 CU 只能驻留 1 个 Block |
| L2 Cache 容量 | `8192 KB` (`8 MB`) | 工作集超过 8 MB 将穿透至 HBM |
| HBM 显存与位宽 | `63.98 GB`, `4096-bit @ 1800 MHz` | 理论峰值约 `1843.2 GB/s` |

### 容器软件栈链路（`09_hygon_ops_and_stack.py`）

- **PyTorch：** `2.5.1`（`torch.version.hip = 6.3.26093`）
- **DTK 数学库（`DTK-26.04`）：** `librocblas.so.4.3`、`libhipblas.so.2.3`、`libhipblaslt.so.0.10`、`libMIOpen.so.1.0`、`librccl.so.1.0`
- **DSL 与算子库：** `triton 3.1.0`（`triton/backends/hcu/`）、`lightop 0.6.0`、`vllm 0.11.0`
- **HSA 运行时：** `/opt/hyhal/lib/libhsa-runtime64.so.1.11.0`

---

## 2. 导出设备端 AMDGCN ISA 汇编

```bash
cd /exp/dcu_microarch_exp
make isa
```

在 `isa_dump/` 生成 `.s` 文件后，重点关注：
- `.vgpr_count` / `.sgpr_count` / `.lds_size`
- 异步访存 `global_load_dword*` 与 `s_waitcnt vmcnt(k)` / `lgkmcnt(k)`
- 分支指令 `s_cbranch_scc*` vs `s_and_saveexec_b64` / `v_cndmask_b32`

---

## 3. 使用 `hipprof` 抓取硬件 PMC 计数器

海光 DTK-26.04 提供的硬件计数器工具为 `/opt/dtk/bin/hipprof`，配合仓库中的 [parse_hipprof_pmc.py](https://github.com/mantuoluozk/dcu-microarch-guide/blob/main/experiments/parse_hipprof_pmc.py) 使用：

```bash
mkdir -p prof_out
hipprof --pmc --pmc-type 3 -o prof_out/my_exp ./my_binary
python3 parse_hipprof_pmc.py prof_out/my_exp.csv
```

核心关注指标：
- `scr`（Scratch Bytes）：`> 0` 代表发生 **寄存器溢出（Register Spill）**。
- `SQ_INSTS_VMEM_RD / WR`：向量全局访存指令数。
- `SQ_LDS_BANK_CONFLICT` & `SQ_WAIT_INST_LDS`：LDS 冲突次数与等待周期。
- `TCC_HIT / TCC_MISS`：L2 Cache 命中率。
- `TCP_TCP_TA_DATA_STALL_CYCLES`：L1/TA 数据停顿周期。
