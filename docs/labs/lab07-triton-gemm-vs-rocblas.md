# E09 · Triton GEMM 计时与 PMC

**研究问题**　在 BW100 上，分别改变 `BK`、流水级数和分组顺序，能否提高同一 FP16 GEMM 的速度？

对 `4096×4096×4096`，本轮最快的 Triton 配置约 **97.33 TFLOPS**，PyTorch 派发的 rocBLAS kernel 约 **257.08 TFLOPS**。单独降低 `BK` 使 VGPR 和 LDS 等待计数下降，耗时却上升。在这个形状上，单一计数器不足以预测性能。

[适用环境](../baselines/bw100-gfx936.md) · [复现命令](#复现命令)

## BW100 结果

FP16 输入与输出、FP32 累加。各配置通过对 FP32 参考计算的误差检查。计时为无 profiler 的 HIP event 测量，5 轮，每轮 20 次。PMC 为另一进程中 6 次目标 dispatch 的后 5 次平均。

| 配置 | 中位数 (ms) | TFLOPS | LDS (B) | arch_vgpr | scr (B) | SQ_WAIT_INST_LDS |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch → rocBLAS | 0.5346 | 257.08 | 65536 | 212 | 0 | 3.93 M |
| 基准 128×128×32，4 warps，1 stage，GROUP_M=8 | 1.4120 | 97.33 | 8704 | 144 | 0 | 212.54 M |
| 仅 BK=16 | 1.4973 | 91.79 | 8704 | 112 | 0 | 151.18 M |
| 仅 stages=2 | 1.5007 | 91.58 | 16384 | 160 | 0 | 196.79 M |
| 仅 GROUP_M=1 | 1.4607 | 94.09 | 8704 | 144 | 0 | 184.61 M |
| 旧配置 128×256×32，8 warps，2 stages | 2.7150 | 50.62 | 24576 | 128 | 96 | 97.81 M |

## 复现命令

下面跑一个明确配置。其他配置用相同命令替换参数并指定独立的输出文件。完整矩阵和采集脚本见 [BW100 基线](../baselines/bw100-gfx936.md)。

```bash
cd experiments
python3 advanced/08_triton_gemm_mmac.py --tile 128 128 32 --isa e09-base.s > e09-base.json
hipprof --pmc --pmc-type 3 -o e09-base-pmc   python3 advanced/08_triton_gemm_mmac.py --tile 128 128 32 --profile-only
python3 parse_hipprof_pmc.py e09-base-pmc.csv --kernel gemm_kernel --expect-dispatches 6
```

[下载逐配置的 JSON、PMC CSV、摘要和 Triton ISA](/data/bw100-20261118.zip) · [Occupancy 与 Roofline](../04-scheduling/occupancy-and-roofline.md)

## 消融结果与归因边界

- **BK 32 → 16**：VGPR 144 → 112，静态 LDS 和 scratch 持平，等待计数下降约 29%，耗时增加约 6%。需要进一步按 LDS 指令、运行工作量和循环开销解释。
- **流水级数 1 → 2**：LDS 8704 → 16384 B，VGPR 144 → 160，耗时增加约 6%。要判断是否因资源消耗影响驻留，仍需该 code object 的驻留上限或动态 occupancy 证据。
- **GROUP_M 8 → 1**：TCC 命中率约 86.6% → 75.0%，TCC miss 约 343 万 → 640 万，耗时增加约 3.5%。该结果支持当前排布的缓存复用收益。TCC 计数的单位和聚合范围需与对应工具版本核对；更高的命中率只描述一部分访存行为。
- **旧配置**：独立采集确认该配置报告 `scr=96 B`。基准 Triton 已报告 `scr=0 B` 但仍慢于 rocBLAS。旧配置同时改了 tile、wave 数和流水级数，单凭两者差值无法估计 scratch 开销。跨 CU 聚合的等待计数也不能直接换算为单次 kernel 的墙钟等待时间。

::: details 查看 Triton 源码

<<< ../../experiments/advanced/08_triton_gemm_mmac.py

:::

::: details 查看完整基线采集脚本

<<< ../../experiments/run_bw100_baseline.sh

:::
