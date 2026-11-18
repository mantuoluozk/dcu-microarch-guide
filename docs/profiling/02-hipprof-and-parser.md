# 2.2 硬件 PMC 计数器采集与自研解析工具

## 1. `hipprof` 常用采集模式

在海光 DTK-26.04 中，`/opt/dtk/bin/hipprof` 支持 Trace、硬件性能计数器（PMC）以及代码对象寄存器分析：

```bash
# 1. 导出硬件 PMC 计数器 CSV（最常用）
hipprof --pmc --pmc-type 3 -o prof_out/exp ./my_app

# 2. 按指定 Kernel 名称过滤抓取
hipprof --pmc --pmc-type 3 --kernel-name gemm_kernel -o prof_out/gemm python3 run.py

# 3. 导出 HIP API 与 Kernel 时间线（Chrome/Perfetto Trace）
hipprof --hip-trace --stats -o prof_out/trace ./my_app

# 4. 分析 ELF/Code Object 的寄存器压力
hipprof --codeobj-analyze my_kernel.hsaco
```

### 核心硬件 PMC 指标速查表

| 指标列名 | 物理含义 | 异常阈值 / 诊断方向 |
|---|---|---|
| `scr` | Scratch 栈溢出字节数 | **`> 0` 说明发生寄存器 Spill**，严重拖慢性能 |
| `arch_vgpr` / `sgpr` | 向量/标量寄存器占用数 | 结合分配粒度判断是否触发 Occupancy 悬崖 |
| `SQ_INSTS_VMEM_RD / WR` | 全局访存读/写指令数 | 检查是否成功向量化（如 `float4` 降为 `1/4`） |
| `SQ_LDS_BANK_CONFLICT` | LDS Bank 冲突次数 | 非 0 且数值巨大时，需改用 Padding 或 XOR Swizzle |
| `SQ_WAIT_INST_LDS` | 等待 LDS 返回的停顿周期 | 偏高通常由 LDS 冲突或过多 LDS 读写引起 |
| `TCC_HIT[0..31]` / `TCC_MISS[0..31]` | L2 Cache 32 通道命中与未命中 | 评估工作集复用率与 L2 Swizzle 效果 |
| `TCP_TCP_TA_DATA_STALL_CYCLES` | L1 / TA 数据停顿周期 | 暴涨通常意味着非合并访存或带宽打满 |

---

## 2. 按配置采集并解析 PMC

先把参数或实现变体拆成独立采集，再用解析器读取每份 CSV。解析器按 kernel 名称和资源字段分组，汇总每组最后五条记录，缺失字段显示为 `N/A`。同名、同资源的变体仍需分开采集。可用预期 dispatch 数量检查某份 CSV 是否包含了指定的采集样本。

```bash
python3 experiments/parse_hipprof_pmc.py e09-base-pmc.csv --kernel gemm_kernel --expect-dispatches 6
```

归因时确认计数器的设备、单位和聚合范围。跨 CU 或通道求和的计数应按相同工作量比较。对于 GEMM，把无 profiler 计时、静态 ISA 和 PMC 保持为同一配置的三份证据。见 [E09](../labs/lab07-triton-gemm-vs-rocblas.md)。

::: details 查看解析器源码

<<< ../../experiments/parse_hipprof_pmc.py

:::

### 扩展工具规划（待补充）
- [ ] **自动化多版本 Diff 工具**：直接对比优化前后两份 `hipprof` CSV，高亮耗时、Spill、Bank Conflict 与 Stall 变化百分比。
- [ ] **`torch.profiler` 热点自动聚合脚本**：针对 `vllm` / `SGLang` 推理 Trace 自动按算子类别（GEMM / Attention / MoE / Norm / 通信）归类占比。
