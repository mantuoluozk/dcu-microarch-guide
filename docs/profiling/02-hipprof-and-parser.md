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

## 2. 自研工具：`parse_hipprof_pmc.py`

`hipprof --pmc --pmc-type 3` 导出的 CSV 包含数百列（L2 分 32 通道、TA 分 16 通道）。为了快速横向对比多个 Kernel，我们编写了自动汇总解析脚本：

```python title="experiments/parse_hipprof_pmc.py"
import csv
import sys
from collections import defaultdict


def sum_prefix(row, prefix):
    total = 0
    for k, v in row.items():
        if k.startswith(prefix + "["):
            try:
                total += int(v)
            except ValueError:
                pass
    return total


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 parse_hipprof_pmc.py <pmc.csv>")

    path = sys.argv[1]
    grouped = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[row.get("KernelName", "unknown")].append(row)

    print(f"=== PMC Summary: {path} ({len(grouped)} unique kernels) ===")
    for kname, rows in grouped.items():
        sample = rows[-5:] if len(rows) > 5 else rows
        n = len(sample)

        def avg_int(col):
            return sum(int(r.get(col, 0) or 0) for r in sample) / n

        def avg_pref(pref):
            return sum(sum_prefix(r, pref) for r in sample) / n

        dur_us = sum((int(r["EndNs"]) - int(r["BeginNs"])) / 1e3 for r in sample) / n
        tcc_hit = avg_pref("TCC_HIT")
        tcc_miss = avg_pref("TCC_MISS")
        l2_hit_rate = (tcc_hit / (tcc_hit + tcc_miss) * 100.0) if (tcc_hit + tcc_miss) > 0 else 0.0

        r0 = sample[0]
        short_name = kname if len(kname) <= 68 else kname[:65] + "..."
        print(f"\nKernel: {short_name} (calls={len(rows)})")
        print(f"  Grid={r0.get('grd')} | Block={r0.get('wgr')} | LDS={r0.get('lds')}B | "
              f"Scratch={r0.get('scr')}B | VGPR={r0.get('arch_vgpr')} | SGPR={r0.get('sgpr')}")
        print(f"  Avg Kernel Time      : {dur_us:10.2f} us | GRBM_GUI_ACTIVE = {avg_int('GRBM_GUI_ACTIVE'):,.0f}")
        print(f"  Insts (VALU/LDS/Rd/Wr): {avg_int('SQ_INSTS_VALU'):,.0f} / "
              f"{avg_int('SQ_INSTS_LDS'):,.0f} / "
              f"{avg_int('SQ_INSTS_VMEM_RD'):,.0f} / "
              f"{avg_int('SQ_INSTS_VMEM_WR'):,.0f}")
        print(f"  SQ_LDS_BANK_CONFLICT : {avg_int('SQ_LDS_BANK_CONFLICT'):,.0f} | "
              f"SQ_WAIT_INST_LDS = {avg_int('SQ_WAIT_INST_LDS'):,.0f}")
        print(f"  L2 (TCC) Hit / Miss  : {tcc_hit:,.0f} / {tcc_miss:,.0f} (Hit Rate = {l2_hit_rate:.2f}%)")
        print(f"  TCP_TA_DATA_STALL    : {avg_pref('TCP_TCP_TA_DATA_STALL_CYCLES'):,.0f}")


if __name__ == "__main__":
    main()
```

### 扩展工具规划（待补充）
- [ ] **自动化多版本 Diff 工具**：直接对比优化前后两份 `hipprof` CSV，高亮耗时、Spill、Bank Conflict 与 Stall 变化百分比。
- [ ] **`torch.profiler` 热点自动聚合脚本**：针对 `vllm` / `SGLang` 推理 Trace 自动按算子类别（GEMM / Attention / MoE / Norm / 通信）归类占比。
