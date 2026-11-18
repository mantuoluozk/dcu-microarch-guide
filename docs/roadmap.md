# 研究路线图与更新日志

## 2026-11-18 · BW100 基线复测

- [E01](labs/lab01-memory-coalescing.md) 修正跨步地址映射，单独对比向量化与合并度，增加全缓冲区校验。
- [E06](labs/lab08-memory-latency-ladder.md) 测完完整随机环，拆出标量和向量加载路径，增加多 seed、间距和独立 PMC。
- [E07](labs/lab09-mmac-builtin-and-valu.md) 复测 MMAC 与 VALU 指令吞吐，并保留设备 ISA。
- [E09](labs/lab07-triton-gemm-vs-rocblas.md) 以独立采集复测 rocBLAS 和各 Triton 配置，加入 BK、流水级数和分组顺序消融。
- 环境、源码校验和、原始数据及一键命令见 [BW100 / gfx936 基线档案](baselines/bw100-gfx936.md)。历史实验见 [实验索引](labs/index.md)。

## 2026-11-18 · 追加微基准

- [E02](labs/lab02-lds-bank-conflict.md) 改用只读 LDS 循环，扫描 1–32 路布局并比较 Padding 与 XOR。
- [E04](labs/lab04-occupancy-ilp.md) 交叉测量四档驻留上限与四档独立加载链数量，核对 16 组配置的动态 LDS。
- [E06](labs/lab08-memory-latency-ladder.md) 增加共用工作集横轴的 TCC 命中率散点图。

## 后续研究

- **现有实验**基于 E04 的依赖链对照，进一步用顺序访问测量并发加载数量与流式带宽饱和的关系。
- **下一代 DCU 对照**复用 E01、E06、E07 和 E09，附实际 DTK 与设备环境，校验计数器的含义及可比性。
- **微架构自动化分析**根据基线数据生成设备档案和 Roofline 诊断。见 [自动化分析计划](05-modeling/auto-analysis.md)。
- **推理专项算子**安排在微基准补充之后。从 [E10 · Triton GEMM 调优](labs/lab10-triton-gemm-tuning.md) 开始，使用真实 shape 验证优化收益，再研究 MoE Grouped GEMM、Attention / KV 访存和融合算子。
