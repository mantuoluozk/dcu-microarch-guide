# 实验索引

先核对[环境与工具链](lab-setup.md)，再按研究问题挑选实验。实验 ID 在页面、侧栏和基线包中一致，原有链接继续可用。

| 实验 | 研究问题 | 结果状态 |
|---|---|---|
| [E01 · 合并访存与向量化](lab01-memory-coalescing.md) | 相同地址覆盖下，stride 或向量宽度如何影响拷贝带宽？ | BW100 已复测，有原始数据 |
| [E02 · LDS Bank Conflict](lab02-lds-bank-conflict.md) | Padding 和 XOR Swizzle 如何改变 LDS 冲突？ | BW100 历史记录 |
| [E03 · SIMT 分支发散](lab03-branch-divergence.md) | 分支掩码、发散和谓词化怎样反映在 ISA 及耗时上？ | BW100 历史记录 |
| [E04 · Occupancy 与 ILP](lab04-occupancy-ilp.md) | 驻留 wave 数与独立访存链怎样影响延迟隐藏？ | BW100 历史记录 |
| [E05 · Wave Shuffle 规约](lab05-wave-shuffle-reduction.md) | 使用 wave 内交换替换部分 LDS 规约能节约多少时间？ | BW100 历史记录 |
| [E06 · 存储延迟阶梯](lab08-memory-latency-ladder.md) | 标量与向量依赖加载在哪些工作集范围内出现延迟转折？ | BW100 已复测，有 ISA 和 PMC |
| [E07 · MMAC 与 VALU 指令吞吐](lab09-mmac-builtin-and-valu.md) | 寄存器常驻操作数下，计算指令能达到多少吞吐？ | BW100 已复测，有 ISA |
| [E08 · 融合 RMSNorm](lab06-triton-lightop-rmsnorm.md) | 融合能让所测 FP16 RMSNorm 改善多少？ | BW100 历史记录 |
| [E09 · GEMM 计时与 PMC](lab07-triton-gemm-vs-rocblas.md) | Tile、流水级数与分组如何影响耗时和资源？ | BW100 已复测，按配置隔离采集 |
| [E10 · GEMM 调优](lab10-triton-gemm-tuning.md) | E09 的结果能否迁移到推理 shape？ | 计划中 |

复测方法、观测口径和原始文件归档见 [BW100 基线档案](../baselines/bw100-gfx936.md)。
