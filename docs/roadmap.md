# 研究路线图与更新日志

## 2026-11-18 · BW100 基线复测

- [E01](labs/lab01-memory-coalescing.md) 修正跨步地址映射，单独对比向量化与合并度，增加全缓冲区校验。
- [E06](labs/lab08-memory-latency-ladder.md) 测完完整随机环，拆出标量和向量加载路径，增加多 seed、间距和独立 PMC。
- [E07](labs/lab09-mmac-builtin-and-valu.md) 复测 MMAC 与 VALU 指令吞吐，并保留设备 ISA。
- [E09](labs/lab07-triton-gemm-vs-rocblas.md) 以独立采集复测 rocBLAS 和各 Triton 配置，加入 BK、流水级数和分组顺序消融。
- 环境、源码校验和、原始数据及一键命令见 [BW100 / gfx936 基线档案](baselines/bw100-gfx936.md)。历史实验见 [实验索引](labs/index.md)。

## 后续研究

- [E10 · Triton GEMM 调优](labs/lab10-triton-gemm-tuning.md) 从推理 trace 选择真实 shape，检验 E09 的参数收益是否迁移。
- **下一代 DCU 对照**复用 E01、E06、E07 和 E09，附实际 DTK 与设备环境，校验计数器的含义及可比性。
- **推理专项算子**在真实调用链验证 MoE Grouped GEMM、Attention / KV 访存和融合算子。LSTM Persistent 案例可作为对照设计参考。见 [大模型算子案例](cases/llm-operators-playbook.md)。
