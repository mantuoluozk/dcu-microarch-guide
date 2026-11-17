# 4.1 vLLM / SGLang 推理热点分析（待补充）

> 本节聚焦**先分析、后优化**：在真实大模型推理框架（`vLLM 0.11.0` / `SGLang`）中定位 Prefill 与 Decode 阶段的真实热点算子，为后续微架构级优化提供精确靶点。

---

## 计划开展的分析流程

1. **基准模型选择：** 选取代表性的 Dense 模型（如 Qwen/Llama 系列）、MoE 模型与长上下文场景。
2. **分阶段 Profiling 拆解：**
   - **Prefill 阶段：** 重点关注大 Shape GEMM、FlashAttention、计算密集型算子占比与 TTFT（首字延迟）。
   - **Decode 阶段：** 重点关注小 Batch GEMV / MoE Grouped GEMM、PagedAttention KV Cache 访存带宽利用率与 TPOT（单 Token 生成延迟）。
3. **热点 Kernel 清单输出：** 使用 `torch.profiler` 与 `hipprof` 导出 Top-10 耗时 Kernel，标注每个热点算子的 Roofline 区间（Compute-bound vs Memory-bound）及所用后端（`rocBLAS` / `lightop` / `Triton` / 框架原生 HIP）。

*(待在 `zk-migraphx` 上跑通真实模型推理 Profiling 后补充实测数据与热点分布图。)*
