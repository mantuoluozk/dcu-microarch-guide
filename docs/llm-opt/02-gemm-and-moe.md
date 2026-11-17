# 4.2 GEMM 与 MoE 算子微架构优化（待补充）

> 本节聚焦大模型推理中最核心的矩阵计算类算子：针对特定 Shape、量化精度（FP16 / BF16 / W8A8 / FP8）及 MoE 稀疏路由特点，从 **MMAC、LDS Swizzle、Register Blocking、计算访存流水** 角度做系统性优化。

---

## 计划研究内容

1. **Shape-aware GEMM 与 `lightop` 对比：**
   - 拆解 `lightop.gemm_w8a8_asm` 等内置汇编/模板算子在 Decode 小 $M$ 场景下的表现。
   - 针对瘦高/扁平矩阵优化 Tile 划分、Split-K 与 Persistent GEMM 策略。
2. **MoE Grouped GEMM 优化：**
   - 分析 `lightop.m_grouped_bf16_gemm_nt_masked` 等算子在 Token 分组、专家负载不均与小矩阵场景下的 MMAC 利用率瓶颈。
   - 区分计算瓶颈与 EP 通信瓶颈，探索路由重排与算子融合。
3. **端到端框架集成验证：** 在 vLLM / SGLang 中验证单算子优化对 TTFT、TPOT 与吞吐量的实际收益。

*(待后续开展专项对比与调优实验后补充源码、PMC 数据与收益对比。)*
