# 4.3 Attention 与 KV Cache 访存优化（待补充）

> 本节聚焦长上下文与高并发推理中的核心访存瓶颈：围绕 **FlashAttention、PagedAttention 与 KV Cache 布局**，从全局内存合并访存、向量化加载、LDS 数据复用与 `s_waitcnt` 流水重叠角度进行优化。

---

## 计划研究内容

1. **Prefill FlashAttention：**
   - QKV 分块加载与 Online Softmax 寄存器规约（结合第 3 章 Wave Shuffle 减少 LDS/Barrier 开销）。
   - 双缓冲与 `s_waitcnt` 调度重叠。
2. **Decode PagedAttention 与 KV Cache 访存：**
   - 分析非连续 Block Table 寻址导致的非合并访存（Uncoalesced）与 L2 Cache 命中率下降。
   - 评估不同 KV Cache 布局（Page Size、量化精度）对 HBM 带宽利用率的影响。
3. **稀疏注意力（Sparse Attention）：**
   - 结合稀疏索引特点，优化非规则访存与 Wavefront 负载不均衡。

*(待后续开展 Attention 与 KV Cache 专项实验后补充实现与评测数据。)*
