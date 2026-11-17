# 1.5 微架构自动化分析与 Roofline 建模（待补充）

> 本节用于沉淀面向不同 DCU 型号的**自动化微架构档案生成**与 **Roofline 自动诊断工具 / Agent Skill**。

---

## 规划内容

1. **一键微架构体检脚本：**
   - 自动串联运行第 3 章的底层微基准（HBM 带宽、L1/L2/HBM 延迟阶梯、VALU/MMAC 峰值），自动输出当前 DCU 型号的硬件档案 JSON/Markdown。
2. **自动化 Roofline 诊断：**
   - 输入算子的 `FLOPs`、理论访存量 `Bytes` 与实测耗时，自动对比本机实测拐点（如 BW100 FP16 约 `188.1 FLOPs/Byte`），给出“Memory-bound / Compute-bound / 调度受限”判定及下一步排查建议。
3. **AI Skill 沉淀：**
   - 将分析流程封装为可复用的 Skill，支持在换代到下一代 DCU 时一键对比两代硬件的微架构差异。

*(待后续实验工具完善后补充具体脚本与实测案例。)*
