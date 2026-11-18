# 1.5 微架构自动化分析与 Roofline 建模（待补充）

> 本节用于沉淀面向不同 DCU 型号的**自动化微架构档案生成**与 **Roofline 自动诊断工具 / Agent Skill**。

---

## 规划内容

1. **一键微架构体检脚本：**
   - [BW100 基线](../baselines/bw100-gfx936.md) 已串联向量拷贝、依赖加载延迟、VALU/MMAC 吞吐与 GEMM。下一步把同一份原始数据自动生成设备档案。
2. **自动化 Roofline 诊断：**
   - 输入运算量、耗时及实测 HBM 字节数，估算 Roofline 位置并给出排查方向。只有逻辑字节时使用经验参考，并注明口径。见 [Roofline 与 BW100 对照](../04-scheduling/occupancy-and-roofline.md#_6-5-roofline-模型与-bw100-对照)。
3. **AI Skill 沉淀：**
   - 将分析流程封装为可复用的 Skill，支持在换代到下一代 DCU 时一键对比两代硬件的微架构差异。

*(待后续实验工具完善后补充具体脚本与实测案例。)*
