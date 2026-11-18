# E10 · Triton GEMM 调优（待验证）

**研究问题**　E09 在方阵上的参数收益能否迁移到实际推理 shape？

[E09](lab07-triton-gemm-vs-rocblas.md) 已完成 BW100 上 `4096³` 基线与逐项消融。现阶段的下一步是从模型 trace 选取一个 prefill 或 decode shape，记录模型、框架、批量和实际派发的 kernel，在固定输入和精度下比较相同参数。首选加入会改变矩阵尾块比例或 grid 大小的工作负载。

验收记录需包含正确性、无 profiler 计时、同配置 ISA 和 PMC，并注明相对同版本 rocBLAS 的收益。`kpack`、`matrix_instr_nonkdim`、`waves_per_eu` 的支持范围由安装的 Triton 后端校验后，再纳入扫描。
