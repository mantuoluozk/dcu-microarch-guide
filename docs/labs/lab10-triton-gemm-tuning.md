# 实验 10：Triton GEMM 调优实战——消除 Spill 与 LDS Wait（待补充）

> **前置线索：** 在 [实验 9：Triton GEMM vs rocBLAS](lab07-triton-gemm-vs-rocblas.md) 中，`hipprof` 已经帮我们抓出了 Triton GEMM（`97 TFLOPS`）跑输 `rocBLAS`（`256.80 TFLOPS`）的三大元凶：`Scratch = 96B`、`SQ_WAIT_INST_LDS = 9900 万周期` 以及 8 倍 `VMEM_RD` 指令。

---

## 本实验调优目标与待测方案

1. **缩小 K 维分块（`BK = 16`）：** 参考 `rocBLAS` 的 `MT256x256x16`，降低单步 LDS 与 VGPR 压力，彻底把 `Scratch` 降回 `0 B`。
2. **开启后端矩阵指令调优参数：** 测试 `kpack=2`、`matrix_instr_nonkdim=16`、`waves_per_eu` 与 `triton.autotune` 搜索空间。
3. **PMC 复测闭环：** 用 `hipprof --pmc --pmc-type 3` 复测调优后版本，记录 `SQ_WAIT_INST_LDS`、Bank Conflict 与 TFLOPS 的真实提升幅度。

*(待在 `zk-migraphx` 上完成参数扫描与 PMC 复测后补充完整源码与实测结果。)*
