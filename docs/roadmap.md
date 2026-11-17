# 后续实验路线图与更新日志

本页面用于记录后续在海光 BW100（`zk-migraphx` 容器）及下一代 DCU 上持续开展的新实验。

---

## 已完成实验归档

- [x] **实验导览 & 硬件/软件栈画像**：[lab-setup.md](labs/lab-setup.md)
- [x] **实验 1：全局内存合并访存与 `float4` 向量化（含 `hipprof` PMC）**：[lab01-memory-coalescing.md](labs/lab01-memory-coalescing.md)
- [x] **实验 2：LDS Bank Conflict 与 Padding / XOR Swizzle（含 `hipprof` PMC）**：[lab02-lds-bank-conflict.md](labs/lab02-lds-bank-conflict.md)
- [x] **实验 3：SIMT 分支发散与 Predicated If-Conversion**：[lab03-branch-divergence.md](labs/lab03-branch-divergence.md)
- [x] **实验 4：Occupancy 悬崖与 TLP / ILP 延迟隐藏**：[lab04-occupancy-ilp.md](labs/lab04-occupancy-ilp.md)
- [x] **实验 5：Wavefront Shuffle 规约 vs LDS 规约**：[lab05-wave-shuffle-reduction.md](labs/lab05-wave-shuffle-reduction.md)
- [x] **实验 6：PyTorch vs `lightop` vs Triton Fused RMSNorm + ISA 剖析**：[lab06-triton-lightop-rmsnorm.md](labs/lab06-triton-lightop-rmsnorm.md)
- [x] **实验 7：Triton GEMM vs `rocBLAS` 与 `hipprof` PMC 根因破案**：[lab07-triton-gemm-vs-rocblas.md](labs/lab07-triton-gemm-vs-rocblas.md)
- [x] **实验 8：BW100 存储延迟阶梯（L1 `16KB` / L2 / HBM Pointer-Chasing）**：[lab08-memory-latency-ladder.md](labs/lab08-memory-latency-ladder.md)
- [x] **实验 9：HCU Builtin MMAC 峰值（`333.49 TFLOPS`）与 VALU 对比**：[lab09-mmac-builtin-and-valu.md](labs/lab09-mmac-builtin-and-valu.md)

---

## 待做实验计划（Next Up）

- [ ] **实验 8（Triton GEMM 调优冲击 200+ TFLOPS）：**
  - 参考 `rocBLAS` 的 `MT256x256x16`，测试 `BK=16`、`kpack=2`、`waves_per_eu`、`matrix_instr_nonkdim=16` 以及 `triton.autotune`。
  - 用 `hipprof --pmc` 验证 `Scratch=0B`（消除 Spill）与 `SQ_WAIT_INST_LDS` 的下降幅度。
- [ ] **实验 9（`lightop` 大模型专项算子拆解）：**
  - 对比 `lightop.gemm_w8a8_asm`、`lightop.m_grouped_bf16_gemm_nt_masked`（MoE Grouped GEMM）、`lightop.fuse_rmsnorm_rope_quant_qkv` 与常规实现的耗时与 `torch.profiler` 轨迹。
- [ ] **实验 10（LSTM Persistent MMAC Kernel ISA 深度诊断）：**
  - 导出 `adaptive_lstm_hip.cu` 的 `.s` 汇编，检查 `.vgpr_count`、MMAC 指令利用率、是否存在 `SCRATCH` 溢出，并量化对比 Persistent vs 非 Persistent 的真实收益边界。
