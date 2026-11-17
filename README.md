# dcu-microarch-guide

海光 DCU 微架构与算子优化实战指南（涵盖 HIP 微基准、Triton DSL、`lightop` 专项算子、`hipprof` 硬件 PMC 计数器分析与 BW100 实测数据）。

- **在线文档站：** https://mantuoluozk.github.io/dcu-microarch-guide/
- **仓库地址：** https://github.com/mantuoluozk/dcu-microarch-guide

## 目录结构

```text
├── docs/                       # MkDocs 文档站源码
│   ├── architecture/           # 微架构原理与执行机制
│   ├── tutorials/              # BW100 实战教程与实验手册
│   └── reports/                # BW100 实测数据报告
├── experiments/                # 可复现实验代码 (HIP / Triton / lightop / hipprof)
│   ├── 01_device_info_and_isa.hip
│   ├── 02_coalescing_and_vec.hip
│   ├── 03_lds_bank_conflict.hip
│   ├── 04_branch_divergence.hip
│   ├── 05_occupancy_and_ilp.hip
│   ├── 06_shuffle_vs_lds.hip
│   ├── parse_hipprof_pmc.py
│   └── advanced/
│       ├── 07_triton_fused_rmsnorm.py
│       ├── 08_triton_gemm_mmac.py
│       ├── 09_hygon_ops_and_stack.py
│       └── 10_lightop_vs_triton_pytorch.py
├── mkdocs.yml                  # MkDocs Material 配置
└── .github/workflows/deploy.yml
```

## 快速运行实验（在 DCU 容器内）

```bash
cd experiments
make run      # 编译并运行实验 1~6
make isa      # 导出设备端 AMDGCN ISA 汇编

cd advanced
python3 09_hygon_ops_and_stack.py
python3 07_triton_fused_rmsnorm.py
python3 08_triton_gemm_mmac.py
python3 10_lightop_vs_triton_pytorch.py
```
