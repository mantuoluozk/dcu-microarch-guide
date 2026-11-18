# dcu-microarch-guide

海光 DCU 微架构与算子优化实战指南（涵盖 HIP 微基准、Triton DSL、`lightop` 专项算子、`hipprof` 硬件 PMC 计数器分析与 BW100 实测数据）。

- **在线文档站：** https://mantuoluozk.github.io/dcu-microarch-guide/
- **仓库地址：** https://github.com/mantuoluozk/dcu-microarch-guide

## BW100 基线复跑

在有 BW100、DTK、PyTorch 和 Triton 的 Linux 环境中，从仓库根目录执行：

```bash
bash experiments/run_bw100_baseline.sh gfx936 ./bw100-repeat
```

- [实验索引](docs/labs/index.md) 包括实验 ID、研究问题和复测状态。
- [BW100 基线档案](docs/baselines/bw100-gfx936.md) 包括实测口径、原始数据和跨代比较规则。

`docs/` 为 VitePress 站点，`experiments/` 为 HIP/Triton 代码与采集工具。运行 `npm install` 和 `npm run docs:build` 可构建静态站点。
