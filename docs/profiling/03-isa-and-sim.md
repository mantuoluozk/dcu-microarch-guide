# 2.3 ISA 反汇编分析与仿真工具（DCUSIM 待补充）

## 1. 如何导出与阅读设备端 AMDGCN ISA？

### HIP Kernel 导出方法

```bash
hipcc -O3 -std=c++17 --offload-arch=gfx936 -save-temps my_kernel.hip -o my_kernel
# 在生成目录查找 *-hip-amdgcn-*.s 汇编文件
```

### Triton Kernel 导出方法

```python
compiled = my_triton_kernel[grid](...)
with open("my_kernel_isa.s", "w") as f:
    f.write(compiled.asm["amdgcn"])
```

### 看懂一份 `.s` 汇编的 5 个关键检查点

1. **元数据区（`.amdhsa_kernel` / `.vgpr_count`）：** 检查 `vgpr_count`、`sgpr_count`、`group_segment_fixed_size`（LDS）以及 `private_segment_fixed_size`（是否发生 `Scratch` 溢出）。
2. **异步访存与等待（`global_load` / `s_waitcnt`）：** 看加载指令与 `s_waitcnt vmcnt(0)` 之间是否穿插了足够的独立计算指令。
3. **分支形态：** 检查是走标量跳转 `s_cbranch_scc`，还是谓词选择 `v_cndmask_b32`，或是 `s_and_saveexec_b64` 分支发散。
4. **矩阵指令生成：** 检查是否正确生成了 `v_mmac_*` / `v_mfma_*` 矩阵乘加指令。
5. **跨 Lane 通信：** 检查规约是否生成了 `v_perm` / `ds_bpermute` / `_dpp` 指令，减少 `s_barrier` 次数。

---

## 2. 微架构仿真工具：DCUSIM（待补充）

> 待获取到研发侧提供的 DCUSIM 工具版本后，在此补充：
> - 仿真环境配置与 Trace 输入准备
> - 指令发射流水线、Scoreboard 停顿与资源冲突的微架构级可视化分析方法
> - 实测 `hipprof` PMC 与 DCUSIM 仿真结果的交叉验证案例
