import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

export default defineConfig({
  base: '/dcu-microarch-guide/',
  lang: 'zh-CN',
  title: 'DCU 微架构指南',
  description: '从 DCU 微架构原理、Profiling 工具链到 HIP / Triton / lightop 实战与大模型推理优化',

  markdown: {
    config: (md) => {
      md.use(mathjax3)
    }
  },

  themeConfig: {
    outline: {
      level: [2, 3],
      label: '本页目录'
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
          modal: {
            noResultsText: '未找到相关内容',
            resetButtonTitle: '清除查询',
            footer: { selectText: '选择', navigateText: '切换' }
          }
        }
      }
    },

    docFooter: {
      prev: '上一章',
      next: '下一章'
    },

    nav: [
      { text: '首页', link: '/' },
      { text: '专题方案', link: '/00-plan/research-plan' },
      { text: '微架构原理', link: '/01-overview/software-stack' },
      { text: 'Profiling 工具', link: '/profiling/01-toolchain-overview' },
      { text: '实战实验', link: '/labs/lab-setup' },
      { text: '大模型优化', link: '/llm-opt/01-framework-profiling' }
    ],

    sidebar: [
      {
        text: '专题总览',
        collapsed: false,
        items: [
          { text: '首页导览', link: '/' },
          { text: '专题实施方案', link: '/00-plan/research-plan' }
        ]
      },
      {
        text: '第一部分：DCU 微架构原理',
        collapsed: false,
        items: [
          { text: '1.1 整体架构与软件栈', link: '/01-overview/software-stack' },
          { text: '1.2 计算单元、SIMT 与 MMAC', link: '/02-compute/cu-and-simt' },
          { text: '1.3 存储体系与 s_waitcnt', link: '/03-memory/memory-hierarchy' },
          { text: '1.4 Occupancy 与 Roofline', link: '/04-scheduling/occupancy-and-roofline' },
          { text: '1.5 自动化分析（待补充）', link: '/05-modeling/auto-analysis' }
        ]
      },
      {
        text: '第二部分：Profiling 工具链',
        collapsed: false,
        items: [
          { text: '2.1 工具链与软件栈探测', link: '/profiling/01-toolchain-overview' },
          { text: '2.2 PMC 采集与自研解析', link: '/profiling/02-hipprof-and-parser' },
          { text: '2.3 ISA 与仿真（待补充）', link: '/profiling/03-isa-and-sim' }
        ]
      },
      {
        text: '第三部分：底层与算子实战',
        collapsed: false,
        items: [
          { text: '实验导览与环境准备', link: '/labs/lab-setup' },
          {
            text: 'HIP 底层微基准',
            collapsed: false,
            items: [
              { text: '实验 1：合并访存与 float4', link: '/labs/lab01-memory-coalescing' },
              { text: '实验 2：LDS Bank Conflict', link: '/labs/lab02-lds-bank-conflict' },
              { text: '实验 3：SIMT 分支发散', link: '/labs/lab03-branch-divergence' },
              { text: '实验 4：Occupancy 与 ILP', link: '/labs/lab04-occupancy-ilp' },
              { text: '实验 5：Wave Shuffle 规约', link: '/labs/lab05-wave-shuffle-reduction' },
              { text: '实验 6：L1 / L2 / HBM 延迟', link: '/labs/lab08-memory-latency-ladder' },
              { text: '实验 7：MMAC Builtin 峰值', link: '/labs/lab09-mmac-builtin-and-valu' }
            ]
          },
          {
            text: 'Triton / lightop 算子',
            collapsed: false,
            items: [
              { text: '实验 8：融合 RMSNorm', link: '/labs/lab06-triton-lightop-rmsnorm' },
              { text: '实验 9：GEMM vs rocBLAS', link: '/labs/lab07-triton-gemm-vs-rocblas' },
              { text: '实验 10：GEMM 调优（待补充）', link: '/labs/lab10-triton-gemm-tuning' }
            ]
          }
        ]
      },
      {
        text: '第四部分：大模型推理优化',
        collapsed: false,
        items: [
          { text: '4.1 推理热点分析（待补充）', link: '/llm-opt/01-framework-profiling' },
          { text: '4.2 GEMM 与 MoE（待补充）', link: '/llm-opt/02-gemm-and-moe' },
          { text: '4.3 Attention 与 KV（待补充）', link: '/llm-opt/03-attention-kvcache' },
          { text: '4.4 LSTM Persistent 拆解', link: '/cases/lstm-persistent-kernel' },
          { text: '4.5 优化六步法与反馈', link: '/cases/llm-operators-playbook' }
        ]
      },
      {
        text: '路线图',
        collapsed: false,
        items: [
          { text: '后续实验路线图与更新日志', link: '/roadmap' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/mantuoluozk/dcu-microarch-guide' }
    ]
  }
})
