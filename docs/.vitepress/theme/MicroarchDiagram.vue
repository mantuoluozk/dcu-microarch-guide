<script setup lang="ts">
defineProps<{ kind: 'mapping' | 'lanes' | 'compute' | 'memory' | 'latency' }>()
</script>

<template>
  <figure v-if="kind === 'mapping'" class="microarch-figure">
    <div class="mapping">
      <section class="panel">
        <strong>软件划分任务</strong>
        <p>Grid · 4 个 Block，每个 256 个线程</p>
        <ol class="blocks" aria-label="Grid 的四个 Block">
          <li>Block 0</li><li>Block 1</li><li>Block 2</li><li>Block 3</li>
        </ol>
        <p>以 Block 3 为例，每 64 个线程组成一个 wave</p>
        <ol class="blocks" aria-label="Block 3 中的四个 wave">
          <li>Wave 0</li><li>Wave 1</li><li>Wave 2</li><li>Wave 3</li>
        </ol>
      </section>
      <span class="handoff">按 Block 分发</span>
      <section class="panel hardware">
        <strong>硬件分配资源</strong>
        <p>任务分发逻辑按可用资源将 Block 分配到 CU。</p>
        <p class="boxed"><strong>示例 CU</strong><br>Block 3 的四个 wave 在同一 CU 上执行。</p>
        <p><strong class="residency-emphasis">同一个 CU 可以驻留多个 Block。</strong> 能同时驻留多少，取决于寄存器、LDS 等资源。</p>
      </section>
    </div>
    <figcaption>HIP 编程模型，采用 BW100 的 Wave64 举例。CU 编号和 Block 的实际执行顺序由设备决定。</figcaption>
  </figure>

  <figure v-else-if="kind === 'lanes'" class="microarch-figure">
    <div class="panel">
      <strong>Block 3 · Wave 3</strong>
      <p>全局索引 960–1023，合法索引截至 999。</p>
      <div class="lanes" aria-hidden="true">
        <span v-for="lane in 64" :key="lane" :class="{ active: lane <= 40 }">{{ lane - 1 }}</span>
      </div>
      <p><strong>Lane 0–39</strong> 处理有效元素 · <strong>Lane 40–63</strong> 不读写数组</p>
    </div>
    <figcaption>边界检查示意。只描述哪些 lane 有有效工作，实际掩码指令以编译输出为准。</figcaption>
  </figure>

  <figure v-else-if="kind === 'compute'" class="microarch-figure">
    <div class="panel">
      <strong>CU 内的 wave 调度</strong>
      <p>驻留 wave 提供待执行工作。调度器按依赖和资源选择可发射的指令。</p>
      <ul class="blocks">
        <li><strong>SALU</strong><br>统一数据与标量控制</li>
        <li><strong>VALU</strong><br>按有效 lane 运算</li>
        <li><strong>VMEM</strong><br>发出向量访存</li>
        <li><strong>MMAC 指令路径</strong><br>协作执行矩阵乘加</li>
      </ul>
    </div>
    <figcaption>CU 功能示意。各代设备的执行资源组织与共享方式以对应架构资料为准。</figcaption>
  </figure>

  <figure v-else-if="kind === 'memory'" class="microarch-figure">
    <div class="panel">
      <strong>向量加法 · 逐元素访问</strong>
      <ol class="steps">
        <li>全局内存读取 A、B</li>
        <li>结果回到 VGPR</li>
        <li>VALU 完成加法</li>
        <li>写回全局内存 C</li>
      </ol>
    </div>
    <div class="panel">
      <strong>分块运算 · 同一个 Block 复用数据</strong>
      <ol class="steps">
        <li>从全局内存读入</li>
        <li>线程将数据写入 LDS</li>
        <li>Block 同步后协作读取</li>
        <li>消费数据并计算</li>
      </ol>
    </div>
    <figcaption>功能路径示意。全局访问可能由缓存服务，具体路径取决于指令和访问模式。</figcaption>
  </figure>

  <figure v-else class="microarch-figure">
    <div class="panel">
      <strong>TLP · 调度其他就绪 wave</strong>
      <ol class="steps">
        <li>Wave A 发出加载</li>
        <li>Wave A 等待，Wave B 发射指令</li>
        <li>数据就绪，Wave A 可继续</li>
      </ol>
    </div>
    <div class="panel">
      <strong>ILP · 同一 wave 有独立工作</strong>
      <ol class="steps">
        <li>Wave A 发出加载</li>
        <li>Wave A 发射不依赖结果的指令</li>
        <li>确认依赖满足后使用加载结果</li>
      </ol>
    </div>
    <figcaption>发射顺序示意，不使用硬件时钟刻度。是否能继续发射还取决于资源和指令依赖。</figcaption>
  </figure>
</template>

<style scoped>
.microarch-figure {
  margin: 1.25rem 0;
  display: grid;
  gap: .55rem;
}

.mapping {
  display: grid;
  grid-template-columns: minmax(0, 1.3fr) auto minmax(0, 1fr);
  align-items: stretch;
  gap: .5rem;
}

.panel {
  padding: .8rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: .7rem;
  background: var(--vp-c-bg-soft);
  font-size: .875rem;
  line-height: 1.55;
}

.panel p {
  margin: .45rem 0;
}

.panel strong, .handoff {
  color: var(--vp-c-brand-1);
}

.panel .residency-emphasis {
  color: var(--vp-c-danger-1);
}

.handoff {
  align-self: center;
  text-align: center;
  font-size: .78rem;
}

.blocks, .steps {
  display: grid;
  gap: .35rem;
  padding: 0;
  margin: .45rem 0 0;
  list-style: none;
}

.blocks {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.blocks li, .steps li, .boxed {
  margin: 0;
  padding: .45rem .55rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: .4rem;
  background: var(--vp-c-bg);
}

.steps {
  grid-auto-flow: column;
  grid-auto-columns: minmax(0, 1fr);
  counter-reset: phase;
}

.steps li::before {
  counter-increment: phase;
  content: counter(phase) ". ";
  color: var(--vp-c-brand-1);
  font-weight: 700;
}

.lanes {
  display: grid;
  grid-template-columns: repeat(8, minmax(0, 1fr));
  gap: .25rem;
  max-width: 28rem;
  margin: .6rem 0;
}

.lanes span {
  padding: .13rem;
  text-align: center;
  border: 1px dashed var(--vp-c-divider);
  border-radius: .3rem;
  font-variant-numeric: tabular-nums;
}

.lanes span.active {
  border-style: solid;
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}

figcaption {
  font-size: .8rem;
  color: var(--vp-c-text-2);
}

@media (max-width: 639px) {
  .mapping {
    grid-template-columns: minmax(0, 1fr);
  }

  .steps {
    grid-auto-flow: row;
  }
}
</style>
