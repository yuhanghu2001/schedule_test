<script setup>
import { computed } from 'vue'
import TrainingLogChart from './TrainingLogChart.vue'
import { linkKey, linkName, referenceDelay } from '../utils/linkTelemetry'

const props = defineProps({ snapshot: Object, links: Object, samples: Array })
defineEmits(['select-node'])
const resources = ['CPU', 'NPU', 'FPGA']
const bands = [
  { name: '高负载', min: 80, range: '≥80%', tone: 'high' },
  { name: '较高', min: 60, range: '60%–<80%', tone: 'raised' },
  { name: '中等', min: 40, range: '40%–<60%', tone: 'mid' },
  { name: '低负载', min: 0, range: '<40%', tone: 'low' },
]
const percent = value => value == null ? '—' : value.toFixed(1) + '%'
const millis = value => value == null ? '—' : value.toFixed(1) + ' ms'
const nodes = computed(() => (props.snapshot?.nodes ?? []).map(node => {
  const shares = node.resource_used.map((used, index) => 100 * used / node.speed[index])
  const load = 100 * node.resource_used.reduce((sum, used) => sum + used, 0) /
    node.speed.reduce((sum, speed) => sum + speed, 0)
  return {
    node, shares, load,
    name: node.kind === 'cloud' ? 'C0' : 'E' + node.id,
    band: bands.find(band => load >= band.min),
  }
}))
const edges = computed(() => nodes.value.filter(item => item.node.kind === 'edge'))
const alerts = computed(() => {
  if (!props.links) return []
  const offline = props.links.offline.map(link => ({
    key: linkKey(link), name: linkName(link), detail: '链路断开', tone: 'high',
  }))
  const overloaded = nodes.value.flatMap(({ node, name, shares }) => {
    const load = Math.max(...shares)
    const queue = 100 * node.queue_used / node.queue_cap
    if (Math.max(load, queue) < 80) return []
    const detail = queue > load
      ? `队列占用 ${node.queue_used}/${node.queue_cap}（${percent(queue)}）`
      : `${resources[shares.indexOf(load)]} 利用率 ${percent(load)}`
    return [{ key: name, name, detail, node, tone: 'high', severity: Math.max(load, queue) }]
  }).sort((a, b) => b.severity - a.severity)
  const congested = props.links.congested.map(link => ({
    key: linkKey(link), name: linkName(link), tone: 'raised',
    detail: `拥塞 ${percent(100 * Math.max(link.background_fwd, link.background_rev))}，单跳 ${millis(Math.max(referenceDelay(link, 'fwd'), referenceDelay(link, 'rev')))}`,
  }))
  return [...offline, ...overloaded, ...congested]
})
const trend = computed(() => ({
  animation: false,
  tooltip: {
    trigger: 'axis', confine: true, backgroundColor: '#082342', borderColor: '#28638f',
    textStyle: { color: '#dcecff' },
    formatter: ([point]) => `已调度 ${point.name} 项任务<br>链路利用率 ${percent(point.value)}`,
  },
  grid: { left: 36, right: 10, top: 8, bottom: 23 },
  xAxis: {
    type: 'category',
    data: props.samples.map(sample => sample.task),
    boundaryGap: false,
    axisLabel: { color: '#abc9e4', fontSize: 10, hideOverlap: true },
    axisTick: { show: false },
    axisLine: { lineStyle: { color: '#28638f' } },
  },
  yAxis: {
    type: 'value', min: 0, max: ({ max }) => Math.max(5, Math.ceil(max / 5) * 5), splitNumber: 2,
    axisLabel: { formatter: '{value}%', color: '#abc9e4', fontSize: 10, hideOverlap: true },
    splitLine: { lineStyle: { color: '#153f63' } },
  },
  series: [{
    type: 'line', data: props.samples.map(sample => sample.linkUse),
    showSymbol: props.samples.length <= 16, symbolSize: 4,
    lineStyle: { color: '#41e7f3', width: 2 }, itemStyle: { color: '#41e7f3' },
    areaStyle: { color: '#41e7f3', opacity: .12 },
  }],
}))
</script>

<template>
  <section class="monitor-panel network-resource-panel">
    <h2 class="monitor-title">网络资源状态与调度瓶颈</h2>
    <div class="network-resource-body">
      <section class="network-resource-block">
        <h3>节点资源负载</h3>
        <div v-if="edges.length" class="network-node-layout">
          <div class="network-node-grid">
            <button
              v-for="item in edges"
              :key="item.node.id"
              type="button"
              class="network-node-load"
              :data-tone="item.band.tone"
              :aria-label="`${item.name} 综合利用率 ${percent(item.load)}，查看节点详情`"
              @click="$emit('select-node', item.node)"
            >
              <b>{{ item.name }}</b><span>{{ percent(item.load) }}</span>
            </button>
          </div>
          <dl class="network-load-bands" aria-label="节点负载分级">
            <div v-for="band in bands" :key="band.tone" :data-tone="band.tone">
              <dt>{{ band.range }}</dt><dd>{{ band.name }}</dd>
            </div>
          </dl>
        </div>
      </section>
      <section class="network-resource-block network-link-block">
        <h3>网络链路状态</h3>
        <dl v-if="links" class="network-link-metrics">
          <div title="当前快照的在线链路数与已知链路总数"><dt>在线链路</dt><dd>{{ links.online }}/{{ links.total }}</dd></div>
          <div title="当前在线链路双向的平均 1 MB 单跳估算时延"><dt>平均单跳时延</dt><dd>{{ millis(links.delay) }}</dd></div>
          <div title="本轮任务占用的在线链路容量占比，按双向容量加权"><dt>链路利用率</dt><dd>{{ percent(links.utilization) }}</dd></div>
          <div title="任一方向的背景拥塞达到 70% 的在线链路"><dt>拥塞链路</dt><dd>{{ links.congested.length }}</dd></div>
        </dl>
        <TrainingLogChart v-if="samples.length" :option="trend" label="链路利用率随已调度任务数变化" />
      </section>
      <section class="network-resource-block network-alert-block">
        <h3>调度瓶颈与异常提醒</h3>
        <ul v-if="alerts.length" class="network-alerts">
          <li v-for="alert in alerts" :key="alert.key" :data-tone="alert.tone">
            <button v-if="alert.node" type="button" @click="$emit('select-node', alert.node)"><b>{{ alert.name }}</b> {{ alert.detail }}</button>
            <p v-else><b>{{ alert.name }}</b> {{ alert.detail }}</p>
          </li>
        </ul>
        <p v-else class="network-alert-empty">{{ links ? '当前未发现高负载、拥塞或断链。' : '等待网络快照。' }}</p>
      </section>
    </div>
  </section>
</template>

<style scoped>
.network-resource-panel { flex: 1; }
.network-resource-panel > .monitor-title { font-size: 18px; letter-spacing: 0; }
.network-resource-body { display: flex; flex-direction: column; min-height: 0; flex: 1; gap: 6px; padding: 6px; overflow: auto; }
.network-resource-block { flex: none; padding: 7px; border: 1px solid #155283; border-radius: 5px; background: #082342; }
.network-resource-block h3 { margin-bottom: 6px; font-size: 13px; }
.network-node-layout { display: grid; grid-template-columns: minmax(0, 1fr) 102px; align-items: start; gap: 7px; }
.network-node-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 4px; }
.network-node-load { display: grid; place-items: center; gap: 1px; min-height: 33px; min-width: 0; padding: 2px; border: 1px solid #205477; border-radius: 4px; color: #e2eeff; background: #092f51; line-height: 1.15; }
.network-node-load b, .network-node-load span { font-size: 11px; font-variant-numeric: tabular-nums; }
.network-node-load span, .network-load-bands dt, .network-alerts b { color: var(--load-ink); }
.network-node-load[data-tone='mid'] { background: #0a3b40; border-color: #196d68; }
.network-node-load[data-tone='raised'] { background: #3a3926; border-color: #7f683d; }
.network-node-load[data-tone='high'] { background: #483328; border-color: #a45c40; }
.network-node-load:hover { border-color: #6ee6ff; }
[data-tone='low'] { --load-ink: #6ee6ff; }
[data-tone='mid'] { --load-ink: #51dfb3; }
[data-tone='raised'] { --load-ink: #ffdb88; }
[data-tone='high'] { --load-ink: #ffad79; }
.network-load-bands { display: grid; gap: 5px; margin: 0; padding: 6px; border: 1px solid #17476d; border-radius: 4px; background: #061c35; font-size: 10px; }
.network-load-bands div { display: flex; justify-content: space-between; gap: 3px; }
.network-load-bands dd { margin: 0; color: #c6dbf4; white-space: nowrap; }
.network-link-block, .network-alert-block { display: flex; flex-direction: column; }
.network-link-block { height: 176px; }
.network-link-metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 4px; margin: 0 0 4px; }
.network-link-metrics > div { padding: 4px 2px; text-align: center; border: 1px solid #205477; border-radius: 4px; background: #092a4c; }
.network-link-metrics dt { color: #abc9e4; font-size: 10px; white-space: nowrap; }
.network-link-metrics dd { margin: 2px 0 0; color: #41e7f3; font-size: 12px; font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }
.network-alert-block { height: 120px; }
.network-alerts { display: grid; align-content: start; gap: 3px; min-height: 0; flex: 1; overflow: auto; padding: 0; margin: 0; list-style: none; }
.network-alerts li { min-width: 0; background: #092a4c; border: 1px solid #17476d; border-radius: 4px; }
.network-alerts button, .network-alerts p { width: 100%; padding: 4px 6px; border: 0; background: transparent; color: #dcecff; line-height: 14px; text-align: left; font-size: 11px; font-variant-numeric: tabular-nums; }
.network-alerts button:hover { background: #10375d; }
.network-alert-empty { padding: 4px 2px; color: #abc9e4; font-size: 11px; }
@media (min-width: 1800px) {
  .network-resource-body { gap: 9px; padding: 9px; }
  .network-resource-block { padding: 9px; }
  .network-node-load { min-height: 39px; }
}
@media (min-width: 1800px) and (min-height: 1000px) {
  .network-link-block { height: 252px; }
  .network-alert-block { height: 122px; }
}
@media (max-width: 1200px) { .network-resource-panel > .monitor-title { font-size: 16px; } }
@media (max-width: 850px) { .network-resource-panel { flex: none; } }
</style>
