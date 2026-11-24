<script setup>
import { computed, ref } from 'vue'
import TrainingLogChart from '../components/TrainingLogChart.vue'
import { useAppStore } from '../stores/app'
import { dispatchAlgorithms as algorithms } from '../utils/dispatchAlgorithms'

const app = useAppStore()
const metrics = [
  { key: 'task_success_rate', name: '任务完成率', unit: '%', factor: 100, digits: 2, direction: 1 },
  { key: 'avg_delay_s', name: '平均完成时延', unit: 's', factor: 1, digits: 2, direction: -1 },
  { key: 'avg_energy_j', name: '平均任务能耗', unit: 'J', factor: 1, digits: 2, direction: -1 },
  { key: 'load_balance_degree', name: '负载均衡度', unit: '', factor: 1, digits: 3, direction: 1 },
  { key: 'avg_dag_makespan_s', name: 'DAG Makespan', unit: 's', factor: 1, digits: 3, direction: -1 },
  { key: 'cpu_load_std', name: 'CPU 负载标准差', unit: '', factor: 1, digits: 3, direction: -1 },
  { key: 'npu_load_std', name: 'NPU 负载标准差', unit: '', factor: 1, digits: 3, direction: -1 },
  { key: 'fpga_load_std', name: 'FPGA 负载标准差', unit: '', factor: 1, digits: 3, direction: -1 },
  { key: 'fpga_reconfig_rate', name: 'FPGA 重构率', unit: '%', factor: 100, digits: 2, direction: -1 },
]
const radarMetrics = [
  metrics[0],
  { key: 'acceptance_rate', name: '任务接收率', factor: 100, direction: 1 },
  ...metrics.slice(1, 4),
]
const selected = ref(algorithms.map(item => item.id))
const selectedMetrics = ref(metrics.map(item => item.key))
const trendKey = ref(metrics[0].key)
const completed = computed(() => app.completedDispatch)
const runs = computed(() => algorithms.filter(item => completed.value[item.id]))
const activeMetrics = computed(() => metrics.filter(item => selectedMetrics.value.includes(item.key)))
const trendMetric = computed(() => activeMetrics.value.find(item => item.key === trendKey.value) ?? activeMetrics.value[0])
const rows = computed(() => runs.value
  .filter(item => selected.value.includes(item.id))
  .map(item => {
    const samples = completed.value[item.id].samples
    return { ...item, samples, last: samples.at(-1) }
  }))
const taskCount = computed(() => runs.value.length ? completed.value[runs.value[0].id].samples.at(-1).tasks : 0)

const value = (row, metric) => row[metric.key] == null ? null : row[metric.key] * metric.factor
const number = (amount, digits) => amount == null ? '—' : amount.toFixed(digits)
const text = (row, metric) => number(value(row, metric), metric.digits)
const radarPerformance = computed(() => radarMetrics.map(metric => {
  const ranked = rows.value.filter(row => row.last[metric.key] != null)
    .sort((a, b) => metric.direction * (b.last[metric.key] - a.last[metric.key]))
  return { ...metric, best: ranked[0], worst: ranked.at(-1) }
}))
const performance = computed(() => activeMetrics.value.map(metric => {
  const ranked = rows.value.filter(row => row.last[metric.key] != null).sort((a, b) => metric.direction * (b.last[metric.key] - a.last[metric.key]))
  return { ...metric, best: ranked[0], worst: ranked.at(-1) }
}))

function difference(metric) {
  if (rows.value.length < 2 || metric.best === metric.worst) return ''
  const units = metric.unit === '%' ? '百分点' : metric.unit
  const delta = value(metric.best.last, metric) - value(metric.worst.last, metric)
  return `${delta > 0 ? '+' : ''}${number(delta, metric.digits)}${units ? ' ' + units : ''}`
}
function barHeight(row, metric) {
  const amount = value(row, metric)
  const max = Math.max(...rows.value.map(item => value(item.last, metric)))
  return amount > 0 ? amount / max * 82 + '%' : '0%'
}
function normalized(row, metric) {
  if (row.last[metric.key] == null) return null
  const best = value(metric.best.last, metric)
  const worst = value(metric.worst.last, metric)
  return best === worst ? 1 : (value(row.last, metric) - worst) / (best - worst)
}
const radarRows = computed(() => rows.value.filter(row => radarMetrics.every(metric => row.last[metric.key] != null)))
const chartText = '#b8d2eb'
const chartBase = {
  animation: false,
  tooltip: { confine: true, backgroundColor: '#082342', borderColor: '#28638f', textStyle: { color: '#dcecff' } },
}
const radar = computed(() => ({
  ...chartBase,
  tooltip: {
    ...chartBase.tooltip,
    formatter: point => [point.name, ...radarMetrics.map((metric, index) =>
      `${metric.name}　${number(point.value[index], 2)}`)].join('<br>'),
  },
  radar: {
    center: ['50%', '52%'], radius: '68%', shape: 'polygon', splitNumber: 5,
    indicator: radarMetrics.map(metric => ({ name: metric.name, max: 1 })),
    axisName: { color: chartText, fontSize: 12 },
    splitArea: { areaStyle: { color: ['#061b34', '#092a4c'] } },
    splitLine: { lineStyle: { color: '#28638f' } },
    axisLine: { lineStyle: { color: '#28638f' } },
  },
  series: [{
    type: 'radar', symbol: 'none',
    data: radarRows.value.map(row => ({
      name: row.name, value: radarPerformance.value.map(metric => normalized(row, metric)),
      lineStyle: { color: row.color, width: 2 },
      areaStyle: { color: row.color, opacity: .09 },
      itemStyle: { color: row.color },
    })),
  }],
}))
const trend = computed(() => {
  const metric = trendMetric.value
  const axis = {
    type: 'value',
    axisLabel: { color: chartText, hideOverlap: true },
    axisLine: { show: true, lineStyle: { color: '#28638f' } },
    splitLine: { lineStyle: { color: '#153f63' } },
    nameTextStyle: { color: chartText, fontSize: 11 },
  }
  return {
    ...chartBase,
    tooltip: {
      ...chartBase.tooltip, trigger: 'axis',
      formatter: points => [`已调度 ${points[0].value[0]} 项任务`, ...points.map(point =>
        `${point.seriesName}　${number(point.value[1], metric.digits)} ${metric.unit}`)].join('<br>'),
    },
    grid: { left: 10, right: 18, top: 20, bottom: 27, containLabel: true },
    xAxis: {
      ...axis, min: 0, max: taskCount.value, minInterval: 1,
      name: '累计已调度任务数', nameLocation: 'middle', nameGap: 23,
    },
    yAxis: {
      ...axis, scale: true,
      axisLabel: { ...axis.axisLabel, formatter: amount => metric.unit === '%' ? amount + '%' : amount },
    },
    series: rows.value.map(row => ({
      type: 'line', name: row.name, showSymbol: row.samples.length <= 24, symbolSize: 4,
      lineStyle: { color: row.color, width: 2 }, itemStyle: { color: row.color },
      data: row.samples.map(sample => [sample.tasks, value(sample, metric)]),
    })),
  }
})
</script>

<template>
  <div class="dispatch-logs">
    <aside class="monitor-panel dispatch-filters">
      <h1 class="monitor-title">调度记录筛选</h1>
      <form @submit.prevent>
        <fieldset>
          <legend>对比算法</legend>
          <div class="dispatch-options">
            <label v-for="item in algorithms" :key="item.id">
              <input v-model="selected" type="checkbox" :value="item.id" :disabled="!completed[item.id]" />
              <span :style="{ color: item.color }">{{ item.name }}</span>
              <small v-if="!completed[item.id]">无完成记录</small>
            </label>
          </div>
        </fieldset>
        <fieldset>
          <legend>统计指标</legend>
          <div class="dispatch-options">
            <label v-for="metric in metrics" :key="metric.key">
              <input v-model="selectedMetrics" type="checkbox" :value="metric.key" :disabled="selectedMetrics.length === 1 && selectedMetrics.includes(metric.key)" />{{ metric.name }}
            </label>
          </div>
        </fieldset>
      </form>
    </aside>

    <div class="dispatch-results">
      <section class="monitor-panel">
        <h2 class="monitor-title">调度效果总览<small v-if="runs.length">{{ taskCount }} 项任务 · {{ runs.length }} 种算法已完成</small></h2>
        <div class="dispatch-kpis" :style="{ '--metric-count': activeMetrics.length }">
          <article v-for="metric in performance" :key="metric.key" class="dispatch-kpi">
            <h3>{{ metric.name }}</h3>
            <div class="dispatch-kpi-value">
              <strong>{{ metric.best ? text(metric.best.last, metric) : '—' }} <small v-if="metric.best && metric.unit">{{ metric.unit }}</small></strong>
              <span v-if="difference(metric)" class="dispatch-delta">较最差 {{ difference(metric) }}</span>
            </div>
            <p v-if="metric.best" :style="{ color: metric.best.color }">{{ metric.best.name }}</p>
          </article>
        </div>
      </section>
      <div class="dispatch-pair dispatch-middle">
        <section class="monitor-panel">
          <h2 class="monitor-title">各算法调度结果对比<small v-if="rows.length">{{ rows.length }} 种算法</small></h2>
          <div v-if="rows.length" class="dispatch-table-scroll">
            <table class="dispatch-table">
              <thead><tr><th scope="col">调度算法</th><th v-for="metric in activeMetrics" :key="metric.key" scope="col">{{ metric.name }}<template v-if="metric.unit"> ({{ metric.unit }})</template></th></tr></thead>
              <tbody><tr v-for="row in rows" :key="row.id">
                <th scope="row" :style="{ color: row.color }">{{ row.name }}</th>
                <td v-for="metric in performance" :key="metric.key" :class="{ 'dispatch-best': metric.best && row.last[metric.key] === metric.best.last[metric.key] }">{{ text(row.last, metric) }}</td>
              </tr></tbody>
            </table>
          </div>
          <p v-else class="dispatch-empty">{{ runs.length ? '当前筛选条件下没有调度记录。' : '请先在系统总览完成一次推演。' }}</p>
        </section>
        <section class="monitor-panel">
          <h2 class="monitor-title">综合性能对比（归一化）</h2>
          <TrainingLogChart v-if="radarRows.length" :option="radar" label="各算法五维归一化雷达图" />
          <p v-else class="dispatch-empty">有完整指标后显示雷达图。</p>
          <div v-if="radarRows.length" class="dispatch-legend">
            <span v-for="row in radarRows" :key="row.id" :style="{ color: row.color }">{{ row.name }}</span>
          </div>
        </section>
      </div>
      <div class="dispatch-pair dispatch-bottom">
        <section class="monitor-panel">
          <h2 class="monitor-title">关键指标对比分析</h2>
          <div v-if="rows.length" class="dispatch-bars" :style="{ '--metric-count': activeMetrics.length }">
            <div v-for="metric in activeMetrics" :key="metric.key" class="dispatch-bar-column">
              <h3>{{ metric.name }}<template v-if="metric.unit"> ({{ metric.unit }})</template></h3>
              <div class="dispatch-bar-group">
                <div v-for="row in rows" :key="row.id" class="dispatch-bar-item" :style="{ '--bar-color': row.color, '--bar-height': barHeight(row.last, metric) }" :title="row.name">
                  <b>{{ text(row.last, metric) }}</b>
                  <span aria-hidden="true"></span>
                </div>
              </div>
            </div>
          </div>
          <p v-else class="dispatch-empty">有调度记录后显示指标对比。</p>
          <div v-if="rows.length" class="dispatch-legend">
            <span v-for="row in rows" :key="row.id" :style="{ color: row.color }">{{ row.name }}</span>
          </div>
        </section>
        <section class="monitor-panel">
          <h2 class="monitor-title">调度指标随任务进度变化<small><select :value="trendMetric.key" aria-label="趋势指标" @change="trendKey = $event.target.value">
            <option v-for="metric in activeMetrics" :key="metric.key" :value="metric.key">{{ metric.name }}</option>
          </select></small></h2>
          <TrainingLogChart v-if="rows.length" :option="trend" :label="`${trendMetric.name}调度趋势`" />
          <p v-else class="dispatch-empty">有调度记录后显示时间趋势。</p>
          <div v-if="rows.length" class="dispatch-legend">
            <span v-for="row in rows" :key="row.id" :style="{ color: row.color }">{{ row.name }}</span>
          </div>
        </section>
      </div>
    </div>
  </div>
</template>

<style scoped>
.dispatch-logs { display: grid; grid-template-columns: minmax(224px, 16.5%) minmax(0, 1fr); height: calc(100vh - 168px); gap: 13px; }
.dispatch-logs .monitor-panel { min-width: 0; }
.dispatch-logs .monitor-title { font-size: 18px; letter-spacing: 0; }
.dispatch-logs .monitor-title small { color: #b8d2eb; }
.dispatch-filters form { flex: 1; min-height: 0; display: flex; flex-direction: column; gap: 16px; overflow: auto; padding: 14px; }
.dispatch-filters fieldset { margin: 0; padding: 0; border: 0; min-width: 0; }
.dispatch-filters legend, .dispatch-filters fieldset > label { display: block; padding: 0; margin-bottom: 6px; color: #e2eeff; font-weight: 600; }
.dispatch-logs select { width: 100%; min-width: 0; padding: 6px 8px; border: 1px solid #28638f; border-radius: 5px; color: #dcecff; background: #082342; font: inherit; }
.dispatch-logs select:focus-visible { outline: 2px solid #6ee6ff; outline-offset: 2px; }
.dispatch-options { display: grid; padding: 4px; border: 1px solid #1b527c; border-radius: 5px; background: #082342; }
.dispatch-options label { display: flex; align-items: center; gap: 7px; padding: 4px 5px; line-height: 17px; font-size: 13px; }
.dispatch-options label:hover { background: #10375d; }
.dispatch-options input { margin: 0; accent-color: #008cff; }
.dispatch-options small { margin-left: auto; color: #b8d2eb; }
.dispatch-results { display: grid; grid-template-rows: auto minmax(0, 1fr) minmax(0, 1.12fr); gap: 12px; min-height: 0; }
.dispatch-kpis { display: grid; grid-template-columns: repeat(var(--metric-count), minmax(0, 1fr)); gap: 12px; padding: 10px; }
.dispatch-kpi { min-width: 0; padding: 9px 12px; border: 1px solid #1b527c; border-radius: 6px; background: #082342; }
.dispatch-kpi h3 { font-size: 14px; }
.dispatch-kpi-value { display: flex; flex-wrap: wrap; align-items: baseline; gap: 1px 10px; }
.dispatch-kpi strong { color: #41e7f3; font-size: 28px; font-variant-numeric: tabular-nums; }
.dispatch-kpi strong small { font-size: 14px; }
.dispatch-delta { color: #51dfb3; font-size: 12px; white-space: nowrap; }
.dispatch-kpi p { font-size: 11px; }
.dispatch-pair { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr); gap: 12px; min-height: 0; }
.dispatch-table-scroll { overflow: auto; margin: 0 9px 8px; }
.dispatch-table { border-collapse: collapse; width: 100%; text-align: center; font-variant-numeric: tabular-nums; white-space: nowrap; }
.dispatch-table th, .dispatch-table td { padding: 7px; border-bottom: 1px solid #144b74; }
.dispatch-table thead { color: #c9dffa; background: #0b335a; font-size: 12px; }
.dispatch-table tbody th { text-align: left; font-size: 13px; }
.dispatch-table tbody tr:nth-child(odd) { background: #082342; }
.dispatch-best { color: #41e7f3; font-weight: 700; background: #0a3453; }
.dispatch-empty { display: grid; place-items: center; flex: 1; padding: 14px; text-align: center; color: #b8d2eb; font-size: 13px; }
.dispatch-legend { display: flex; flex-wrap: wrap; justify-content: center; gap: 5px 13px; padding: 7px 9px 9px; font-size: 11px; }
.dispatch-bars { display: grid; grid-template-columns: repeat(var(--metric-count), minmax(0, 1fr)); min-height: 0; flex: 1; padding: 12px 5px 4px; }
.dispatch-bar-column { display: flex; flex-direction: column; min-width: 0; min-height: 0; padding: 0 6px; }
.dispatch-bar-column + .dispatch-bar-column { border-left: 1px solid #144b74; }
.dispatch-bar-column h3 { font-size: 13px; text-align: center; margin-bottom: 7px; }
.dispatch-bar-group { display: flex; justify-content: center; flex: 1; min-height: 0; border-bottom: 1px solid #28638f; }
.dispatch-bar-item { position: relative; flex: 1; min-width: 0; max-width: 58px; }
.dispatch-bar-item span { position: absolute; bottom: 0; left: 11%; width: 78%; height: var(--bar-height); background: var(--bar-color); border-radius: 3px 3px 0 0; }
.dispatch-bar-item b { position: absolute; bottom: var(--bar-height); left: 50%; transform: translateX(-50%); padding-bottom: 3px; color: #e2eeff; font-size: 11px; font-variant-numeric: tabular-nums; }
.dispatch-results .monitor-title select { padding: 3px 5px; font-size: 12px; }
@media (max-width: 1700px) {
  .dispatch-logs { height: calc(100vh - 147px); }
  .dispatch-logs .monitor-title { font-size: 16px; }
  .dispatch-filters form { gap: 7px; padding: 9px 11px; }
  .dispatch-options label { padding-block: 3px; line-height: 16px; }
  .dispatch-kpi strong { font-size: 24px; }
}
@media (max-width: 1300px), (max-height: 790px) {
  .dispatch-logs { height: auto; }
  .dispatch-results { grid-template-rows: auto minmax(270px, auto) minmax(290px, auto); }
}
@media (max-width: 950px) {
  .dispatch-logs, .dispatch-pair { grid-template-columns: minmax(0, 1fr); }
  .dispatch-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .dispatch-bars { min-height: 220px; }
  .dispatch-logs :deep(.log-chart) { min-height: 220px; }
}
</style>
