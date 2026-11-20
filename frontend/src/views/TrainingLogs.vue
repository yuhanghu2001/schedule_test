<script setup>
import { computed, onMounted, ref } from 'vue'
import api from '../api/client'
import TrainingLogChart from '../components/TrainingLogChart.vue'

const metrics = [
  { key: 'task_success_rate', name: '任务完成率', unit: '%', factor: 100, digits: 2 },
  { key: 'avg_delay_s', name: '平均完成时延', unit: 's', factor: 1, digits: 3 },
  { key: 'avg_energy_j', name: '平均任务能耗', unit: 'J', factor: 1, digits: 3 },
  { key: 'load_balance_degree', name: '负载均衡度', unit: '', factor: 1, digits: 3 },
]
const rewardMetric = { key: 'episode_reward', unit: '', factor: 1, digits: 2 }
const models = {
  gatppo: ['GAT-PPO', '#52c7ff'],
  ppo: ['PPO', '#51dfb3'],
  shuffled_gat: ['GAT-PPO（图打乱）', '#ffbd63'],
}
const aliases = { gat: 'gatppo', gatpppo: 'gatppo', flat: 'ppo' }
const statuses = { queued: '等待训练', running: '训练中', completed: '已完成', stopped: '已停止', failed: '训练失败' }
const sources = { training: '训练回合', evaluation: '固定种子验证' }
const batches = ref([])
const batchId = ref('')
const selectedIds = ref([])
const source = ref('training')
const fromEpisode = ref('')
const toEpisode = ref('')
const busy = ref(true)
const applied = ref(null)
const trendKey = ref(metrics[0].key)
const page = ref(1)
const pageSize = 5

const batch = computed(() => batches.value.find(item => item.id === batchId.value))
const lastEpisode = computed(() => Math.max(...batch.value.runs.map(run => run.current_episode)))
const groups = computed(() => applied.value?.groups ?? [])
const rows = computed(() => groups.value.flatMap(group => group.rows.map(row => ({ ...row, group })))
  .sort((a, b) => b.episode - a.episode || a.group.name.localeCompare(b.group.name)))
const pageCount = computed(() => Math.max(1, Math.ceil(rows.value.length / pageSize)))
const visibleRows = computed(() => rows.value.slice((page.value - 1) * pageSize, page.value * pageSize))
const byEpisode = computed(() => groups.value.map(group => new Map(group.rows.map(row => [row.episode, row]))))
const summaryEpisode = computed(() => {
  if (!byEpisode.value.length) return null
  const [first, ...others] = byEpisode.value
  return [...first.keys()].reverse().find(episode => others.every(index => index.has(episode))) ?? null
})
const summaries = computed(() => summaryEpisode.value == null ? [] : groups.value.map((group, index) => ({
  ...group, row: byEpisode.value[index].get(summaryEpisode.value),
})))
const trendMetric = computed(() => metrics.find(metric => metric.key === trendKey.value))
const metricValue = (row, metric) => row[metric.key] == null ? null : row[metric.key] * metric.factor
const format = (value, digits) => value == null ? '—' : Number(value).toFixed(digits)
const metricText = (row, metric) => format(metricValue(row, metric), metric.digits)
const formatBatch = id => id.replace(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/, '$1-$2-$3 $4:$5:$6')

function selectBatch() {
  selectedIds.value = batch.value.runs.map(run => run.run_id)
  fromEpisode.value = ''
  toEpisode.value = ''
}

async function query() {
  const selection = batch.value.runs.filter(run => selectedIds.value.includes(run.run_id))
  const filters = { batchId: batchId.value, source: source.value, from: fromEpisode.value, to: toEpisode.value }
  busy.value = true
  try {
    const groups = await Promise.all(selection.map(async run => {
      const data = await api.get(`/training/fs-runs/${encodeURIComponent(run.run_id)}/metrics`)
      const evaluation = filters.source === 'evaluation'
      const suffix = evaluation ? '_mean' : ''
      const rows = data[filters.source]
        .filter(row => (!filters.from || row.episode >= filters.from) && (!filters.to || row.episode <= filters.to))
        .map(row => ({
          episode: row.episode,
          sampleCount: evaluation ? row.evaluation_seeds.length : row.tasks,
          diagnostic: evaluation ? row.objective_score_mean : row.loss,
          episode_reward: row[rewardMetric.key + suffix],
          ...Object.fromEntries(metrics.map(metric => [metric.key, row[metric.key + suffix]])),
        }))
        .sort((a, b) => a.episode - b.episode)
      return { ...run, rows }
    }))
    applied.value = { ...filters, groups }
    page.value = 1
  } finally {
    busy.value = false
  }
}

onMounted(async () => {
  try {
    const runs = await api.get('/training/fs-runs')
    const grouped = new Map()
    for (const run of runs) {
      const id = run.run_id.split('/')[0]
      if (!grouped.has(id)) grouped.set(id, { id, runs: [] })
      const [name, color] = models[aliases[run.model_type] ?? run.model_type]
      grouped.get(id).runs.push({ ...run, name, color })
    }
    batches.value = [...grouped.values()].sort((a, b) => b.id.localeCompare(a.id))
    for (const item of batches.value) item.runs.sort((a, b) => a.name.localeCompare(b.name))
    if (batches.value.length) {
      batchId.value = batches.value[0].id
      selectBatch()
      await query()
    }
  } finally {
    busy.value = false
  }
})

function barFraction(row, metric) {
  const value = metricValue(row, metric)
  if (!value) return 0
  const max = metric.key === 'task_success_rate' ? 100 : metric.key === 'load_balance_degree' ? 1
    : Math.max(...summaries.value.map(item => metricValue(item.row, metric)))
  return value / max
}

const axis = {
  type: 'value',
  nameTextStyle: { color: '#b8d2eb', fontSize: 11 },
  axisLabel: { color: '#b8d2eb', hideOverlap: true },
  axisLine: { show: true, lineStyle: { color: '#28638f' } },
  splitLine: { lineStyle: { color: '#153f63' } },
}
const chartBase = {
  animation: false,
  grid: { left: 10, right: 22, top: 20, bottom: 26, containLabel: true },
  tooltip: {
    confine: true, backgroundColor: '#082342', borderColor: '#28638f',
    textStyle: { color: '#dcecff' },
  },
}
const roundChart = metric => ({
  ...chartBase,
  tooltip: {
    ...chartBase.tooltip, trigger: 'axis',
    formatter: points => [`第 ${points[0].value[0]} 轮`, ...points.map(point =>
      `${point.seriesName}　${format(point.value[1], metric.digits)} ${metric.unit}`)].join('<br>'),
  },
  xAxis: { ...axis, minInterval: 1, name: '回合', nameLocation: 'middle', nameGap: 24 },
  yAxis: {
    ...axis, scale: true,
    axisLabel: { ...axis.axisLabel, formatter: value => metric.unit === '%' ? `${value}%` : value },
  },
  series: groups.value.map(group => ({
    name: group.name, type: 'line', showSymbol: group.rows.length < 35, symbolSize: 4,
    lineStyle: { width: 2, color: group.color }, itemStyle: { color: group.color },
    data: group.rows.map(row => [row.episode, metricValue(row, metric)]),
  })),
})
const trend = computed(() => roundChart(trendMetric.value))
const rewardChart = computed(() => roundChart(rewardMetric))
const hasRewards = computed(() => rows.value.some(row => row.episode_reward != null))
const wrongRange = computed(() => Boolean(fromEpisode.value && toEpisode.value && fromEpisode.value > toEpisode.value))
</script>

<template>
  <div class="training-logs">
    <aside class="monitor-panel log-filters">
      <h1 class="monitor-title">日志查询与对比配置</h1>
      <form v-if="batch" class="log-filter-form" @submit.prevent="query">
        <fieldset :disabled="busy">
          <label for="log-batch">实验批次</label>
          <select id="log-batch" v-model="batchId" @change="selectBatch">
            <option v-for="item in batches" :key="item.id" :value="item.id">{{ formatBatch(item.id) }}</option>
          </select>
          <p class="log-note">北京时间 · {{ batch.runs.length }} 个模型</p>
        </fieldset>
        <fieldset :disabled="busy">
          <legend>对比模型</legend>
          <div class="log-options">
            <label v-for="run in batch.runs" :key="run.run_id">
              <input v-model="selectedIds" type="checkbox" :value="run.run_id" />
              <b :style="{ color: run.color }">{{ run.name }}</b>
              <small>{{ statuses[run.status] }} · {{ run.current_episode }}/{{ run.episodes }} 轮</small>
            </label>
          </div>
        </fieldset>
        <fieldset :disabled="busy">
          <legend>统计口径</legend>
          <div class="log-options log-source">
            <label v-for="(label, key) in sources" :key="key"><input v-model="source" type="radio" name="log-source" :value="key" />{{ label }}</label>
          </div>
          <p class="log-note">{{ source === 'evaluation' ? '每条记录是该回合的固定种子均值。' : '每条记录对应一次实际训练回合。' }}</p>
        </fieldset>
        <fieldset :disabled="busy">
          <legend>回合范围</legend>
          <div class="log-range">
            <input v-model.number="fromEpisode" type="number" min="1" :max="Math.max(1, lastEpisode)" placeholder="起始回合" aria-label="起始回合" />
            <span>至</span>
            <input v-model.number="toEpisode" type="number" min="1" :max="Math.max(1, lastEpisode)" placeholder="结束回合" aria-label="结束回合" />
          </div>
          <p class="log-note" :class="{ warn: wrongRange }">{{ wrongRange ? '起始回合不能晚于结束回合。' : '留空查看全部已记录回合。' }}</p>
        </fieldset>
        <button class="log-query" :disabled="busy || !selectedIds.length || wrongRange">{{ busy ? '读取日志中' : '查询与对比' }}</button>
      </form>
      <p v-else class="log-empty">{{ busy ? '读取实验批次中' : '暂无可查询的训练批次。' }}</p>
    </aside>

    <div class="log-results" :aria-busy="busy">
      <section class="monitor-panel log-summary">
        <h2 class="monitor-title">调度结果对比<small v-if="applied">{{ formatBatch(applied.batchId) }} · {{ sources[applied.source] }}<template v-if="summaryEpisode != null"> · 第 {{ summaryEpisode }} 轮</template></small></h2>
        <div v-if="summaries.length" class="log-scroll">
          <table class="log-table summary-table">
            <thead><tr><th scope="col">训练模型</th><th v-for="metric in metrics" :key="metric.key" scope="col">{{ metric.name }}{{ metric.unit ? ' (' + metric.unit + ')' : '' }}</th></tr></thead>
            <tbody><tr v-for="item in summaries" :key="item.run_id">
              <th scope="row" :style="{ color: item.color }">{{ item.name }}</th>
              <td v-for="metric in metrics" :key="metric.key">{{ metricText(item.row, metric) }}</td>
            </tr></tbody>
          </table>
        </div>
        <p v-else class="log-empty">{{ rows.length ? '所选模型没有共同的记录回合。' : '当前查询暂无可比较的回合记录。' }}</p>
      </section>

      <div class="log-analysis">
        <section class="monitor-panel">
          <h2 class="monitor-title">多指标对比分析<small v-if="summaryEpisode != null">第 {{ summaryEpisode }} 轮</small></h2>
          <div v-if="summaries.length" class="log-bars">
            <div v-for="metric in metrics" :key="metric.key" class="log-bar-column">
              <h3>{{ metric.name }}<small v-if="metric.unit"> ({{ metric.unit }})</small></h3>
              <div class="log-bar-group">
                <div v-for="item in summaries" :key="item.run_id" class="log-bar-item" :style="{ '--model-color': item.color, '--bar-height': `${barFraction(item.row, metric) * 86}%` }">
                  <div class="log-bar-track">
                    <b>{{ metricText(item.row, metric) }}</b>
                    <span aria-hidden="true"></span>
                  </div>
                  <span>{{ item.name }}</span>
                </div>
              </div>
            </div>
          </div>
          <p v-else class="log-empty">有共同回合后显示指标对比。</p>
        </section>
        <section class="monitor-panel">
          <h2 class="monitor-title">指标随回合变化<small><select v-model="trendKey" aria-label="趋势指标"><option v-for="metric in metrics" :key="metric.key" :value="metric.key">{{ metric.name }}</option></select></small></h2>
          <TrainingLogChart v-if="rows.length" :option="trend" :label="`${trendMetric.name}随回合变化`" />
          <p v-else class="log-empty">暂无趋势记录。</p>
          <div v-if="rows.length" class="log-legend">
            <span v-for="group in groups" :key="group.run_id" :style="{ color: group.color }">{{ group.name }}</span>
          </div>
        </section>
      </div>

      <div class="log-details">
        <section class="monitor-panel">
          <h2 class="monitor-title">训练日志明细<small v-if="applied">{{ sources[applied.source] }} · {{ rows.length }} 条</small></h2>
          <div v-if="rows.length" class="log-scroll detail-scroll">
            <table class="log-table detail-table">
              <thead><tr>
                <th scope="col">回合</th><th scope="col">模型</th>
                <th scope="col">{{ applied.source === 'training' ? '任务数' : '种子数' }}</th>
                <th scope="col">完成率 (%)</th><th scope="col">时延 (s)</th><th scope="col">能耗 (J)</th>
                <th scope="col">均衡度</th><th scope="col">{{ applied.source === 'training' ? '损失' : '目标得分' }}</th>
              </tr></thead>
              <tbody><tr v-for="row in visibleRows" :key="`${row.group.run_id}/${row.episode}`">
                <td>{{ row.episode }}</td><th scope="row" :style="{ color: row.group.color }">{{ row.group.name }}</th>
                <td>{{ row.sampleCount ?? '—' }}</td>
                <td v-for="metric in metrics" :key="metric.key">{{ metricText(row, metric) }}</td>
                <td>{{ format(row.diagnostic, 4) }}</td>
              </tr></tbody>
            </table>
          </div>
          <p v-else class="log-empty">当前条件下没有日志记录。</p>
          <div v-if="rows.length" class="log-pagination" aria-label="日志分页">
            <button :disabled="page === 1" @click="page--">上一页</button>
            <span aria-live="polite">第 {{ page }} / {{ pageCount }} 页</span>
            <button :disabled="page === pageCount" @click="page++">下一页</button>
          </div>
        </section>
        <section class="monitor-panel">
          <h2 class="monitor-title">奖励函数对比<small v-if="applied">{{ applied.source === 'evaluation' ? '累计奖励 · 固定种子均值' : '回合累计奖励' }}</small></h2>
          <TrainingLogChart v-if="hasRewards" :option="rewardChart" label="各模型的累计奖励随回合变化" />
          <p v-else class="log-empty">当前条件下没有奖励记录。</p>
          <div v-if="hasRewards" class="log-legend">
            <span v-for="group in groups" :key="group.run_id" :style="{ color: group.color }">{{ group.name }}</span>
          </div>
        </section>
      </div>
    </div>
  </div>
</template>

<style scoped>
.training-logs { display: grid; grid-template-columns: minmax(230px, 16.5%) minmax(0, 1fr); gap: 13px; height: calc(100vh - 168px); }
.training-logs .monitor-panel { min-width: 0; }
.training-logs .monitor-title { font-size: 18px; letter-spacing: 0; }
.training-logs .monitor-title small { color: #b8d2eb; }
.log-filter-form { display: flex; flex-direction: column; flex: 1; gap: 24px; padding: 18px 14px; }
fieldset { min-width: 0; margin: 0; padding: 0; border: 0; }
fieldset > label, legend { display: block; padding: 0; margin-bottom: 9px; color: #e2eeff; font-size: 15px; font-weight: 600; }
select, .log-range input, .log-pagination button, .log-query {
  border: 1px solid #28638f; border-radius: 5px; background: var(--el-bg-color);
  color: #dcecff; font: inherit;
}
select, .log-range input { width: 100%; padding: 7px 8px; background: #082342; }
select:focus-visible, .log-range input:focus-visible, input[type='checkbox']:focus-visible, input[type='radio']:focus-visible { outline: 2px solid #6ee6ff; outline-offset: 2px; }
.log-note { margin-top: 7px; color: #adc9e4; font-size: 12px; line-height: 1.5; }
.log-note.warn { color: #ffbd63; }
.log-options { display: grid; gap: 1px; padding: 3px; border: 1px solid #1b527c; border-radius: 6px; background: #082342; }
.log-options label { display: flex; flex-wrap: wrap; align-items: center; gap: 5px 7px; padding: 7px 5px; border-radius: 4px; cursor: pointer; }
.log-options label:has(input:checked) { background: #10375d; }
.log-options input { margin: 0; accent-color: #008cff; }
.log-options small { margin-left: auto; color: #adc9e4; font-size: 11px; white-space: nowrap; }
.log-range { display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr); align-items: center; gap: 6px; color: #b8d2eb; }
.log-range input { min-width: 0; }
.log-range input::placeholder { color: #adc9e4; font-size: 12px; }
.log-query { margin-top: auto; min-height: 39px; padding: 8px 12px; background: linear-gradient(#075183, #007dcc); border-color: #1caefb; color: #fff; font-weight: 600; }
.log-query:hover:not(:disabled), .log-pagination button:hover:not(:disabled) { border-color: #6ee6ff; background: #11547b; }
.log-query:disabled, .log-pagination button:disabled { opacity: .5; cursor: default; }
.log-results { display: grid; grid-template-rows: minmax(108px, auto) minmax(0, 1fr) minmax(0, 1.13fr); min-height: 0; gap: 12px; }
.log-analysis, .log-details { display: grid; grid-template-columns: minmax(0, 1.27fr) minmax(0, 1fr); min-height: 0; gap: 12px; }
.log-details { grid-template-columns: minmax(0, 1.31fr) minmax(0, 1fr); }
.log-scroll { overflow: auto; min-height: 0; }
.summary-table { margin-bottom: 7px; }
.log-table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; text-align: center; white-space: nowrap; }
.log-table thead { background: #0b335a; color: #c9dffa; }
.log-table th, .log-table td { padding: 7px 8px; border-bottom: 1px solid #144b74; }
.log-table th { font-weight: 600; }
.log-table tbody tr:nth-child(odd) { background: #082342; }
.log-table tbody tr:hover { background: #0a3354; }
.summary-table td { font-size: 20px; font-weight: 600; color: #dcecff; }
.summary-table th:first-child { text-align: left; padding-left: 16px; }
.log-bars { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); flex: 1; min-height: 0; overflow: auto; padding: 16px 0; align-items: start; }
.log-bar-column { display: flex; flex-direction: column; padding: 0 12px; height: 100%; }
.log-bar-column + .log-bar-column { border-left: 1px solid #144b74; }
.log-bar-column h3 { font-size: 14px; margin-bottom: 7px; text-align: center; }
.log-bar-column h3 small { color: #b8d2eb; font-weight: 400; }
.log-bar-group { display: flex; justify-content: center; flex: 1; min-height: 0; }
.log-bar-item { display: flex; flex: 1; flex-direction: column; max-width: 100px; min-width: 0; text-align: center; color: var(--model-color); font-size: 11px; }
.log-bar-item > span { min-height: 2.6em; padding: 5px 2px 0; }
.log-bar-track { position: relative; flex: 1; border-bottom: 1px solid #28638f; }
.log-bar-track b { position: absolute; bottom: var(--bar-height); left: 50%; transform: translateX(-50%); padding-bottom: 4px; color: #e2eeff; font-size: 13px; font-variant-numeric: tabular-nums; }
.log-bar-track > span { position: absolute; bottom: 0; left: 18%; width: 64%; height: var(--bar-height); border-radius: 3px 3px 0 0; background: var(--model-color); }
.log-results .monitor-title select { padding: 3px 5px; font-size: 12px; }
.log-legend { display: flex; justify-content: center; flex-wrap: wrap; gap: 6px 16px; padding: 6px 10px 9px; font-size: 12px; }
.detail-scroll { flex: 1; }
.detail-table thead { position: sticky; top: 0; }
.detail-table { font-size: 12px; }
.detail-table th, .detail-table td { height: 29px; padding: 4px 5px; }
.log-pagination { display: flex; align-items: center; justify-content: flex-end; gap: 10px; padding: 7px 10px; font-size: 12px; color: #b8d2eb; border-top: 1px solid #144b74; }
.log-pagination button { padding: 4px 8px; }
.log-empty { display: grid; place-items: center; flex: 1; padding: 14px; color: #adc9e4; text-align: center; font-size: 13px; }
@media (max-width: 1700px) {
  .training-logs { height: calc(100vh - 147px); }
  .training-logs .monitor-title { font-size: 16px; }
  .log-filter-form { gap: 19px; padding: 14px 11px; }
  .log-bar-column { padding: 0 9px; }
}
@media (max-width: 1200px), (max-height: 790px) {
  .training-logs { height: auto; }
  .log-results { grid-template-rows: auto minmax(250px, auto) minmax(280px, auto); }
}
@media (max-width: 850px) {
  .training-logs, .log-analysis, .log-details { grid-template-columns: minmax(0, 1fr); }
  .log-bars { min-height: 180px; }
  .log-chart { min-height: 210px; }
}
</style>
