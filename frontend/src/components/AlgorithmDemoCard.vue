<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps({ algorithm: String, title: String, selected: Boolean, autoplay: Boolean, reducedMotion: Boolean })
// 固定教学样例，不读取或写入真实 session。每个节点仅展示一个已通过约束的资源/路径组合。
// 对应 src/baselines.py；收益使用 src/config.py 默认 beta=(2,1,1)、t_ref=1s、E_ref=3J。
const candidates = [
  { name: 'E1', resource: 'CPU', hops: 1, compute: 80, delay: .7, energy: 1.5, y: 14, points: [[54, 42], [84, 42], [192, 14]] },
  { name: 'E2', resource: 'NPU', hops: 1, compute: 120, delay: .9, energy: 1.2, y: 42, points: [[54, 42], [84, 42], [192, 42]] },
  { name: 'E3', resource: 'FPGA', hops: 3, compute: 180, delay: .5, energy: .6, y: 70, points: [[54, 42], [84, 42], [114, 70], [151, 70], [192, 70]] },
].map(candidate => ({ ...candidate, score: 2 - candidate.delay / 1 - candidate.energy / 3 }))
const shortest = [...candidates].sort((a, b) => a.hops - b.hops || b.compute - a.compute || a.delay - b.delay || a.energy - b.energy)[0]
const greedy = [...candidates].sort((a, b) => b.score - a.score)[0]
const elapsed = ref(0)
const playing = ref(props.autoplay)
const randomChoice = ref(Math.floor(Math.random() * 4))
const duration = 6000
let frame = 0
let previousTime = null
const phase = computed(() => elapsed.value < 1400 ? 0 : elapsed.value < 2600 ? 1 : elapsed.value < 3600 ? 2 : elapsed.value < 4500 ? 3 : 4)
const scanning = computed(() => Math.floor(elapsed.value / (props.algorithm === 'random' ? 140 : 430)) % (props.algorithm === 'random' ? 4 : 3))
const chosen = computed(() => props.algorithm === 'shortest_fastest' ? candidates.indexOf(shortest) : props.algorithm === 'greedy_estimate' ? candidates.indexOf(greedy) : randomChoice.value)
const resultName = computed(() => chosen.value === 3 ? '拒绝' : candidates[chosen.value].name)
const stages = computed(() => {
  if (props.algorithm === 'shortest_fastest') return ['搜索可行路径', '比较跳数', '同跳比较分配算力', `选中 ${resultName.value}`, '传输任务']
  if (props.algorithm === 'greedy_estimate') return ['枚举可行方案', '估计时延与能耗', '比较归一化收益', `最高收益 ${resultName.value}`, '传输任务']
  return ['任务到达', '扫描可行动作', '等概率抽取', `抽中 ${resultName.value}`, chosen.value === 3 ? '本次拒绝，不传输' : '传输任务']
})
const stageLabels = computed(() => ({
  shortest_fastest: ['搜索路径', '比较跳数', '同跳比算力', `选中 ${resultName.value}`, '传输任务'],
  greedy_estimate: ['枚举方案', '估计指标', '比较收益', `选中 ${resultName.value}`, '传输任务'],
  random: ['任务到达', '扫描动作', '随机抽取', `抽中 ${resultName.value}`, chosen.value === 3 ? '拒绝不传输' : '传输任务'],
})[props.algorithm])
const explanation = computed(() => ({
  shortest_fastest: '先比跳数，再比算力，最后比较时延与能耗。',
  greedy_estimate: '估计时延与能耗，选择当前综合收益最高的方案。',
  random: '从可行方案及拒绝动作中等概率抽取。',
})[props.algorithm])
const formula = computed(() => props.algorithm === 'shortest_fastest'
  ? '排序：跳数少 → 算力大 → 时延低 → 能耗低；跳数不含端侧无线接入。算力单位 GFLOP/s。'
  : props.algorithm === 'greedy_estimate'
    ? '默认配置示例：收益 = 2 − 时延 / 1s − 能耗 / 3J；评分条按收益 / 2 显示，越长越优。'
    : '三个预设可行方案与拒绝动作各占 1/4；真实系统按全部可行动作（含拒绝）等概率抽取，并非按节点均匀抽取。')
function path(candidate) {
  return candidate.points.map(([x, y]) => `${x},${y}`).join(' ')
}
// 按折线路程插值，让数据包沿实际绘制路径前进；同一时钟控制全部图形，暂停时保留当前帧。
function pointOnPath(points, progress) {
  const lengths = points.slice(1).map(([x, y], index) => Math.hypot(x - points[index][0], y - points[index][1]))
  let remaining = lengths.reduce((sum, length) => sum + length, 0) * progress
  for (let index = 0; index < lengths.length; index++) {
    if (remaining <= lengths[index] || index === lengths.length - 1) {
      const ratio = remaining / lengths[index]
      return { x: points[index][0] + (points[index + 1][0] - points[index][0]) * ratio, y: points[index][1] + (points[index + 1][1] - points[index][1]) * ratio }
    }
    remaining -= lengths[index]
  }
}
const packet = computed(() => chosen.value === 3 ? null : pointOnPath(candidates[chosen.value].points, props.reducedMotion ? 1 : Math.min(1, (elapsed.value - 4500) / 1100)))
const searchPacket = computed(() => scanning.value === 3 ? null : pointOnPath(candidates[scanning.value].points, props.reducedMotion ? .5 : (elapsed.value % 430) / 430))
function active(index) {
  return phase.value >= 3 ? chosen.value === index : scanning.value === index
}
function scoreProgress(index) {
  return props.reducedMotion ? Number(phase.value >= 2) : Math.min(1, Math.max(0, (elapsed.value - 1400 - index * 300) / 650))
}
function tick(time) {
  if (previousTime !== null) elapsed.value = Math.min(duration, elapsed.value + time - previousTime)
  previousTime = time
  if (elapsed.value === duration) {
    if (!props.autoplay) {
      playing.value = false
      return
    }
    elapsed.value = 0
    randomChoice.value = Math.floor(Math.random() * 4)
  }
  frame = requestAnimationFrame(tick)
}
function syncClock() {
  cancelAnimationFrame(frame)
  previousTime = null
  if (playing.value && !document.hidden) frame = requestAnimationFrame(tick)
}
function restart() {
  elapsed.value = 0
  randomChoice.value = Math.floor(Math.random() * 4)
  playing.value = true
  syncClock()
}
function toggle() {
  if (!playing.value && elapsed.value === duration) restart()
  else playing.value = !playing.value
}
watch(playing, syncClock)
watch(() => props.autoplay, value => { playing.value = value })
watch(() => props.selected, value => { if (value) restart() })
onMounted(() => {
  syncClock()
  document.addEventListener('visibilitychange', syncClock)
})
onBeforeUnmount(() => {
  cancelAnimationFrame(frame)
  document.removeEventListener('visibilitychange', syncClock)
})
</script>

<template>
  <article class="algorithm-demo-card" :class="{ selected }" :data-algorithm="algorithm" :data-phase="phase + 1" :data-playing="playing" :data-choice="resultName" :aria-labelledby="`demo-heading-${algorithm}`">
    <header class="demo-card-header">
      <h3 :id="`demo-heading-${algorithm}`">{{ title }}</h3>
      <div class="demo-playback">
        <button :aria-label="`${playing ? '暂停' : '播放'}${title}演示`" @click="toggle">{{ playing ? '暂停' : '播放' }}</button>
        <button :aria-label="`重新播放${title}演示`" @click="restart">重播</button>
      </div>
    </header>
    <svg class="demo-canvas" viewBox="0 0 360 92" role="img" :aria-label="`${title}，原理示意：${stages[phase]}。${formula}`">
      <title>{{ title }} · 原理示意</title>
      <desc>{{ formula }}</desc>
      <defs>
        <marker :id="`demo-arrow-${algorithm}`" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L8 4 L0 8 Z" fill="context-stroke" /></marker>
      </defs>
      <g v-for="(candidate, index) in candidates" :key="candidate.name" :class="['demo-option', { searching: phase < 3 && active(index), chosen: phase >= 3 && chosen === index, dimmed: phase >= 3 && chosen !== index }]">
        <polyline class="demo-path" :points="path(candidate)" :marker-end="`url(#demo-arrow-${algorithm})`" />
        <g v-if="algorithm === 'shortest_fastest'">
          <circle v-for="(point, relay) in candidate.points.slice(1, -1)" :key="relay" :cx="point[0]" :cy="point[1]" r="3" class="demo-relay" />
          <text x="112" :y="candidate.y + (index === 0 ? 3 : 16)" class="demo-route-label">{{ phase >= 1 ? `${candidate.hops}跳` : '搜索' }}<tspan v-if="phase >= 2 && candidate.hops === 1"> / {{ candidate.compute }}</tspan></text>
        </g>
        <rect x="194" :y="candidate.y - 12" width="42" height="24" rx="4" class="demo-server" />
        <text x="215" :y="candidate.y + 4" class="demo-server-label">{{ candidate.name }}</text>
        <template v-if="algorithm === 'shortest_fastest'">
          <text x="244" :y="candidate.y + 2" class="demo-metric">{{ candidate.resource }} · {{ candidate.compute }}</text>
          <text x="244" :y="candidate.y + 16" class="demo-metric secondary">{{ phase >= 1 ? `${candidate.delay.toFixed(1)}s / ${candidate.energy.toFixed(1)}J` : '待比较' }}</text>
        </template>
        <template v-else-if="algorithm === 'greedy_estimate'">
          <rect x="244" :y="candidate.y - 10" width="72" height="8" rx="3" class="demo-score-track" />
          <rect x="244" :y="candidate.y - 10" :width="72 * candidate.score / 2 * scoreProgress(index)" height="8" rx="3" class="demo-score-bar" />
          <text x="323" :y="candidate.y + 2" class="demo-score">{{ scoreProgress(index) === 1 ? candidate.score.toFixed(2) : '…' }}</text>
          <text x="244" :y="candidate.y + 13" class="demo-metric secondary">{{ scoreProgress(index) > 0 ? `${candidate.delay.toFixed(1)}s / ${candidate.energy.toFixed(1)}J` : '时延 / 能耗' }}</text>
        </template>
        <text v-else x="247" :y="candidate.y + 4" class="demo-metric">{{ phase >= 3 && chosen === index ? '抽中执行' : '可行 · 1/4' }}</text>
      </g>
      <g class="demo-task">
        <circle cx="32" cy="42" r="19" />
        <text x="32" y="39">任务</text><text x="32" y="54">T1</text>
      </g>
      <text v-if="algorithm === 'shortest_fastest'" x="84" y="59" class="demo-access">接入</text>
      <template v-if="algorithm === 'random'">
        <rect x="2" y="1" width="70" height="19" rx="4" :class="['demo-draw', { 'draw-reject': phase >= 3 && chosen === 3 }]" />
        <text x="37" y="14" class="demo-draw-label">{{ phase >= 3 ? `抽中 ${resultName}` : `抽取 ${scanning === 3 ? '拒绝' : candidates[scanning].name}` }}</text>
        <text x="103" y="87" class="demo-metric secondary">{{ phase >= 3 && chosen === 3 ? '本次不占用资源' : '另含拒绝动作 · 1/4' }}</text>
      </template>
      <text x="2" y="74" class="demo-stage">{{ stageLabels[phase] }}</text>
      <text x="2" y="87" class="demo-clock">{{ phase + 1 }}/5 · {{ (elapsed / 1000).toFixed(1) }}/6s</text>
      <circle v-if="algorithm === 'shortest_fastest' && phase === 0 && searchPacket" :cx="searchPacket.x" :cy="searchPacket.y" r="3" class="demo-search-packet" />
      <rect v-if="phase === 4 && packet" :x="packet.x - 4" :y="packet.y - 4" width="8" height="8" rx="2" class="demo-packet" />
    </svg>
    <p class="demo-caption" :title="formula">{{ explanation }}</p>
    <div class="demo-progress" role="progressbar" :aria-label="`${title}播放进度`" :aria-valuenow="Math.round(elapsed / duration * 100)" aria-valuemin="0" aria-valuemax="100"><span :style="{ transform: `scaleX(${elapsed / duration})` }" /></div>
  </article>
</template>

<style scoped>
.algorithm-demo-card { position: relative; display: flex; flex-direction: column; min-height: 0; padding: 8px 9px 6px; border: 1px solid #1b527c; border-radius: 7px; background: #061c35; }
.algorithm-demo-card.selected { border-color: #41d9ee; background: #082942; box-shadow: inset 0 0 7px #008cff28; }
.demo-card-header { display: flex; justify-content: space-between; align-items: center; gap: 4px; flex-shrink: 0; }
.demo-card-header h3 { font-size: 15px; line-height: 22px; white-space: nowrap; }
.selected .demo-card-header h3 { color: #6ee6ff; }
.demo-playback { display: flex; gap: 4px; }
.demo-playback button { color: #c6dbf4; background: #082342; border: 1px solid #22638d; border-radius: 4px; padding: 2px 5px; font-size: 11px; line-height: 16px; }
.demo-playback button:hover { color: #fff; border-color: #6ee6ff; background: #10466e; }
.demo-canvas { display: block; width: 100%; flex: 1; min-height: 0; margin: 3px 0; overflow: hidden; }
.demo-canvas text { font-family: inherit; font-variant-numeric: tabular-nums; }
.demo-path { stroke: #5685aa; stroke-width: 1.5; stroke-dasharray: 5 4; fill: none; }
.demo-server { fill: #082342; stroke: #2878a7; stroke-width: 1; }
.demo-server-label { fill: #e2eeff; font-size: 14px; text-anchor: middle; font-weight: 700; }
.demo-relay { fill: #0b294b; stroke: #6eacca; }
.demo-route-label { fill: #c6dbf4; font-size: 12px; paint-order: stroke; stroke: #061c35; stroke-width: 4px; stroke-linejoin: round; }
.demo-metric { fill: #dcecff; font-size: 12px; }
.demo-metric.secondary, .demo-access { fill: #abc8e2; font-size: 11px; }
.demo-access { text-anchor: middle; }
.searching .demo-path, .searching .demo-server { stroke: #6ee6ff; }
.searching .demo-path { stroke-width: 2; }
.chosen .demo-path { stroke: #51dfb3; stroke-width: 2.5; stroke-dasharray: none; filter: drop-shadow(0 0 2px #51dfb366); }
.chosen .demo-server { fill: #06473f; stroke: #51dfb3; stroke-width: 1.5; }
.chosen .demo-server-label, .chosen .demo-score { fill: #76f0ca; }
.dimmed .demo-path { opacity: .35; }
.demo-task circle { fill: #07385b; stroke: #4dcfff; stroke-width: 1.5; }
.demo-task text { fill: #e2eeff; text-anchor: middle; font-size: 13px; font-weight: 700; }
.demo-search-packet { fill: #6ee6ff; }
.demo-packet { fill: #a2ffdc; stroke: #51dfb3; }
.demo-score-track { fill: #153959; }
.demo-score-bar { fill: #46baf5; }
.chosen .demo-score-bar { fill: #51dfb3; }
.demo-score { fill: #dcecff; font-size: 12px; font-weight: 700; }
.demo-draw { fill: #0b3355; stroke: #3fbbe9; }
.demo-draw.draw-reject { stroke: #ffbc6a; }
.demo-draw-label { fill: #dcecff; text-anchor: middle; font-size: 12px; }
.demo-caption { font-size: 12px; line-height: 18px; color: #c6dbf4; flex-shrink: 0; }
.demo-stage { fill: #dcecff; font-size: 11px; }
.demo-clock { fill: #abc8e2; font-size: 10px; }
.demo-progress { position: absolute; bottom: 0; left: 8px; right: 8px; height: 2px; background: #1b527c; overflow: hidden; }
.demo-progress > span { display: block; height: 100%; background: #6ee6ff; transform-origin: left; }
@media (max-width: 1700px) {
  .algorithm-demo-card { padding: 5px 7px 4px; }
  .demo-card-header h3 { font-size: 13px; line-height: 20px; }
  .demo-playback button { padding: 0 4px; }
  .demo-canvas { margin: 2px 0; }
  .demo-caption { font-size: 11px; line-height: 16px; }
}
</style>
