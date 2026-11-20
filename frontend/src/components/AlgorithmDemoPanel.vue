<script setup>
import { onBeforeUnmount, ref } from 'vue'
import AlgorithmDemoCard from './AlgorithmDemoCard.vue'

defineProps({ modelValue: String, disabled: Boolean, status: String })
const emit = defineEmits(['update:modelValue'])
const algorithms = [
  { id: 'shortest_fastest', tab: '最短最快', title: '① 最短最快启发式' },
  { id: 'greedy_estimate', tab: '贪心估计', title: '② 贪心估计启发式' },
  { id: 'random', tab: '随机算法', title: '③ 随机算法' },
]
const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)')
const reducedMotion = ref(motionPreference.matches)
const autoplay = ref(!reducedMotion.value)
function changeMotion(event) {
  reducedMotion.value = event.matches
  if (event.matches) autoplay.value = false
}
function moveTab(event, index) {
  const offset = { ArrowRight: 1, ArrowLeft: -1, Home: -index, End: 2 - index }[event.key]
  if (offset === undefined) return
  event.preventDefault()
  const next = (index + offset + algorithms.length) % algorithms.length
  emit('update:modelValue', algorithms[next].id)
  event.currentTarget.parentElement.children[next].focus()
}
motionPreference.addEventListener('change', changeMotion)
onBeforeUnmount(() => motionPreference.removeEventListener('change', changeMotion))
</script>

<template>
  <section class="monitor-panel algorithm-demo-panel" aria-labelledby="algorithm-demo-title">
    <h2 id="algorithm-demo-title" class="monitor-title">调度策略 · 算法原理演示<small>{{ status }} · 原理示意</small></h2>
    <div class="algorithm-demo-body">
      <div class="demo-toolbar">
        <div class="demo-tabs" role="tablist" aria-label="选择实际调度算法">
          <button
            v-for="(algorithm, index) in algorithms"
            :id="`demo-tab-${algorithm.id}`"
            :key="algorithm.id"
            role="tab"
            :aria-selected="modelValue === algorithm.id"
            :aria-controls="`demo-${algorithm.id}`"
            :tabindex="modelValue === algorithm.id || (!modelValue && index === 0) ? 0 : -1"
            :disabled="disabled"
            @click="emit('update:modelValue', algorithm.id)"
            @keydown="moveTab($event, index)"
          >{{ algorithm.tab }}</button>
        </div>
        <label class="demo-autoplay"><input v-model="autoplay" type="checkbox" />自动轮播</label>
      </div>
      <div class="demo-cards">
        <AlgorithmDemoCard
          v-for="algorithm in algorithms"
          :id="`demo-${algorithm.id}`"
          :key="algorithm.id"
          :algorithm="algorithm.id"
          :title="algorithm.title"
          :selected="modelValue === algorithm.id"
          :autoplay="autoplay"
          :reduced-motion="reducedMotion"
        />
      </div>
      <div class="demo-common-flow" aria-label="公共调度流程">
        <template v-for="(step, index) in ['任务到达', '生成可行方案', '算法决策', '执行反馈']" :key="step">
          <span v-if="index" class="demo-flow-arrow" aria-hidden="true">→</span><span>{{ step }}</span>
        </template>
      </div>
    </div>
  </section>
</template>

<style scoped>
.algorithm-demo-panel { flex: 1; }
.algorithm-demo-panel > .monitor-title { font-size: 18px; letter-spacing: 0; }
.algorithm-demo-body { padding: 10px; display: flex; flex: 1; min-height: 0; flex-direction: column; gap: 8px; }
.demo-toolbar { display: flex; gap: 10px; align-items: center; }
.demo-tabs { display: flex; flex: 1; min-width: 0; border: 1px solid #22638d; border-radius: 5px; overflow: hidden; }
.demo-tabs button { flex: 1; padding: 6px 0; border: 0; border-right: 1px solid #22638d; background: #082342; color: #c6dbf4; font-size: 13px; white-space: nowrap; }
.demo-tabs button:last-child { border-right: 0; }
.demo-tabs button[aria-selected='true'] { background: #0068a6; color: #fff; box-shadow: inset 0 0 7px #008cff; }
.demo-tabs button:hover:not(:disabled) { background: #10466e; }
.demo-tabs button:disabled { cursor: not-allowed; opacity: .65; }
.demo-tabs button:focus-visible { outline-offset: -3px; }
.demo-autoplay { display: flex; align-items: center; gap: 4px; white-space: nowrap; color: #c6dbf4; font-size: 12px; cursor: pointer; }
.demo-autoplay input { margin: 0; accent-color: #008cff; }
.algorithm-demo-panel > .monitor-title small { color: #abc8e2; }
.demo-cards { flex: 1; min-height: 0; display: grid; grid-template-rows: repeat(3, minmax(0, 1fr)); gap: 8px; }
.demo-common-flow { display: flex; justify-content: space-between; align-items: center; gap: 3px; font-size: 12px; line-height: 25px; white-space: nowrap; }
.demo-common-flow > span:not(.demo-flow-arrow) { padding: 0 4px; background: #082342; border: 1px solid #15577d; border-radius: 4px; }
.demo-flow-arrow { color: #6ee6ff; }
@media (max-width: 1700px) {
  .algorithm-demo-panel > .monitor-title { font-size: 16px; }
  .algorithm-demo-body { padding: 7px; gap: 6px; }
  .demo-toolbar { gap: 7px; }
  .demo-tabs button { font-size: 12px; padding: 5px 0; }
  .demo-autoplay, .demo-common-flow { font-size: 11px; }
  .demo-cards { gap: 6px; }
  .demo-common-flow { line-height: 22px; }
  .demo-common-flow > span:not(.demo-flow-arrow) { padding: 0 3px; }
}
@media (max-width: 850px) { .algorithm-demo-panel { flex: auto; min-height: 650px; } }
</style>
