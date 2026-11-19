<script setup>
import { ref, watch, onMounted, onBeforeUnmount } from 'vue'
import * as echarts from 'echarts'
const props = defineProps({ samples: Array })
const el = ref(null)
let chart, observer
function render() {
  chart.setOption({
    color: ['#40bcff', '#51dfb3', '#ffbf63'],
    tooltip: { trigger: 'axis', valueFormatter: (v) => v.toFixed(1) + '%' },
    legend: {
      top: 0,
      right: 0,
      textStyle: { color: '#c6dcf6', fontSize: 11 },
      itemWidth: 12,
      itemHeight: 6,
    },
    grid: { left: 38, right: 12, top: 35, bottom: 27 },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: props.samples.map((s) => '#' + s.task),
      axisLabel: { color: '#a8c4e3', fontSize: 10 },
      axisLine: { lineStyle: { color: '#1a466d' } },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 100,
      axisLabel: { color: '#a8c4e3', formatter: '{value}%', fontSize: 10 },
      splitLine: { lineStyle: { color: '#12365a' } },
    },
    series: ['CPU', 'NPU', 'FPGA'].map((name, i) => ({
      name,
      type: 'line',
      symbolSize: 4,
      data: props.samples.map((s) => s.values[i]),
      lineStyle: { width: 2 },
      areaStyle: { opacity: 0.04 },
    })),
  })
}
onMounted(() => {
  chart = echarts.init(el.value)
  render()
  observer = new ResizeObserver(() => chart.resize())
  observer.observe(el.value)
})
watch(
  () => props.samples,
  () => {
    if (chart) render()
  },
  { deep: true },
)
onBeforeUnmount(() => {
  observer.disconnect()
  chart.dispose()
})
</script>
<template>
  <div
    ref="el"
    class="resource-trend"
    aria-label="资源利用率随已调度任务变化的折线图"
  ></div>
</template>
