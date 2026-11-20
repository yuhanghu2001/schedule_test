<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({ option: Object, label: String })
const canvas = ref(null)
let chart
let observer

function render() {
  chart.setOption(props.option, true)
}

onMounted(() => {
  chart = echarts.init(canvas.value)
  render()
  observer = new ResizeObserver(() => chart.resize())
  observer.observe(canvas.value)
})
watch(() => props.option, render)
onBeforeUnmount(() => {
  observer.disconnect()
  chart.dispose()
})
</script>

<template>
  <div ref="canvas" class="log-chart" role="img" :aria-label="label"></div>
</template>

<style scoped>
.log-chart { width: 100%; flex: 1; min-height: 0; }
</style>
