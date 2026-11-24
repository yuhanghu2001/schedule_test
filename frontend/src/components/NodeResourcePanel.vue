<script setup>
import { computed } from 'vue'
const props = defineProps({ node: Object, nodes: { type: Array, default: () => [] } })
const emit = defineEmits(['select-node'])
const node = computed(() => props.node?.speed ? props.node : props.nodes.find(n => n.kind === 'edge'))
const resources = ['CPU', 'NPU', 'FPGA']
const operators = ['CONTROL_PARSE', 'STREAM_FILTER_ROI', 'FFT_FIR', 'TENSOR_INFER', 'AGGREGATE_TRACK']
const tiers = { gateway: '网关', middle: '中层', access: '接入', cloud: '云' }
const pct = (v, max) => max ? Math.min(100, 100 * v / max) : 0
const amount = v => Number(v).toFixed(1)
const capacities = computed(() => node.value?.tier ? [
  ['CPU 核数', 'cpu_cores', '核'], ['CPU 内存', 'ram_gb', 'GB'],
  ['NPU 显存', 'npu_vram_gb', 'GB'], ['FPGA LUT', 'fpga_lut', ''],
  ['FPGA FF', 'fpga_ff', ''], ['FPGA BRAM', 'fpga_bram', ''], ['FPGA DSP', 'fpga_dsp', ''],
] : [])
</script>
<template>
  <div class="resource-panel">
    <div class="panel-heading"><h3>节点资源</h3></div>
    <el-select v-if="nodes.length" :model-value="node?.id" size="small" @change="id => emit('select-node', nodes.find(n => n.id === id))"><el-option v-for="n in nodes" :key="n.id" :label="n.label" :value="n.id" /></el-select>
    <div v-if="node">
      <p><strong>{{ node.label }}</strong> {{ tiers[node.tier] || (node.kind === 'cloud' ? '云' : '边缘') }}</p>
      <div class="minor-label">计算算力余量（GFLOP/s）</div>
      <div v-for="(name, i) in resources" :key="name" class="resource-row">
        <div class="resource-label"><b>{{ name }}</b><span>{{ amount(node.speed[i] - node.resource_used[i]) }} / {{ amount(node.speed[i]) }}</span></div>
        <div class="bar"><i class="new" :style="{ width: pct(node.speed[i] - node.resource_used[i], node.speed[i]) + '%' }"></i></div>
      </div>
      <div v-for="r in capacities" :key="r[1]" class="water"><span>{{ r[0] }}</span><div class="bar"><i class="new" :style="{ width: pct(node[r[1] + '_free'], node[r[1] + '_cap']) + '%' }"></i></div><small>{{ amount(node[r[1] + '_free']) }} / {{ amount(node[r[1] + '_cap']) }} {{ r[2] }}</small></div>
      <template v-if="node.tier"><p>NPU 驻留算子：{{ node.npu_resident_op < 0 ? '无' : operators[node.npu_resident_op] }}</p><p>FPGA 驻留算子：{{ node.fpga_resident_op < 0 ? '无' : operators[node.fpga_resident_op] }}</p></template>
      <template v-else><div v-for="r in [['内存',node.mem_used,node.mem_cap,'GB'],['存储',node.storage_used,node.storage_cap,'GB'],['队列',node.queue_used,node.queue_cap,'']]" :key="r[0]" class="water"><span>{{ r[0] }}</span><div class="bar"><i class="new" :style="{width:pct(r[1],r[2])+'%'}"></i></div><small>{{ amount(r[1]) }} / {{ r[2] }} {{ r[3] }}</small></div></template>
      <p>已部署服务：{{ node.services.map(s => ['图像识别','目标检测','视频分析','语音处理','信号计算'][s]).join('、') || '无' }}</p><p>在途任务：{{ node.active_task_ids.length ? node.active_task_ids.map(id => '#' + id).join('、') : '无' }}</p>
    </div><el-empty v-else description="选择节点查看详情" :image-size="60" />
  </div>
</template>
