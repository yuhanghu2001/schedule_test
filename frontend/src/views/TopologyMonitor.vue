<script setup>
import { ref, computed, onMounted } from 'vue'
import api from '../api/client'
import { useAppStore } from '../stores/app'
import NetworkMonitorScene from '../components/NetworkMonitorScene.vue'
import NodeResourcePanel from '../components/NodeResourcePanel.vue'
const app = useAppStore()
const snapshot = ref(null),
  busy = ref(false),
  selection = ref(null),
  drawer = ref(false)
const selected = computed(() => {
  if (!selection.value) return null
  const { kind, id } = selection.value
  return kind === 'terminal'
    ? { ...snapshot.value.terminals.find((n) => n.id === id), kind }
    : snapshot.value.nodes.find((n) => n.id === id)
})
async function preview() {
  busy.value = true
  try {
    if (!app.env) await app.loadPresets()
    snapshot.value = await api.post('/env/preview', app.env)
  } finally {
    busy.value = false
  }
}
function inspect(node) {
  selection.value = { kind: node.kind, id: node.id }
  drawer.value = true
}
onMounted(preview)
</script>
<template>
  <section
    class="topology-monitor"
    aria-label="拓扑监控大屏"
    v-loading="!snapshot && busy"
    element-loading-background="#06152f"
  >
    <div class="monitor-scene-wrap">
      <NetworkMonitorScene :snapshot="snapshot" @select-node="inspect" />
    </div>
    <el-drawer
      v-model="drawer"
      :title="selected?.kind === 'terminal' ? '端侧设备详情' : '计算节点详情'"
      size="420px"
      ><template v-if="selected?.kind === 'terminal'"
        ><h3>U{{ selected.id }} · 端侧任务源</h3>
        <p class="terminal-details">
          接入 E{{ selected.access_edge }}<br />实际无线速率
          {{
            selected.radio_rate_mbps.toFixed(2)
          }}
          Mbps<br />端侧仅产生任务与无线收发
        </p></template
      ><NodeResourcePanel
        v-else
        :node="selected"
        :nodes="snapshot?.nodes || []"
        @select-node="inspect"
    /></el-drawer>
  </section>
</template>
