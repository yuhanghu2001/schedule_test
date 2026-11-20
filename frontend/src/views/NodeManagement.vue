<script setup>
import { ref, computed, watch, onMounted, toRaw } from 'vue'
import { ElMessage } from 'element-plus'
import { useAppStore } from '../stores/app'
import api from '../api/client'
import NodeGlyph from '../components/NodeGlyph.vue'
const app = useAppStore()
const nodes = ref([]),
  snapshot = ref(null),
  config = ref(null),
  selected = ref(null),
  type = ref('all'),
  search = ref(''),
  table = ref(null),
  saving = ref(false),
  loading = ref(false)
const resources = ['CPU', 'NPU', 'FPGA'],
  services = ['图像识别', '目标检测', '视频分析', '语音处理', '信号计算']
const types = [
  ['all', '全部节点'],
  ['terminal', '端侧节点'],
  ['edge', '边缘节点'],
  ['cloud', '远端节点'],
]
const label = (n) =>
  n.kind === 'cloud' ? 'C0' : (n.kind === 'terminal' ? 'U' : 'E') + n.id
const kindName = (n) =>
  ({ terminal: '端侧任务源', edge: '边缘计算节点', cloud: '远端云节点' })[
    n.kind
  ]
const filtered = computed(() =>
  nodes.value.filter(
    (n) =>
      (type.value === 'all' || n.kind === type.value) &&
      `${label(n)} ${kindName(n)} ${n.profile || ''}`
        .toLowerCase()
        .includes(search.value.toLowerCase()),
  ),
)
watch([type, search], () => table.value.setScrollTop(0))
const neighbors = computed(() => {
  if (!selected.value || !snapshot.value) return []
  const n = selected.value
  if (n.kind === 'terminal') return ['E' + n.access_edge]
  return [
    ...snapshot.value.links
      .filter(
        (l) =>
          l.status === 'active' && (l.source === n.id || l.target === n.id),
      )
      .map((l) => {
        const id = l.source === n.id ? l.target : l.source
        return id === config.value.n_edges ? 'C0' : 'E' + id
      }),
    ...snapshot.value.terminals
      .filter((t) => t.access_edge === n.id)
      .map((t) => 'U' + t.id),
  ]
})
async function load() {
  loading.value = true
  try {
    if (!Object.keys(app.presets).length) await app.loadPresets()
    config.value = structuredClone(
      toRaw({
        ...toRaw(app.presets[app.env.preset_name]),
        ...toRaw(app.env.config),
      }),
    )
    snapshot.value = await api.post('/env/preview', {
      preset_name: app.env.preset_name,
      config: config.value,
    })
    nodes.value = [
      ...snapshot.value.nodes.map((n) => ({
        ...n,
        speed: [...n.speed],
        services: [...n.services],
      })),
      ...snapshot.value.terminals.map((n) => ({ ...n, kind: 'terminal' })),
    ]
    selected.value = nodes.value.find((n) => n.kind === 'edge')
  } finally {
    loading.value = false
  }
}
function edit(node) {
  selected.value = node
}
async function save() {
  saving.value = true
  try {
    const compute = nodes.value.filter((n) => n.kind !== 'terminal'),
      edges = compute.filter((n) => n.kind === 'edge')
    config.value.edge_speed_matrix = edges.map((n) => [...n.speed])
    config.value.cloud_speed = [
      ...compute.find((n) => n.kind === 'cloud').speed,
    ]
    config.value.service_deployment = edges.map((n) =>
      services.map((_, i) => Number(n.services.includes(i))),
    )
    config.value.node_memory_gb = compute.map((n) => n.mem_cap)
    config.value.node_storage_gb = compute.map((n) => n.storage_cap)
    config.value.node_queue_capacity = compute.map((n) => n.queue_cap)
    config.value.terminal_radio_rates_mbps = nodes.value
      .filter((n) => n.kind === 'terminal')
      .map((n) => n.radio_rate_mbps)
    const environment = {
      preset_name: app.env.preset_name,
      config: structuredClone(toRaw(config.value)),
    }
    snapshot.value = await api.post('/env/preview', environment)
    app.env = environment
    ElMessage.success('配置已保存，将用于新建推演和训练环境')
  } finally {
    saving.value = false
  }
}
onMounted(load)
</script>
<template>
  <div
    class="node-management"
    v-loading="loading"
    element-loading-background="#06152f"
  >
    <section class="monitor-panel node-config-list">
      <h2 class="monitor-title">节点列表与属性配置</h2>
      <div class="node-config-toolbar">
        <div class="node-type-tabs">
          <button
            v-for="t in types"
            :key="t[0]"
            :class="{ active: type === t[0] }"
            @click="type = t[0]"
          >
            {{ t[1] }}（{{
              nodes.filter((n) => t[0] === 'all' || n.kind === t[0]).length
            }}）
          </button>
        </div>
        <el-input
          v-model="search"
          placeholder="搜索节点 ID、类型或资源"
          clearable
        /><el-button :disabled="saving || loading" @click="load"
          >恢复已保存配置</el-button
        ><el-button
          type="primary"
          :loading="saving"
          :disabled="loading"
          @click="save"
          >保存配置</el-button
        >
      </div>
      <div class="node-config-table">
        <el-table
          ref="table"
          :data="filtered"
          height="100%"
          scrollbar-always-on
          border
          highlight-current-row
          @row-click="edit"
          empty-text="没有匹配的节点"
          ><el-table-column label="选择" width="54" align="center"
            ><template #default="{ row }"
              ><span :class="['row-selector', { checked: selected === row }]">{{
                selected === row ? '✓' : ''
              }}</span></template
            ></el-table-column
          ><el-table-column label="节点 ID" width="86"
            ><template #default="{ row }"
              ><b>{{ label(row) }}</b></template
            ></el-table-column
          ><el-table-column label="节点类型" width="115"
            ><template #default="{ row }">{{
              kindName(row)
            }}</template></el-table-column
          ><el-table-column label="状态" width="92"
            ><template #default
              ><span class="good">可用</span></template
            ></el-table-column
          ><el-table-column
            v-for="(r, i) in resources"
            :key="r"
            :label="r + ' 算力'"
            min-width="96"
            ><template #default="{ row }">{{
              row.speed ? row.speed[i].toLocaleString() : '—'
            }}</template></el-table-column
          ><el-table-column label="内存容量" min-width="100"
            ><template #default="{ row }">{{
              row.speed ? row.mem_cap + ' GB' : '—'
            }}</template></el-table-column
          ><el-table-column label="存储容量" min-width="110"
            ><template #default="{ row }">{{
              row.speed ? row.storage_cap.toLocaleString() + ' GB' : '—'
            }}</template></el-table-column
          ><el-table-column label="队列 / 无线速率" min-width="135"
            ><template #default="{ row }">{{
              row.speed
                ? row.queue_cap + ' 个任务'
                : row.radio_rate_mbps.toFixed(1) + ' Mbps'
            }}</template></el-table-column
          ><el-table-column label="操作" width="80" fixed="right"
            ><template #default="{ row }"
              ><el-button size="small" @click.stop="edit(row)"
                >编辑</el-button
              ></template
            ></el-table-column
          ></el-table
        >
      </div>
    </section>
    <aside class="node-config-detail">
      <section class="monitor-panel">
        <h2 class="monitor-title">节点详情</h2>
        <div v-if="selected" class="node-details-body">
          <div class="node-detail-identity">
            <NodeGlyph :kind="selected.kind" />
            <h3>
              {{ label(selected) }} <small>（{{ kindName(selected) }}）</small>
            </h3>
            <span class="good">可用</span>
          </div>
          <dl>
            <div>
              <dt>节点类型</dt>
              <dd>{{ kindName(selected) }}</dd>
            </div>
            <div>
              <dt>所属环境</dt>
              <dd>{{ app.env?.preset_name }}</dd>
            </div>
            <div>
              <dt>接入 / 邻接</dt>
              <dd>{{ neighbors.join('、') || '无' }}</dd>
            </div>
            <div v-if="selected.speed">
              <dt>主导资源</dt>
              <dd>
                {{
                  selected.kind === 'cloud'
                    ? '云端异构资源'
                    : resources[
                        selected.speed.indexOf(Math.max(...selected.speed))
                      ]
                }}
              </dd>
            </div>
          </dl>
        </div>
      </section>
      <section class="monitor-panel node-edit-panel">
        <h2 class="monitor-title">
          {{ selected?.kind === 'terminal' ? '端侧属性配置' : '算力资源配置' }}
        </h2>
        <el-form
          v-if="selected"
          label-position="top"
          class="node-edit-form"
        >
          <template v-if="selected.speed"
            ><label v-for="(r, i) in resources" :key="r" class="node-edit-field"
              ><span :class="'resource-name-' + i"
                >{{ r }} <small>GFLOP/s</small></span
              ><el-input-number
                v-model="selected.speed[i]"
                :min="1"
                :controls="false"
                :aria-label="r + ' 算力'" /></label
            ><label
              v-for="f in [
                ['mem_cap', '内存', 'GB'],
                ['storage_cap', '存储', 'GB'],
                ['queue_cap', '队列上限', '个任务'],
              ]"
              :key="f[0]"
              class="node-edit-field"
              ><span
                >{{ f[1] }} <small>{{ f[2] }}</small></span
              ><el-input-number
                v-model="selected[f[0]]"
                :min="1"
                :precision="f[0] === 'queue_cap' ? 0 : 2"
                :controls="false"
                :aria-label="f[1]"
            /></label>
            <div class="node-services">
              <h4>部署服务</h4>
              <el-checkbox-group
                v-model="selected.services"
                :disabled="selected.kind === 'cloud'"
                ><el-checkbox
                  v-for="(s, i) in services"
                  :key="i"
                  :label="s"
                  :value="i"
              /></el-checkbox-group></div
          ></template>
          <template v-else
            ><label class="node-edit-field"
              ><span>无线参考速率 <small>Mbps</small></span
              ><el-input-number
                v-model="selected.radio_rate_mbps"
                :min="1"
                :controls="false"
                aria-label="无线参考速率"
            /></label>
            <p class="node-config-note">
              接入 E{{
                selected.access_edge
              }}。此速率用于该终端任务的上传、回传与时限计算。端侧没有可分配的本地算力。
            </p></template
          >
          <el-button type="primary" :loading="saving" @click="save"
            >保存配置并应用于新环境</el-button
          ></el-form
        >
      </section>
    </aside>
  </div>
</template>
