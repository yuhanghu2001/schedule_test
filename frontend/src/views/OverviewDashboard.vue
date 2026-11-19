<script setup>
import { ref, computed, toRaw, onMounted, onBeforeUnmount } from 'vue'
import { useRoute } from 'vue-router'
import { useAppStore } from '../stores/app'
import api from '../api/client'
import TopologyGraph from '../components/TopologyGraph.vue'
import NodeResourcePanel from '../components/NodeResourcePanel.vue'
import TaskFlowTable from '../components/TaskFlowTable.vue'
import NodeGlyph from '../components/NodeGlyph.vue'
import ResourceTrend from '../components/ResourceTrend.vue'
const app = useAppStore(),
  route = useRoute()
const snapshot = ref(null),
  session = ref(null),
  selectedPath = ref(null),
  selected = ref(null),
  drawer = ref(false),
  busy = ref(false),
  playing = ref(false)
const policy = ref('shortest_fastest'),
  activePolicy = ref(null),
  checkpoint = ref(null),
  activeCheckpoint = ref(null),
  preset = ref('draft'),
  config = ref(null),
  samples = ref([])
let playback = 0
const resources = ['CPU', 'NPU', 'FPGA'],
  taskNames = ['图像识别', '目标检测', '视频分析', '语音处理', '信号计算']
const policyNames = {
  shortest_fastest: '最短最快启发式',
  greedy_estimate: '贪心估计启发式',
  random: '随机策略',
  checkpoint: 'Checkpoint 模型',
}
const used = computed(() =>
  resources.map((_, i) =>
    snapshot.value
      ? (snapshot.value.nodes.reduce((sum, n) => sum + n.resource_used[i], 0) /
          snapshot.value.nodes.reduce((sum, n) => sum + n.speed[i], 0)) *
        100
      : 0,
  ),
)
const taskCounts = computed(() =>
  taskNames.map(
    (_, i) =>
      (snapshot.value?.recent_tasks || []).filter((t) => t.kind === i).length,
  ),
)
const completed = computed(
  () =>
    (snapshot.value?.recent_tasks || []).filter(
      (t) => t.accepted && t.completed,
    ).length,
)
const cards = computed(() =>
  snapshot.value
    ? [
        ...snapshot.value.nodes,
        ...snapshot.value.terminals.map((n) => ({ ...n, kind: 'terminal' })),
      ]
    : [],
)
const environment = () =>
  preset.value === 'draft' ? app.env : { preset_name: preset.value, config: {} }
const strategy = computed(() =>
  activePolicy.value === 'checkpoint'
    ? `Checkpoint · ${app.checkpoints.find((c) => c.path === activeCheckpoint.value)?.name || ''}`
    : policyNames[activePolicy.value || policy.value],
)
const fmt = (v, n = 2) => (v == null ? '—' : Number(v).toFixed(n))
const percentage = (node, i) => (100 * node.resource_used[i]) / node.speed[i]
function receive(data, reset = false) {
  snapshot.value = data
  selectedPath.value = null
  if (reset) samples.value = []
  samples.value = [
    ...samples.value,
    { task: data.task_index, values: [...used.value] },
  ].slice(-40)
  if (selected.value?.speed)
    selected.value = data.nodes.find((n) => n.id === selected.value.id)
}
async function preview() {
  pause()
  busy.value = true
  try {
    if (session.value) {
      await api.delete(`/sim/sessions/${session.value}`)
      session.value = null
      activePolicy.value = null
    }
    if (!Object.keys(app.presets).length) await app.loadPresets()
    config.value = structuredClone(
      toRaw({
        ...toRaw(app.presets[environment().preset_name]),
        ...toRaw(environment().config),
      }),
    )
    receive(await api.post('/env/preview', environment()), true)
  } finally {
    busy.value = false
  }
}
async function create() {
  pause()
  busy.value = true
  try {
    if (session.value) await api.delete(`/sim/sessions/${session.value}`)
    session.value = null
    const data = await api.post('/sim/sessions', {
      ...environment(),
      policy: policy.value,
      checkpoint_path: checkpoint.value,
    })
    config.value = structuredClone(
      toRaw({
        ...toRaw(app.presets[environment().preset_name]),
        ...toRaw(environment().config),
      }),
    )
    session.value = data.session_id
    activePolicy.value = policy.value
    activeCheckpoint.value = checkpoint.value
    receive(data.snapshot, true)
  } finally {
    busy.value = false
  }
}
async function step(slot = false) {
  busy.value = true
  try {
    receive(
      await api.post(
        `/sim/sessions/${session.value}/${slot ? 'step-slot' : 'step'}`,
        {},
      ),
    )
    if (snapshot.value.done) pause()
  } finally {
    busy.value = false
  }
}
function pause() {
  playing.value = false
  playback++
}
async function play() {
  const generation = ++playback
  playing.value = true
  try {
    while (playing.value && generation === playback) {
      await step()
      if (playing.value)
        await new Promise((resolve) => setTimeout(resolve, 800))
    }
  } finally {
    if (generation === playback) playing.value = false
  }
}
async function reset() {
  pause()
  busy.value = true
  try {
    receive(await api.post(`/sim/sessions/${session.value}/reset`), true)
  } finally {
    busy.value = false
  }
}
function inspect(node) {
  selected.value = node
  drawer.value = true
}
onMounted(preview)
onBeforeUnmount(() => {
  pause()
  if (session.value) api.delete(`/sim/sessions/${session.value}`)
})
</script>
<template>
  <div
    class="overview"
    :class="{
      'focus-resources': route.path === '/resources',
      'focus-topology': route.path === '/topology',
    }"
    v-loading="!snapshot && busy"
    element-loading-background="#06152f"
  >
    <div class="dashboard-left">
      <section class="monitor-panel run-overview">
        <h2 class="monitor-title">调度运行概况</h2>
        <div class="overview-body">
          <div class="dashboard-kpis">
            <div
              v-for="(k, i) in [
                ['任务完成率', 'task_success_rate', '%', 100],
                ['平均完成时延', 'avg_delay_s', 's', 1],
                ['平均任务能耗', 'avg_energy_j', 'J', 1],
                ['负载方差', 'mean_load_variance', '', 1],
              ]"
              :key="k[1]"
              class="dashboard-kpi"
            >
              <div>
                <span>{{ k[0] }}</span><strong
                  >{{
                    snapshot?.metrics[k[1]] == null
                      ? '—'
                      : fmt(
                          snapshot.metrics[k[1]] * k[3],
                          i === 3 ? 3 : i === 0 ? 1 : 2,
                        )
                  }}<small>{{ k[2] }}</small></strong
                >
              </div>
            </div>
          </div>
          <div class="task-totals">
            <span
              >本轮任务总数 <b>{{ snapshot?.total_tasks ?? '—' }}</b></span
            ><span
              >按期完成 <b>{{ completed }}</b></span
            >
          </div>
          <div class="distribution">
            <h3>
              业务任务分布（5 类）<small
                >已调度 {{ snapshot?.task_index || 0 }} 个</small
              >
            </h3>
            <div class="distribution-bars">
              <div v-for="(name, i) in taskNames" :key="name">
                <span
                  :style="{
                    color: [
                      '#a49eff',
                      '#ffc36c',
                      '#51dfb3',
                      '#68adff',
                      '#8782ff',
                    ][i],
                  }"
                  >{{ name }}</span><b
                  >{{
                    snapshot?.task_index
                      ? fmt((taskCounts[i] / snapshot.task_index) * 100, 1)
                      : '0.0'
                  }}%</b
                >
              </div>
            </div>
          </div>
        </div>
      </section>
      <section class="monitor-panel utilization">
        <h2 class="monitor-title">资源利用率</h2>
        <div class="util-body">
          <div
            v-for="(name, i) in resources"
            :key="name"
            class="total-resource"
          >
            <b :class="'resource-name-' + i">{{ name }}</b>
            <div class="meter">
              <i :class="'resource-' + i" :style="{ width: used[i] + '%' }"></i>
            </div>
            <span>{{ fmt(used[i], 1) }}%</span>
          </div>
          <div class="trend-heading">
            资源利用率趋势 <small>最近 40 次快照 · 全部计算节点</small>
          </div>
          <ResourceTrend :samples="samples" />
        </div>
      </section>
    </div>
    <div class="dashboard-center">
      <section class="monitor-panel network-panel">
        <h2 class="monitor-title">
          动态云边端网络拓扑
          <small :class="playing ? 'good' : ''">{{
            playing
              ? '仿真运行中'
              : session
                ? snapshot?.done
                  ? '本轮已结束'
                  : '推演已暂停'
                : '初始预览'
          }}</small>
        </h2>
        <div class="overview-controls">
          <el-select
            v-model="preset"
            aria-label="仿真场景"
            size="small"
            :disabled="playing || busy"
            style="width: 104px"
            @change="preview"
            ><el-option label="节点管理配置" value="draft" /><el-option
              v-for="p in ['default', 'high_congestion', 'small_smoke']"
              :key="p"
              :label="p"
              :value="p" /></el-select
          ><el-select
            v-model="policy"
            aria-label="调度策略"
            size="small"
            :disabled="playing || busy"
            style="width: 126px"
            ><el-option
              v-for="(name, key) in policyNames"
              :key="key"
              :label="name"
              :value="key" /></el-select
          ><el-select
            v-if="policy === 'checkpoint'"
            v-model="checkpoint"
            aria-label="模型权重"
            size="small"
            style="width: 170px"
            :disabled="playing || busy"
            placeholder="选择已有权重"
            filterable
            ><el-option
              v-for="c in app.checkpoints"
              :key="c.path"
              :value="c.path"
              :label="c.run_dir + ' / ' + c.name" /></el-select
          ><el-button
            size="small"
            :disabled="busy || (policy === 'checkpoint' && !checkpoint)"
            @click="create"
            >{{ session ? '重建会话' : '创建会话' }}</el-button
          ><el-button
            size="small"
            type="primary"
            :disabled="!session || busy || snapshot?.done"
            @click="play"
            v-if="!playing"
            >开始推演</el-button
          ><el-button size="small" type="warning" @click="pause" v-else
            >暂停</el-button
          ><el-button
            size="small"
            :disabled="!session || busy || playing || snapshot?.done"
            @click="step()"
            >单任务</el-button
          ><el-button
            size="small"
            :disabled="!session || busy || playing || snapshot?.done"
            @click="step(true)"
            >整时隙</el-button
          ><el-button
            size="small"
            :disabled="!session || busy || playing"
            @click="reset"
            >重置</el-button
          >
        </div>
        <TopologyGraph
          :snapshot="snapshot"
          :selected="selectedPath"
          @select-node="inspect"
        />
      </section>
      <section class="monitor-panel dispatch-panel">
        <h2 class="monitor-title">
          任务调度明细<small>点击任务查看实际路径</small>
        </h2>
        <TaskFlowTable
          :snapshot="snapshot"
          compact
          @select-path="selectedPath = $event"
        />
      </section>
    </div>
    <div class="dashboard-right">
      <section class="monitor-panel node-status">
        <h2 class="monitor-title">
          计算节点状态<small>全部 {{ cards.length }} 个节点</small>
        </h2>
        <div class="node-card-list">
          <button
            v-for="node in cards"
            :key="node.kind + node.id"
            class="status-node-card"
            @click="inspect(node)"
          >
            <NodeGlyph :kind="node.kind" />
            <div class="status-node-info">
              <div class="status-node-title">
                <strong
                  >{{
                    node.kind === 'cloud'
                      ? 'C0'
                      : (node.kind === 'terminal' ? 'U' : 'E') +
                        String(node.id).padStart(2, '0')
                  }}
                  <em
                    >（{{
                      node.kind === 'cloud'
                        ? '云端'
                        : node.kind === 'terminal'
                          ? '端侧'
                          : node.profile === 'balanced'
                            ? '中继'
                            : node.profile
                    }}）</em
                  ></strong
                ><span class="good">在线</span>
              </div>
              <div
                v-for="(r, i) in node.kind === 'terminal' ? [] : resources"
                :key="r"
                class="node-meter"
              >
                <span>{{ r }}</span>
                <div class="meter">
                  <i
                    :class="'resource-' + i"
                    :style="{ width: percentage(node, i) + '%' }"
                  ></i>
                </div>
                <span>{{ fmt(percentage(node, i), 0) }}%</span>
              </div>
              <p v-if="node.kind === 'terminal'" class="terminal-status">
                接入 E{{ node.access_edge }} ·
                {{ fmt(node.radio_rate_mbps, 1) }} Mbps<br />端侧任务源 ·
                不执行本地计算
              </p>
            </div>
          </button>
        </div>
      </section>
      <section class="monitor-panel strategy-panel">
        <h2 class="monitor-title">调度策略</h2>
        <div class="strategy-body">
          <div class="strategy-name">
            <b>{{ strategy }}</b><small :class="playing ? 'good' : 'muted'">{{
              playing ? '运行中' : session ? '已就绪' : '未启动'
            }}</small>
          </div>
          <div class="strategy-param">
            路径最大跳数：<b>{{ config?.max_hops ?? '—' }}</b>
          </div>
          <div class="strategy-param">
            拓扑动态更新：<b>{{
              snapshot?.topology_period_slots
                ? '每 ' + snapshot.topology_period_slots + ' 时隙'
                : '—'
            }}</b>
          </div>
          <div
            v-if="activePolicy === 'checkpoint'"
            class="checkpoint-source"
            :title="activeCheckpoint"
          >
            {{ activeCheckpoint }}
          </div>
          <div class="workflow-heading">GAT-PPO 调度原理</div>
          <div class="strategy-flow">
            <template
              v-for="(s, i) in [
                ['任务', '请求'],
                ['GAT', '拓扑感知'],
                ['PPO', '联合决策'],
                ['执行与', '反馈'],
              ]"
              :key="i"
              ><span v-if="i" class="flow-arrow">➜</span>
              <div>
                <b>{{ s[0] }}<br />{{ s[1] }}</b>
              </div></template
            >
          </div>
          <p class="strategy-note">
            {{
              activePolicy === 'checkpoint'
                ? '当前按所选权重推理，模型类型以训练记录为准。'
                : '当前使用启发式策略；上方为 GAT-PPO 原理示意。'
            }}
          </p>
        </div>
      </section>
    </div>
    <el-drawer
      v-model="drawer"
      :title="selected?.kind === 'terminal' ? '端侧设备详情' : '计算节点详情'"
      size="420px"
      ><template v-if="selected?.kind === 'terminal'"
        ><h3>U{{ selected.id }} · 端侧任务源</h3>
        <p class="terminal-details">
          接入节点 E{{ selected.access_edge }}<br />实际无线速率
          {{ fmt(selected.radio_rate_mbps) }} Mbps
        </p></template
      ><NodeResourcePanel
        v-else
        :node="selected"
        :nodes="snapshot?.nodes || []"
        @select-node="selected = $event"
    /></el-drawer>
  </div>
</template>
