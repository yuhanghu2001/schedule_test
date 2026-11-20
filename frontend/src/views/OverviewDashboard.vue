<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { useRoute } from 'vue-router'
import { useAppStore } from '../stores/app'
import api from '../api/client'
import NetworkMonitorScene from '../components/NetworkMonitorScene.vue'
import NodeResourcePanel from '../components/NodeResourcePanel.vue'
import TaskFlowTable from '../components/TaskFlowTable.vue'
import ResourceTrend from '../components/ResourceTrend.vue'
const app = useAppStore(),
  route = useRoute()
const snapshot = ref(null),
  session = ref(null),
  selectedPath = ref(null),
  selected = ref(null),
  drawer = ref(false),
  busy = ref(false),
  playing = ref(false),
  paused = ref(false)
const policy = ref(''),
  samples = ref([])
let disposed = false
const resources = ['CPU', 'NPU', 'FPGA'],
  taskNames = ['图像识别', '目标检测', '视频分析', '语音处理', '信号计算']
const policies = {
  shortest_fastest: {
    name: '最短最快启发式算法',
    rule: '优先选跳数最少的方案。跳数相同则选分配算力最大的，再依次比较估算时延和能耗，取更低者。',
  },
  greedy_estimate: {
    name: '贪心估计启发式算法',
    rule: '将估算时延和能耗分别归一化后加权求和，选择总代价最低的可行方案。',
  },
  random: {
    name: '随机算法',
    rule: '在可行方案和拒绝动作之间等概率抽样，因此有可行方案时也可能拒绝。',
  },
}
const strategySteps = ['任务到达', '筛选候选', '选择动作', '执行反馈']
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
const maxHops = computed(() =>
  app.env?.config.max_hops ?? app.presets[app.env?.preset_name]?.max_hops,
)
const strategy = computed(() => policies[policy.value]?.name || '请选择调度算法')
const strategyStatus = computed(() => {
  if (!session.value) return '未加载'
  if (snapshot.value.done) return '推演完成'
  if (paused.value) return '已暂停'
  if (playing.value) return '运行中'
  return snapshot.value.task_index ? '推演已中断' : '任务已加载'
})
const fmt = (v, n = 2) => (v == null ? '—' : Number(v).toFixed(n))
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
  busy.value = true
  try {
    await releaseSession()
    if (!Object.keys(app.presets).length) await app.loadPresets()
    receive(await api.post('/env/preview', app.env), true)
  } finally {
    busy.value = false
  }
}
function releaseSession() {
  const id = session.value
  session.value = null
  paused.value = false
  if (id) return api.delete(`/sim/sessions/${id}`)
}
function changePolicy() {
  if (session.value) return preview()
}
async function loadTasks() {
  busy.value = true
  try {
    await releaseSession()
    const data = await api.post('/sim/sessions', {
      ...app.env,
      policy: policy.value,
    })
    if (disposed) {
      await api.delete(`/sim/sessions/${data.session_id}`)
      return
    }
    session.value = data.session_id
    receive(data.snapshot, true)
  } finally {
    busy.value = false
  }
}
async function play() {
  paused.value = false
  playing.value = true
  try {
    while (!disposed && !paused.value && !snapshot.value.done) {
      receive(await api.post(`/sim/sessions/${session.value}/step`, {}))
      if (!paused.value && !snapshot.value.done)
        await new Promise((resolve) => setTimeout(resolve, 800))
    }
  } finally {
    if (snapshot.value.done) paused.value = false
    playing.value = false
  }
}
function inspect(node) {
  selected.value = node
  drawer.value = true
}
onMounted(preview)
onBeforeUnmount(() => {
  disposed = true
  releaseSession()
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
                ['任务平均完成时间', 'avg_delay_s', 's', 1],
                ['任务平均能耗', 'avg_energy_j', 'J', 1],
                ['负载均衡度', 'load_balance_degree', '', 1],
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
        </div>
      </section>
      <section class="monitor-panel strategy-panel">
        <h2 class="monitor-title">调度策略</h2>
        <div class="strategy-body">
          <div class="strategy-name">
            <b>{{ strategy }}</b>
            <small :class="playing && !paused ? 'good' : 'muted'">{{ strategyStatus }}</small>
          </div>
          <div class="strategy-params">
            <div class="strategy-param">
              边缘段最大跳数 <b>{{ maxHops ?? '—' }}</b>
            </div>
            <div class="strategy-param">
              拓扑重建 <b>{{
                snapshot?.topology_period_slots
                  ? '每 ' + snapshot.topology_period_slots + ' 时隙'
                  : '—'
              }}</b>
            </div>
          </div>
          <div class="workflow-heading">共同调度流程</div>
          <div class="strategy-flow">
            <template v-for="(step, i) in strategySteps" :key="step">
              <span v-if="i" class="flow-arrow" aria-hidden="true">→</span>
              <div><b>{{ step }}</b></div>
            </template>
          </div>
          <p class="strategy-explanation">
            按服务、资源容量、链路和时限筛选执行节点、计算资源与路径的组合。执行后更新资源和任务状态。无可行方案则拒绝。
          </p>
          <div class="strategy-rules">
            <article
              v-for="(algorithm, key) in policies"
              :key="key"
              :class="{ selected: policy === key }"
            >
              <h3>{{ algorithm.name }}</h3>
              <p>{{ algorithm.rule }}</p>
            </article>
          </div>
        </div>
      </section>
    </div>
    <div class="dashboard-center">
      <section class="monitor-panel network-panel">
        <h2 class="monitor-title">动态云边端网络拓扑</h2>
        <div class="overview-controls">
          <el-select
            v-model="policy"
            aria-label="调度算法"
            placeholder="请选择调度算法"
            size="small"
            :disabled="playing || busy"
            style="width: 172px"
            @change="changePolicy"
          >
            <el-option
              v-for="(algorithm, key) in policies"
              :key="key"
              :label="algorithm.name"
              :value="key"
            />
          </el-select>
          <el-button
            size="small"
            :disabled="!policy || busy || playing"
            @click="loadTasks"
          >任务加载</el-button>
          <el-button
            size="small"
            type="primary"
            :disabled="!session || busy || playing || paused || snapshot.done"
            @click="play"
          >开始推演</el-button>
          <el-button
            size="small"
            :disabled="paused ? busy || playing : !playing"
            @click="paused ? play() : (paused = true)"
          >{{ paused ? '继续推演' : '暂停推演' }}</el-button>
        </div>
        <div class="overview-network-scene">
          <NetworkMonitorScene
            :snapshot="snapshot"
            :selected="selectedPath"
            compact
            @select-node="inspect"
          />
        </div>
      </section>
      <section class="monitor-panel dispatch-panel">
        <h2 class="monitor-title">任务调度明细</h2>
        <TaskFlowTable
          :snapshot="snapshot"
          compact
          @select-path="selectedPath = $event"
        />
      </section>
    </div>
    <div class="dashboard-right">
      <section class="monitor-panel task-generation">
        <h2 class="monitor-title">任务产生</h2>
        <div v-if="snapshot" class="task-generation-body">
          <div class="task-totals">
            <span
              >本轮任务总数 <b>{{ snapshot?.total_tasks ?? '—' }}</b></span
            ><span
              >按期完成 <b>{{ completed }}</b></span
            >
          </div>
          <div class="distribution">
            <h3>
              已调度业务分布（5 类）<small
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
          <div class="task-generation-progress">
            <span
              >已调度 <b>{{ snapshot?.task_index ?? '—' }}</b></span
            >
            <span
              >待调度
              <b>{{
                snapshot ? snapshot.total_tasks - snapshot.task_index : '—'
              }}</b></span
            >
          </div>
          <div class="pending-task">
            <h3>
              当前待调度任务<small v-if="snapshot?.current_task"
                >T-{{ snapshot.current_task.task_index }}</small
              >
            </h3>
            <dl v-if="snapshot?.current_task">
              <div>
                <dt>业务类型</dt>
                <dd>{{ snapshot.current_task.kind_name }}</dd>
              </div>
              <div>
                <dt>源端 / 接入</dt>
                <dd>
                  U{{ snapshot.current_task.source }} / E{{
                    snapshot.current_task.access_edge
                  }}
                </dd>
              </div>
              <div>
                <dt>输入 / 输出</dt>
                <dd>
                  {{ fmt(snapshot.current_task.data_mb) }} /
                  {{ fmt(snapshot.current_task.output_mb) }} MB
                </dd>
              </div>
              <div>
                <dt>计算量</dt>
                <dd>{{ fmt(snapshot.current_task.compute_gflop) }} GFLOP</dd>
              </div>
              <div>
                <dt>任务时限</dt>
                <dd>{{ fmt(snapshot.current_task.deadline_s) }} s</dd>
              </div>
              <div>
                <dt>到达时隙 / 优先级</dt>
                <dd>
                  {{ snapshot.current_task.slot }} / P{{
                    snapshot.current_task.priority
                  }}
                </dd>
              </div>
            </dl>
            <p v-else class="muted">
              {{ snapshot?.done ? '本轮任务已全部调度' : '正在加载任务' }}
            </p>
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
          <ResourceTrend :samples="samples" />
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
