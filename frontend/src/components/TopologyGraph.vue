<script setup>
import { computed, ref, watch, useId } from 'vue'
import { storage } from '../utils/storage'
import NodeGlyph from './NodeGlyph.vue'
const props = defineProps({ snapshot: Object, selected: Object })
const emit = defineEmits(['select-node'])
const arrow = useId().replace(/:/g, '')
const svg = ref(null),
  dragging = ref(null),
  positions = ref({}),
  layouts = storage.get('layout', {})
const bucket = computed(
  () =>
    'layered-' +
    (props.snapshot?.nodes.filter((n) => n.kind === 'edge').length || 0),
)
watch(
  bucket,
  (key) => {
    positions.value = { ...(layouts[key] || {}) }
  },
  { immediate: true },
)
const items = computed(() => {
  if (!props.snapshot) return []
  const edges = props.snapshot.nodes.filter((n) => n.kind === 'edge'),
    cols = Math.ceil(edges.length / 2)
  const nodes = props.snapshot.nodes.map((n) => ({
    ...n,
    key: 'n' + n.id,
    x:
      n.kind === 'cloud'
        ? 500
        : 70 + ((n.id % cols) * 860) / Math.max(1, cols - 1),
    y: n.kind === 'cloud' ? 85 : 225 + Math.floor(n.id / cols) * 125,
  }))
  props.snapshot.terminals.forEach((n, i) =>
    nodes.push({
      ...n,
      kind: 'terminal',
      key: 't' + n.id,
      x: 35 + ((i % 12) * 930) / 11,
      y: 465 + Math.floor(i / 12) * 62,
    }),
  )
  return nodes.map((n) => ({ ...n, ...positions.value[n.key] }))
})
const map = computed(() =>
  Object.fromEntries(items.value.map((n) => [n.key, n])),
)
const chosen = computed(() => props.selected || props.snapshot?.last_decision)
const pathLines = computed(() => {
  if (!chosen.value || chosen.value.destination === -2) return []
  const keys = [
    't' + chosen.value.source,
    ...chosen.value.path.map((n) => 'n' + n),
  ]
  return keys
    .slice(1)
    .map((key, i) => [map.value[keys[i]], map.value[key]])
    .filter((p) => p[0] && p[1])
})
function pathEnd(a, b) {
  const dx = b.x - a.x,
    dy = b.y - a.y
  const distance = Math.hypot(dx, dy)
  const offset = b.kind === 'cloud' ? 80 : 50
  return {
    x: b.x - (dx / distance) * offset,
    y: b.y - (dy / distance) * offset,
  }
}
function color(l) {
  const v = Math.max(l.background_fwd, l.background_rev)
  return l.status === 'flapped'
    ? '#59738f'
    : v > 0.7
      ? '#f06465'
      : v > 0.4
        ? '#ffa94c'
        : '#009fe8'
}
function start(e, n) {
  dragging.value = n
  e.currentTarget.setPointerCapture(e.pointerId)
}
function move(e) {
  if (!dragging.value) return
  const p = svg.value.createSVGPoint()
  p.x = e.clientX
  p.y = e.clientY
  const point = p.matrixTransform(svg.value.getScreenCTM().inverse())
  positions.value = {
    ...positions.value,
    [dragging.value.key]: {
      x: Math.max(30, Math.min(970, point.x)),
      y: Math.max(40, Math.min(530, point.y)),
    },
  }
}
function end() {
  if (!dragging.value) return
  layouts[bucket.value] = positions.value
  storage.set('layout', layouts)
  dragging.value = null
}
</script>
<template>
  <div class="topology layered-topology">
    <div v-if="!snapshot" class="empty-graph">正在加载真实网络拓扑…</div>
    <template v-else
      ><div class="topology-meta">
        <span>▣ 端侧任务源：{{ snapshot.terminals.length }} 个</span
        ><span>▤ 边缘计算节点：{{ snapshot.nodes.length - 1 }} 个</span
        ><span>☁ 云端节点：1 个</span>
      </div>
      <div class="topology-key">
        <span>☁ 云节点</span><span>▤ 边缘节点</span><span>▣ 端设备</span
        ><span class="good">➜ 当前调度路径</span>
      </div>
      <svg
        ref="svg"
        viewBox="0 0 1000 565"
        @pointermove="move"
        @pointerup="end"
        @pointercancel="end"
      >
        <defs>
          <marker
            :id="arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M0 0L10 5L0 10Z" fill="#52ffbf" />
          </marker>
        </defs>
        <path
          d="M0 560L500 365L1000 560M150 565L500 365L850 565M325 565L500 365L675 565M0 520H1000M0 545H1000"
          stroke="#0877aa"
          opacity=".16"
          fill="none"
        />
        <g class="radio-links">
          <line
            v-for="t in snapshot.terminals"
            :key="t.id"
            :x1="map['t' + t.id].x"
            :y1="map['t' + t.id].y - 12"
            :x2="map['n' + t.access_edge].x"
            :y2="map['n' + t.access_edge].y + 20"
            stroke="#0475b0"
            stroke-width="1"
            opacity=".36"
          />
        </g>
        <g v-for="(l, i) in snapshot.links" :key="i" class="topology-link">
          <line
            :x1="map['n' + l.source].x"
            :y1="map['n' + l.source].y"
            :x2="map['n' + l.target].x"
            :y2="map['n' + l.target].y"
            :stroke="color(l)"
            :stroke-width="l.kind === 'cloud' ? 1 : 1.8"
            :stroke-opacity="l.kind === 'cloud' ? 0.4 : 0.75"
            :stroke-dasharray="l.status === 'flapped' ? '5 5' : ''"
          />
          <title>
            {{ l.source }} → {{ l.target }} · {{ l.rate_mbps.toFixed(1) }} Mbps
            · 拥塞 {{ (100 * l.background_fwd).toFixed(1) }}% ·
            {{ l.status === 'flapped' ? '已断开' : '连通' }}
          </title>
        </g>
        <g class="active-path">
          <line
            v-for="(p, i) in pathLines"
            :key="i"
            :x1="p[0].x"
            :y1="p[0].y"
            :x2="pathEnd(p[0], p[1]).x"
            :y2="pathEnd(p[0], p[1]).y"
            class="route-line"
            :marker-end="`url(#${arrow})`"
          />
        </g>
        <g
          v-for="n in items"
          :key="n.key"
          class="graph-node"
          role="button"
          tabindex="0"
          :aria-label="n.kind === 'terminal' ? '终端 U' + n.id : n.label"
          @pointerdown="start($event, n)"
          @click="emit('select-node', n)"
          @keydown.enter="emit('select-node', n)"
        >
          <rect
            v-if="n.kind !== 'terminal'"
            :x="n.x - (n.kind === 'cloud' ? 70 : 36)"
            :y="n.y - 32"
            :width="n.kind === 'cloud' ? 140 : 72"
            :height="n.kind === 'cloud' ? 88 : 84"
            rx="8"
            :class="['node-back', { involved: chosen?.path?.includes(n.id) }]"
          />
          <foreignObject
            :x="
              n.x - (n.kind === 'cloud' ? 55 : n.kind === 'terminal' ? 20 : 31)
            "
            :y="
              n.y - (n.kind === 'cloud' ? 52 : n.kind === 'terminal' ? 20 : 32)
            "
            :width="n.kind === 'cloud' ? 110 : n.kind === 'terminal' ? 40 : 62"
            :height="n.kind === 'cloud' ? 88 : n.kind === 'terminal' ? 32 : 50"
            ><NodeGlyph :kind="n.kind"
          /></foreignObject>
          <text
            :x="n.x"
            :y="
              n.y + (n.kind === 'cloud' ? 45 : n.kind === 'terminal' ? 26 : 27)
            "
            class="node-name"
          >
            {{
              n.kind === 'cloud'
                ? 'C0（云端）'
                : n.kind === 'terminal'
                  ? 'U' + n.id
                  : 'E' + n.id
            }}
          </text>
          <text
            v-if="n.kind === 'edge'"
            :x="n.x"
            :y="n.y + 44"
            :class="['node-profile', n.profile]"
          >
            {{ n.profile === 'balanced' ? '中继' : n.profile }}
          </text>
          <title>
            {{ n.label
            }}{{
              n.kind === 'terminal'
                ? ` · 接入 E${n.access_edge} · ${n.radio_rate_mbps.toFixed(1)} Mbps`
                : ` · ${n.profile} · CPU/NPU/FPGA ${n.resource_used.map((v, i) => ((v / n.speed[i]) * 100).toFixed(1) + '%').join(' / ')}`
            }}
          </title>
        </g>
      </svg>
      <div class="graph-legend">
        <span><i style="background: #009fe8"></i>正常</span
        ><span><i style="background: #ffa94c"></i>拥塞</span
        ><span><i style="background: #f06465"></i>高拥塞</span
        ><span>虚线：失效链路</span><span>拖动微调 · 点击查看</span>
      </div></template
    >
  </div>
</template>
