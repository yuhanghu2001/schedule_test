<script setup>
import { computed, ref, watch, useId } from 'vue'
import { storage } from '../utils/storage'
import NodeGlyph from './NodeGlyph.vue'
const props = defineProps({ snapshot: Object, selected: Object })
const emit = defineEmits(['select-node'])
const id = useId().replace(/:/g, '')
const svg = ref(null),
  dragging = ref(null),
  positions = ref({})
const layouts = storage.get('layout', {})
const edges = computed(
  () => props.snapshot?.nodes.filter((n) => n.kind === 'edge') || [],
)
const columns = computed(() => Math.ceil(edges.value.length / 2))
const columnX = (i) => 270 + (i * 1000) / Math.max(1, columns.value - 1)
const groupWidth = computed(() => Math.min(208, 1240 / columns.value - 4))
const groups = computed(() =>
  Array.from({ length: columns.value }, (_, i) => ({
    x: columnX(i),
    terminals: props.snapshot.terminals
      .filter(
        (t) =>
          edges.value.findIndex((n) => n.id === t.access_edge) %
            columns.value ===
          i,
      )
      .sort((a, b) => a.access_edge - b.access_edge || a.id - b.id),
  })),
)
const height = computed(() =>
  Math.max(
    760,
    ...groups.value.map((g) => 650 + Math.ceil(g.terminals.length / 4) * 65),
  ),
)
const bucket = computed(
  () =>
    'three-tier-' +
    edges.value.length +
    '-' +
    (props.snapshot?.terminals.length || 0),
)
watch(
  bucket,
  (key) => {
    positions.value = { ...layouts[key] }
  },
  { immediate: true },
)
const items = computed(() => {
  if (!props.snapshot) return []
  const nodes = props.snapshot.nodes.map((n) => {
    const index = edges.value.findIndex((e) => e.id === n.id)
    return {
      ...n,
      key: 'n' + n.id,
      x: n.kind === 'cloud' ? 755 : columnX(index % columns.value),
      y:
        n.kind === 'cloud'
          ? 110
          : 300 + Math.floor(index / columns.value) * 160,
    }
  })
  groups.value.forEach((group) =>
    group.terminals.forEach((t, i) => {
      const count = Math.min(4, group.terminals.length - Math.floor(i / 4) * 4)
      nodes.push({
        ...t,
        kind: 'terminal',
        key: 't' + t.id,
        x:
          group.x + (((i % 4) - (count - 1) / 2) * (groupWidth.value - 16)) / 4,
        y: 665 + Math.floor(i / 4) * 65,
      })
    }),
  )
  return nodes.map((n) => ({ ...n, ...positions.value[n.key] }))
})
const map = computed(() =>
  Object.fromEntries(items.value.map((n) => [n.key, n])),
)
const frames = computed(() =>
  groups.value
    .filter((g) => g.terminals.length)
    .map((g) => {
      const points = g.terminals.map((t) => map.value['t' + t.id])
      return {
        x: Math.min(...points.map((p) => p.x)) - 24,
        y: Math.min(...points.map((p) => p.y)) - 29,
        width:
          Math.max(...points.map((p) => p.x)) -
          Math.min(...points.map((p) => p.x)) +
          48,
        height:
          Math.max(...points.map((p) => p.y)) -
          Math.min(...points.map((p) => p.y)) +
          67,
      }
    }),
)
const chosen = computed(() => props.selected || props.snapshot?.last_decision)
const active = computed(() => chosen.value && chosen.value.destination >= 0)
const pathPairs = computed(() => {
  if (!active.value) return []
  const keys = [
    't' + chosen.value.source,
    ...chosen.value.path.map((n) => 'n' + n),
  ]
  return keys.slice(1).map((key, i) => [map.value[keys[i]], map.value[key]])
})
function boundary(a, b) {
  const centerY =
    b.y + (b.kind === 'cloud' ? -22 : b.kind === 'terminal' ? -8 : 0)
  const dx = b.x - a.x,
    dy = centerY - a.y
  const halfWidth = b.kind === 'cloud' ? 98 : b.kind === 'terminal' ? 19 : 48
  const halfHeight = b.kind === 'cloud' ? 64 : b.kind === 'terminal' ? 22 : 53
  const ratio = Math.min(halfWidth / Math.abs(dx), halfHeight / Math.abs(dy))
  return { x: b.x - dx * ratio, y: centerY - dy * ratio }
}
function routePath(a, b) {
  const start = boundary(b, a),
    end = boundary(a, b)
  return `M${start.x},${start.y} L${end.x},${end.y}`
}
function networkPath(link) {
  const a = map.value['n' + link.source],
    b = map.value['n' + link.target]
  if (link.kind === 'cloud')
    return `M${b.x},${b.y + 40} Q${a.x},${a.y - 110} ${a.x},${a.y - 50}`
  const start = boundary(b, a),
    end = boundary(a, b)
  if (Math.abs(a.y - b.y) < 5 && Math.abs(a.x - b.x) > 250) {
    const bend = a.y === 300 ? 220 : 545
    return `M${start.x},${start.y} Q${(a.x + b.x) / 2},${bend} ${end.x},${end.y}`
  }
  return `M${start.x},${start.y} L${end.x},${end.y}`
}
function radioPath(t) {
  const terminal = map.value['t' + t.id],
    edge = map.value['n' + t.access_edge]
  const upper = edge.y < 380,
    bus = upper ? 585 : 607,
    side = edge.x - (upper ? 65 : 0)
  return `M${edge.x - (upper ? 48 : 0)},${edge.y + (upper ? 0 : 50)} H${side} V${bus} H${terminal.x} V${terminal.y - 20}`
}
function color(link) {
  const load = Math.max(link.background_fwd, link.background_rev)
  return link.status === 'flapped'
    ? '#647b94'
    : load > 0.7
      ? '#ff6978'
      : load > 0.4
        ? '#ffba62'
        : '#00d5ff'
}
function involved(node) {
  return (
    active.value &&
    (node.kind === 'terminal'
      ? chosen.value.source === node.id
      : chosen.value.path.includes(node.id))
  )
}
function start(event, node) {
  dragging.value = node
  event.currentTarget.setPointerCapture(event.pointerId)
}
function move(event) {
  if (!dragging.value) return
  const point = svg.value.createSVGPoint()
  point.x = event.clientX
  point.y = event.clientY
  const local = point.matrixTransform(svg.value.getScreenCTM().inverse())
  positions.value = {
    ...positions.value,
    [dragging.value.key]: {
      x: Math.max(28, Math.min(1372, local.x)),
      y: Math.max(65, Math.min(height.value - 40, local.y)),
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
  <div class="topology layered-topology tiered-topology">
    <div v-if="!snapshot" class="empty-graph">正在加载真实网络拓扑…</div>
    <template v-else
      ><svg
        ref="svg"
        :viewBox="`0 0 1400 ${height}`"
        @pointermove="move"
        @pointerup="end"
        @pointercancel="end"
      >
        <defs>
          <marker
            :id="id + '-arrow'"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto"
          >
            <path d="M0 0L10 5L0 10Z" fill="#72ff66" />
          </marker>
          <linearGradient :id="id + '-node'" x2="0" y2="1">
            <stop stop-color="#075078" />
            <stop offset=".35" stop-color="#052a49" />
            <stop offset="1" stop-color="#03192f" />
          </linearGradient>
        </defs>
        <g class="layer-dividers"><path d="M20 220H1380M20 565H1380" /></g>
        <g class="radio-links">
          <path
            v-for="t in snapshot.terminals"
            :key="t.id"
            :d="radioPath(t)"
            :data-source="'U' + t.id"
            :data-target="'E' + t.access_edge"
          >
            <title>
              U{{ t.id }} → E{{ t.access_edge }} ·
              {{ t.radio_rate_mbps.toFixed(1) }} Mbps
            </title>
          </path>
        </g>
        <g v-for="(link, i) in snapshot.links" :key="i" class="topology-link">
          <path
            :d="networkPath(link)"
            :stroke="color(link)"
            :class="[
              'physical-link',
              link.kind,
              { 'failed-link': link.status === 'flapped' },
            ]"
            :data-source="link.source"
            :data-target="link.target"
          >
            <title>
              {{ link.source }} ↔ {{ link.target }} ·
              {{ link.rate_mbps.toFixed(1) }} Mbps · 拥塞
              {{
                (
                  100 * Math.max(link.background_fwd, link.background_rev)
                ).toFixed(1)
              }}% · {{ link.status === 'flapped' ? '失效' : '连通' }}
            </title>
          </path>
        </g>
        <g class="terminal-frames">
          <rect v-for="(frame, i) in frames" :key="i" v-bind="frame" rx="12" />
        </g>
        <g class="active-path">
          <path
            v-for="(pair, i) in pathPairs"
            :key="i"
            :d="routePath(...pair)"
            class="route-line"
            :marker-end="`url(#${id}-arrow)`"
          />
        </g>
        <g
          v-for="node in items"
          :key="node.key"
          :class="[
            'graph-node',
            'scene-' + node.kind,
            { involved: involved(node) },
          ]"
          role="button"
          tabindex="0"
          :aria-label="
            node.kind === 'terminal' ? '终端 U' + node.id : node.label
          "
          @pointerdown="start($event, node)"
          @click="emit('select-node', node)"
          @keydown.enter="emit('select-node', node)"
        >
          <rect
            v-if="node.kind === 'edge'"
            :x="node.x - 48"
            :y="node.y - 50"
            width="96"
            height="100"
            rx="12"
            class="node-back"
            :style="{ fill: `url(#${id}-node)` }"
          />
          <foreignObject
            :x="
              node.x -
              (node.kind === 'cloud' ? 100 : node.kind === 'terminal' ? 21 : 36)
            "
            :y="
              node.y -
              (node.kind === 'cloud' ? 84 : node.kind === 'terminal' ? 27 : 49)
            "
            :width="
              node.kind === 'cloud' ? 200 : node.kind === 'terminal' ? 42 : 72
            "
            :height="
              node.kind === 'cloud' ? 150 : node.kind === 'terminal' ? 38 : 58
            "
            ><NodeGlyph :kind="node.kind"
          /></foreignObject>
          <text
            v-if="node.kind !== 'cloud'"
            :x="node.x"
            :y="node.y + (node.kind === 'terminal' ? 29 : 17)"
            class="node-name"
          >
            {{ (node.kind === 'terminal' ? 'U' : 'E') + node.id }}
          </text>
          <text
            v-if="node.kind === 'edge'"
            :x="node.x"
            :y="node.y + 40"
            :class="['node-profile', node.profile]"
          >
            {{ node.profile === 'balanced' ? '中继' : node.profile }}
          </text>
          <title>
            {{ node.label
            }}{{
              node.kind === 'terminal'
                ? ` · 接入 E${node.access_edge} · ${node.radio_rate_mbps.toFixed(1)} Mbps`
                : ` · CPU/NPU/FPGA ${node.resource_used.map((v, i) => ((100 * v) / node.speed[i]).toFixed(1) + '%').join(' / ')}`
            }}
          </title>
        </g>
        <text x="700" :y="height - 15" class="group-caption">
          终端按真实接入节点分组 · 每条连接均来自当前仿真快照
        </text>
      </svg>
    </template>
  </div>
</template>
