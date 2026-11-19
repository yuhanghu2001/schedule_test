<script setup>
import { computed, useId } from 'vue'
import NodeGlyph from './NodeGlyph.vue'
const props = defineProps({
  snapshot: Object,
  selected: Object,
  compact: Boolean,
})
const emit = defineEmits(['select-node'])
const id = useId().replace(/:/g, '')
const nodes = computed(() =>
  props.snapshot
    ? [
        ...props.snapshot.nodes.map((n) => ({
          ...n,
          key: 'n' + n.id,
          x: n.position[0],
          y: n.position[1],
        })),
        ...props.snapshot.terminals.map((t) => ({
          ...t,
          kind: 'terminal',
          key: 't' + t.id,
          x: t.position[0],
          y: t.position[1],
        })),
      ]
    : [],
)
const map = computed(() =>
  Object.fromEntries(nodes.value.map((n) => [n.key, n])),
)
const decision = computed(() => props.selected || props.snapshot?.last_decision)
const route = computed(() => {
  if (!decision.value || decision.value.destination < 0) return []
  const keys = [
    't' + decision.value.source,
    ...decision.value.path.map((n) => 'n' + n),
  ]
  return keys.slice(1).map((key, i) => [map.value[keys[i]], map.value[key]])
})
const isActive = (n) =>
  decision.value?.destination >= 0 &&
  (n.kind === 'terminal'
    ? decision.value.source === n.id
    : decision.value.path.includes(n.id))
function boundary(node, target) {
  const dx = target.x - node.x,
    dy = target.y - node.y
  const scale = 1 / Math.max(Math.abs(dx) / 42, Math.abs(dy) / 50)
  return { x: node.x + dx * scale, y: node.y + dy * scale }
}
function networkPath(a, b) {
  let points
  if (a.kind === 'cloud' || b.kind === 'cloud') {
    const cloud = a.kind === 'cloud' ? a : b,
      edge = a.kind === 'cloud' ? b : a
    const port = { x: cloud.x + (edge.x - cloud.x) * 0.14, y: cloud.y + 90 }
    points = [port, boundary(edge, port)]
    if (a.kind !== 'cloud') points.reverse()
  } else points = [boundary(a, b), boundary(b, a)]
  return points.map((p, i) => `${i ? 'L' : 'M'}${p.x} ${p.y}`).join(' ')
}
function physicalPath(link) {
  return networkPath(map.value['n' + link.source], map.value['n' + link.target])
}
function radioPath(t) {
  const a = map.value['n' + t.access_edge],
    b = map.value['t' + t.id]
  const side = Math.sign(b.x - a.x) || 1
  return `M${a.x + side * 42} ${a.y + 24} C${a.x + side * 120} ${a.y + 100},${b.x} ${b.y - 100},${b.x} ${b.y - 32}`
}
function currentPath(a, b) {
  if (a.kind !== 'terminal') return networkPath(a, b)
  const side = Math.sign(a.x - b.x) || 1
  return `M${a.x} ${a.y - 32} C${a.x} ${a.y - 100},${b.x + side * 120} ${b.y + 100},${b.x + side * 42} ${b.y + 24}`
}
const nodeName = (n) =>
  n.kind === 'cloud' ? 'C0' : (n.kind === 'terminal' ? 'U' : 'E') + n.id
const resourceName = (n) => (n.profile === 'balanced' ? '中继' : n.profile)
function linkColor(l) {
  return l.status === 'flapped' ||
    Math.max(l.background_fwd, l.background_rev) > 0.7
    ? '#ff687d'
    : l.kind === 'cloud'
      ? '#009bff'
      : '#ffb52e'
}
</script>
<template>
  <svg
    class="network-monitor-scene"
    :class="{ 'network-monitor-scene--compact': compact }"
    viewBox="0 0 1800 900"
    :preserveAspectRatio="compact ? 'xMidYMid meet' : 'none'"
    aria-label="真实云边端网络拓扑"
  >
    <defs>
      <linearGradient :id="id + '-tower'" x2="0" y2="1">
        <stop stop-color="#43b8fa" />
        <stop offset=".35" stop-color="#095a9d" />
        <stop offset="1" stop-color="#06335c" />
      </linearGradient>
      <linearGradient :id="id + '-cloud'" x2="0" y2="1">
        <stop stop-color="#c2faff" />
        <stop offset=".32" stop-color="#20b9fb" />
        <stop offset="1" stop-color="#025aab" />
      </linearGradient>
      <linearGradient :id="id + '-deck'" x2="0" y2="1">
        <stop stop-color="#063d6c" />
        <stop offset="1" stop-color="#01142e" />
      </linearGradient>
      <pattern
        :id="id + '-dots'"
        width="9"
        height="9"
        patternUnits="userSpaceOnUse"
      >
        <circle cx="2" cy="2" r="1.6" fill="#087fc5" />
      </pattern>
      <clipPath :id="id + '-world'">
        <path
          d="M110 74l75-28 95 18 42 40-20 35-54 6-7 32-42 12-9 48-28-16-10-52-45-26-25-40Zm126 167 52 8 37 57-10 61-29 68-18-13-14-74-32-65Zm315-144 48-21 53 9 34 40 61-12 21 31-48 28-48-15-21 34-37-25-42 8-36-33Zm76 102 67 12 37 72-30 78-33 9-23-62-34-45Zm109-106 73-33 113 15 62 32 72-8 64 44-16 33-66 10-18 33-50-1-14 32-47-29-19-52-58 5-46-31-59 5Zm231 220 53-9 50 34-11 44-56 8-34-27Z"
          transform="translate(160 0) scale(1.22 .82)"
        />
      </clipPath>
      <marker
        v-for="color in ['blue', 'orange', 'cyan', 'green']"
        :key="color"
        :id="id + '-' + color"
        viewBox="0 0 10 10"
        refX="9"
        refY="5"
        markerWidth="5"
        markerHeight="5"
        orient="auto-start-reverse"
      >
        <path
          d="M0 0L10 5L0 10Z"
          :fill="
            {
              blue: '#00a6ff',
              orange: '#ffb52e',
              cyan: '#00d9ff',
              green: '#66ff83',
            }[color]
          "
        />
      </marker>
      <g :id="id + '-server'">
        <path d="M-29-54 4-68 34-55 1-41Z" fill="#96d9ff" stroke="#b9f0ff" />
        <path d="M1-41 34-55 34 22 1 39Z" fill="#073e73" stroke="#087bc4" />
        <path
          d="M-29-54 1-41 1 39-29 25Z"
          :fill="`url(#${id}-tower)`"
          stroke="#45caff"
        />
        <g v-for="y in [-39, -20, -1, 18]" :key="y">
          <path
            :d="`M-24 ${y} -4 ${y + 8}v10l-20-8Z`"
            fill="#021f42"
            stroke="#159ae1"
          />
          <path
            :d="`M-21 ${y + 4} -11 ${y + 8}`"
            stroke="#b8f4ff"
            stroke-width="2"
          />
          <circle cx="-6" :cy="y + 11" r="1.4" fill="#4dffd6" />
        </g>
      </g>
    </defs>
    <g class="monitor-map" aria-hidden="true">
      <rect
        x="0"
        y="0"
        width="1800"
        height="390"
        :fill="`url(#${id}-dots)`"
        :clip-path="`url(#${id}-world)`"
      />
    </g>
    <g class="monitor-grid" aria-hidden="true">
      <path
        v-for="x in Array.from({ length: 23 }, (_, i) => -200 + i * 100)"
        :key="x"
        :d="`M900 510L${x} 900`"
      />
      <path
        v-for="y in [560, 590, 625, 665, 710, 765, 830, 895]"
        :key="y"
        :d="`M0 ${y}H1800`"
      />
      <path
        d="M0 690Q900 420 1800 690M0 738Q900 492 1800 738"
        class="grid-horizon"
      />
    </g>
    <g class="monitor-platform" aria-hidden="true">
      <path
        d="M90 745H1710l64 115-38 25H64l-38-25Z"
        :fill="`url(#${id}-deck)`"
      />
      <path d="M26 860 64 885H1736l38-25v14l-38 25H64l-38-25Z" fill="#032248" />
      <path
        d="M90 745 26 860l38 25H1736l38-25-64-115M64 885H1736"
        class="platform-edge"
      />
    </g>
    <template v-if="snapshot">
      <g class="monitor-physical-links">
        <path
          v-for="(link, i) in snapshot.links"
          :key="i"
          :d="physicalPath(link)"
          :stroke="linkColor(link)"
          :class="[
            'monitor-network-link',
            link.kind,
            { flapped: link.status === 'flapped' },
          ]"
          :data-source="link.source"
          :data-target="link.target"
          :marker-start="
            link.status === 'active'
              ? `url(#${id}-${link.kind === 'cloud' ? 'blue' : 'orange'})`
              : null
          "
          :marker-end="
            link.status === 'active'
              ? `url(#${id}-${link.kind === 'cloud' ? 'blue' : 'orange'})`
              : null
          "
        >
          <title>
            {{ link.source }} ↔ {{ link.target }} ·
            {{ link.rate_mbps.toFixed(1) }} Mbps · 双向背景拥塞
            {{ (100 * link.background_fwd).toFixed(1) }}% /
            {{ (100 * link.background_rev).toFixed(1) }}% ·
            {{ link.status === 'active' ? '连通' : '失效' }}
          </title>
        </path>
      </g>
      <g class="monitor-radio-links">
        <path
          v-for="t in snapshot.terminals"
          :key="t.id"
          :d="radioPath(t)"
          :data-source="t.id"
          :data-target="t.access_edge"
          :marker-end="`url(#${id}-cyan)`"
        >
          <title>
            U{{ t.id }} ↔ E{{ t.access_edge }} ·
            {{ t.radio_rate_mbps.toFixed(1) }} Mbps
          </title>
        </path>
      </g>
      <g class="monitor-current-path">
        <path
          v-for="(pair, i) in route"
          :key="i"
          :d="currentPath(...pair)"
          :marker-end="`url(#${id}-green)`"
        />
      </g>
      <g
        v-for="n in nodes"
        :key="n.key"
        :class="['monitor-scene-node', n.kind, { active: isActive(n) }]"
        role="button"
        tabindex="0"
        :aria-label="nodeName(n) + ' ' + n.label"
        @click="emit('select-node', n)"
        @keydown.enter="emit('select-node', n)"
      >
        <template v-if="n.kind === 'cloud'">
          <ellipse
            :cx="n.x"
            :cy="n.y + 89"
            rx="182"
            ry="22"
            class="cloud-orbit"
          />
          <ellipse
            :cx="n.x"
            :cy="n.y + 89"
            rx="160"
            ry="16"
            class="cloud-orbit inner"
          />
          <path
            :d="`M${n.x - 135} ${n.y + 60}C${n.x - 205} ${n.y + 45} ${n.x - 170} ${n.y - 58} ${n.x - 105} ${n.y - 46}C${n.x - 79} ${n.y - 122} ${n.x + 1} ${n.y - 118} ${n.x + 35} ${n.y - 79}C${n.x + 98} ${n.y - 99} ${n.x + 149} ${n.y - 56} ${n.x + 140} ${n.y - 15}C${n.x + 204} ${n.y - 7} ${n.x + 193} ${n.y + 65} ${n.x + 140} ${n.y + 68}Z`"
            :fill="`url(#${id}-cloud)`"
            class="monitor-cloud-body"
          />
          <use
            v-for="(s, i) in [0.42, 0.6, 0.85, 0.6, 0.42]"
            :key="i"
            :href="`#${id}-server`"
            :transform="`translate(${n.x + (i - 2) * 40} ${n.y - 7}) scale(${s})`"
          />
          <text :x="n.x" :y="n.y + 120" class="monitor-cloud-label">C0</text>
        </template>
        <template v-else-if="n.kind === 'edge'">
          <ellipse
            :cx="n.x"
            :cy="n.y"
            rx="142"
            ry="77"
            class="monitor-service-region"
          />
          <text :x="n.x" :y="n.y - 56" class="monitor-edge-title">
            E{{ n.id }}
          </text>
          <ellipse
            :cx="n.x"
            :cy="n.y + 45"
            rx="53"
            ry="14"
            class="server-platform"
          />
          <ellipse
            :cx="n.x"
            :cy="n.y + 45"
            rx="43"
            ry="10"
            class="server-platform inner"
          />
          <use
            :href="`#${id}-server`"
            :transform="`translate(${n.x} ${n.y + 12}) scale(.9)`"
          />
          <text :x="n.x" :y="n.y + 70" class="monitor-resource-profile">
            {{ resourceName(n) }}
          </text>
        </template>
        <template v-else>
          <ellipse
            :cx="n.x"
            :cy="n.y + 15"
            rx="29"
            ry="8"
            class="terminal-platform"
          />
          <foreignObject :x="n.x - 28" :y="n.y - 39" width="56" height="48"
            ><NodeGlyph kind="terminal"
          /></foreignObject>
          <text :x="n.x" :y="n.y + 40" class="monitor-terminal-title">
            U{{ n.id }}
          </text>
        </template>
        <title>
          {{ n.label
          }}{{
            n.kind === 'terminal'
              ? ` · 接入 E${n.access_edge} · 端侧只产生任务和收发`
              : ` · CPU/NPU/FPGA ${n.speed.join('/')} GFLOP/s`
          }}
        </title>
      </g>
    </template>
  </svg>
</template>
