<script setup>
import { computed, ref, watch } from 'vue'
import { storage } from '../utils/storage'
const props = defineProps({ snapshot: Object, selected: Object })
const emit = defineEmits(['select-node'])
const svg = ref(null), dragging = ref(null), positions = ref({}), layouts = storage.get('layout', {})
const bucket = computed(() => String(props.snapshot?.nodes?.filter(n => n.kind === 'edge').length || 0))
watch(bucket, key => { positions.value = { ...(layouts[key] || {}) } }, { immediate: true })
const items = computed(() => {
  if (!props.snapshot) return []
  const edges = props.snapshot.nodes.filter(n => n.kind === 'edge'), terminals = props.snapshot.terminals || [], cloud = props.snapshot.nodes.find(n => n.kind === 'cloud')
  const nodes = []
  if (cloud) nodes.push({ ...cloud, key: `n${cloud.id}`, x: 500, y: 240 })
  edges.forEach((n, i) => { const a = 2 * Math.PI * i / edges.length - Math.PI / 2; nodes.push({ ...n, key: `n${n.id}`, x: 500 + 225 * Math.cos(a), y: 240 + 150 * Math.sin(a) }) })
  terminals.forEach((n, i) => { const a = 2 * Math.PI * i / terminals.length - Math.PI / 2; nodes.push({ ...n, kind: 'terminal', key: `t${n.id}`, x: 500 + 435 * Math.cos(a), y: 240 + 215 * Math.sin(a) }) })
  return nodes.map(n => ({ ...n, ...(positions.value[n.key] || {}) }))
})
const map = computed(() => Object.fromEntries(items.value.map(n => [n.key, n])))
const chosen = computed(() => props.selected || props.snapshot?.last_decision)
function highlighted(a, b) { const path = chosen.value?.path || []; return path.some((n, i) => i && ((n === a && path[i-1] === b) || (n === b && path[i-1] === a))) }
function radioHighlighted(t) { return chosen.value && chosen.value.source === t.id && chosen.value.access_edge === t.access_edge }
function color(l) { if (l.status === 'flapped') return '#9ca9b5'; const v = Math.min(1, l.background_fwd + (l.capacity_fwd_mb ? l.used_fwd_mb / l.capacity_fwd_mb : 0)); return v > .7 ? '#e25d57' : v > .4 ? '#e5a34d' : '#53ad96' }
function start(e, n) { dragging.value = n; e.currentTarget.setPointerCapture(e.pointerId) }
function move(e) { if (!dragging.value) return; const p = svg.value.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; const point = p.matrixTransform(svg.value.getScreenCTM().inverse()); positions.value = { ...positions.value, [dragging.value.key]: { x: point.x, y: point.y } } }
function end() { if (!dragging.value) return; layouts[bucket.value] = positions.value; storage.set('layout', layouts); dragging.value = null }
</script>
<template><div class="topology"><div v-if="!snapshot" class="empty-graph">生成预览或创建会话后，拓扑将显示在这里</div><svg v-else ref="svg" viewBox="0 0 1000 480" @pointermove="move" @pointerup="end" @pointercancel="end">
<g v-for="t in snapshot.terminals" :key="'r'+t.id"><line v-if="map['t'+t.id] && map['n'+t.access_edge]" :x1="map['t'+t.id].x" :y1="map['t'+t.id].y" :x2="map['n'+t.access_edge].x" :y2="map['n'+t.access_edge].y" :class="{ 'route-line': radioHighlighted(t) }" stroke="#cedbe0" stroke-width="1"/><title>{{ t.radio_rate_mbps.toFixed(1) }} Mbps</title></g>
<g v-for="(l,i) in snapshot.links" :key="i"><template v-if="map['n'+l.source] && map['n'+l.target]"><line :x1="map['n'+l.source].x" :y1="map['n'+l.source].y" :x2="map['n'+l.target].x" :y2="map['n'+l.target].y" :stroke="color(l)" :stroke-width="Math.max(1.2, Math.min(4,l.rate_mbps / 500))" :stroke-dasharray="l.status === 'flapped' ? '5 5' : ''" :class="{ 'route-line': highlighted(l.source,l.target) }"/><text class="link-label" :x="(map['n'+l.source].x+map['n'+l.target].x)/2" :y="(map['n'+l.source].y+map['n'+l.target].y)/2">{{ Math.round(l.rate_mbps) }} · Δ{{ l.congestion_delta_fwd.toFixed(2) }}</text><title>{{ l.rate_mbps.toFixed(1) }} Mbps · 拥塞 {{ (100*(l.background_fwd + (l.capacity_fwd_mb ? l.used_fwd_mb/l.capacity_fwd_mb : 0))).toFixed(1) }}%</title></template></g>
<g v-for="n in items" :key="n.key" class="graph-node" @pointerdown="start($event,n)" @click="emit('select-node',n)"><circle :cx="n.x" :cy="n.y" :r="n.kind === 'cloud' ? 23 : n.kind === 'edge' ? 17 : 6" :class="['node-circle', n.kind]"/><circle v-if="n.kind === 'edge'" :cx="n.x" :cy="n.y" r="20" fill="none" :stroke="n.resource_used.some((v,i)=>v/n.speed[i] > .7) ? '#e5a34d' : '#abd6cd'" stroke-width="2"/><text v-if="n.kind !== 'terminal'" class="node-name" :x="n.x" :y="n.y+4">{{ n.kind === 'cloud' ? '云' : 'E'+n.id }}</text><text v-if="n.kind === 'edge'" class="node-detail" :x="n.x" :y="n.y+33">{{ n.profile }} · {{ n.services.map(s=>'T'+s).join(' ') }}</text><text v-if="n.kind === 'terminal'" class="terminal-label" :x="n.x" :y="n.y-10">U{{ n.id }} · {{ Math.round(n.radio_rate_mbps) }}</text><title>{{ n.label }}{{ n.radio_rate_mbps ? ` · ${n.radio_rate_mbps.toFixed(1)} Mbps` : '' }}</title></g></svg><div class="graph-legend"><span><i style="background:#53ad96"></i>低拥塞</span><span><i style="background:#e5a34d"></i>中拥塞</span><span><i style="background:#e25d57"></i>高拥塞</span><span>虚线：瞬断</span><span>拖动节点调整布局</span></div></div></template>
