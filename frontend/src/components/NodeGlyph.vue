<script setup>
import { useId } from 'vue'
const id = useId().replace(/:/g, '')
defineProps({ kind: { type: String, default: 'edge' } })
</script>
<template>
  <svg viewBox="0 0 70 56" class="node-glyph" aria-hidden="true">
    <defs>
      <linearGradient :id="id + '-server'" x2="0" y2="1">
        <stop stop-color="#b5efff" />
        <stop offset=".45" stop-color="#299fe7" />
        <stop offset="1" stop-color="#08477c" />
      </linearGradient>
      <linearGradient :id="id + '-cloud'" x2="0" y2="1">
        <stop stop-color="#8ceaff" />
        <stop offset="1" stop-color="#0086ee" />
      </linearGradient>
    </defs>
    <path
      v-if="kind === 'cloud'"
      d="M15 45C0 44 0 24 15 23C14 3 44 0 49 21C70 16 77 45 57 46Z"
      :fill="`url(#${id}-cloud)`"
      stroke="#8ce9ff"
      stroke-width="2"
    />
    <g
      v-if="kind !== 'terminal'"
      :transform="
        kind === 'cloud' ? 'translate(14 13) scale(.6)' : 'translate(6 7)'
      "
    >
      <g v-for="y in [0, 17]" :key="y">
        <rect
          x="0"
          :y="y"
          width="58"
          height="14"
          rx="3"
          :fill="`url(#${id}-server)`"
          stroke="#5fcbff"
        />
        <rect x="6" :y="y + 4" width="24" height="6" rx="1" fill="#052447" />
        <path :d="`M9 ${y + 6}h4m3 0h4m3 0h4`" stroke="#75bfff" />
        <circle cx="43" :cy="y + 7" r="2" fill="#66ffcc" />
        <circle cx="51" :cy="y + 7" r="2" fill="#b8f7ff" />
      </g>
    </g>
    <g v-else transform="translate(15 10)">
      <path
        d="M3 4h38v23H3z"
        fill="#0d4d81"
        stroke="#67d4ff"
        stroke-width="2"
      />
      <path d="M8 9h28v12H8z" fill="#b1e9ff" />
      <path d="M16 28v6m12-6v6M11 35h22" stroke="#4dc9ff" stroke-width="3" />
    </g>
  </svg>
</template>
