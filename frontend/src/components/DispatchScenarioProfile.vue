<script setup>
defineProps({ config: { type: Object, required: true } })
const percent = share => Math.round(100 * share) + '%'
</script>

<template>
  <dl class="dispatch-scenario-profile" aria-label="场景配置">
    <div>
      <dt>基础到达率</dt>
      <dd>{{ config.arrival_rate }} 个/时隙，{{ config.episode_slots }} 个时隙</dd>
    </div>
    <div>
      <dt>基准拥塞</dt>
      <dd>{{ percent(config.congestion_min) }}–{{ percent(config.congestion_max) }}，断链概率 {{ percent(config.link_flap_prob) }}</dd>
    </div>
    <div>
      <dt>边缘 / 边云基准速率</dt>
      <dd>{{ config.link_rate_min_mbps }}–{{ config.link_rate_max_mbps }} / {{ config.cloud_rate_min_mbps }}–{{ config.cloud_rate_max_mbps }} Mbps</dd>
    </div>
  </dl>
</template>
