<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { useAppStore } from './stores/app'
const route = useRoute()
const app = useAppStore(),
  now = ref(new Date())
let timer
const clock = () =>
  now.value.toLocaleString('sv-SE', { timeZone: 'Asia/Shanghai' })
onMounted(() => {
  app.checkHealth()
  app.loadPresets()
  app.loadCheckpoints()
  timer = setInterval(() => {
    now.value = new Date()
    if (now.value.getSeconds() % 30 === 0) app.checkHealth()
  }, 1000)
})
onUnmounted(() => clearInterval(timer))
</script>
<template>
  <div class="console-shell">
    <header class="screen-header">
      <div class="title-frame"><h1>云边端异构资源协同调度系统</h1></div>
      <div class="system-status">
        <span class="status-pill" :class="{ disconnected: !app.health }">{{
          app.health ? '系统已连接' : '后端未连接'
        }}</span><time>{{ clock() }}</time>
      </div>
    </header>
    <nav class="screen-nav" aria-label="主导航">
      <router-link to="/" exact-active-class="active">系统总览</router-link
      ><router-link to="/logs" active-class="active">模型训练</router-link
      ><router-link to="/dispatch-logs" active-class="active">调度分析</router-link
      ><router-link to="/topology" active-class="active">拓扑监控</router-link
      ><router-link to="/nodes" active-class="active">节点管理</router-link>
      <div class="secondary-nav">
        <span :class="app.health?.db_connected ? 'good' : 'warn'"
          >MySQL {{ app.health?.db_connected ? '已连接' : '未连接' }}</span
        >
      </div>
    </nav>
    <main
      class="console-main"
      :class="{ 'topology-main': route.path === '/topology' }"
    >
      <router-view />
    </main>
  </div>
</template>
