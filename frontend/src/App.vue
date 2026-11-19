<script setup>
import { onMounted, onUnmounted } from 'vue'
import { useAppStore } from './stores/app'
const app = useAppStore(); let timer
onMounted(() => { app.checkHealth(); app.loadPresets(); app.loadCheckpoints(); timer = setInterval(() => app.checkHealth(), 30000) })
onUnmounted(() => clearInterval(timer))
</script>
<template><div class="shell" :class="{ collapsed: app.prefs.collapsed }"><aside class="sidebar"><div class="brand"><span class="brand-mark">S<span>·</span></span><div v-if="!app.prefs.collapsed"><b>SCHEDULE</b><small>云边端调度实验平台</small></div></div><div class="nav-label" v-if="!app.prefs.collapsed">WORKSPACE / 工作区</div><nav><router-link to="/" exact-active-class="active"><span>◫</span><em>训练工作台</em></router-link><router-link to="/sandbox"><span>⌘</span><em>仿真推演沙盘</em></router-link><router-link to="/compare"><span>▥</span><em>对比评估</em></router-link><router-link to="/history"><span>▤</span><em>实验记录</em></router-link></nav><div class="sidebar-foot"><button @click="app.prefs.collapsed = !app.prefs.collapsed">{{ app.prefs.collapsed ? '展开 →' : '← 收起导航' }}</button></div></aside><div class="workspace"><header class="topbar"><div><span class="top-kicker">RESEARCH CONSOLE</span><span class="top-divider">/</span> 动态多跳调度</div><div class="status-group"><span><i :class="app.health ? 'online' : 'offline'"></i>后端 {{ app.health ? '已连接' : '未连接' }}</span><span><i :class="app.health?.db_connected ? 'online' : 'offline'"></i>MySQL {{ app.health?.db_connected ? '已连接' : '未连接' }}</span></div></header><main><router-view /></main></div></div></template>
