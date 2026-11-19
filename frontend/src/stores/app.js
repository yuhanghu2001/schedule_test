import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import api from '../api/client'
import { storage } from '../utils/storage'
export const useAppStore = defineStore('app', () => {
  const env = ref(storage.get('env', null)), train = ref(storage.get('train', { model_type: 'gat', episodes: 1000, seed: 42, device: 'cpu', threads: 2, resume_path: null }))
  const activeRun = ref(storage.get('run', null)), prefs = ref(storage.get('prefs', { collapsed: false, speed: 1 })), presets = ref({}), checkpoints = ref([]), health = ref(null)
  watch(env, v => storage.set('env', v), { deep: true }); watch(train, v => storage.set('train', v), { deep: true }); watch(activeRun, v => storage.set('run', v)); watch(prefs, v => storage.set('prefs', v), { deep: true })
  async function loadPresets() { const data = await api.get('/env/presets'); presets.value = data.presets; if (!env.value) env.value = { preset_name: 'default', config: structuredClone(data.presets.default) } }
  async function loadCheckpoints() { checkpoints.value = await api.get('/training/checkpoints') }
  async function checkHealth() { try { health.value = await api.get('/health') } catch { health.value = null } }
  function resetEnv() { env.value = { preset_name: 'default', config: structuredClone(presets.value.default) } }
  function resetTrain() { train.value = { model_type: 'gat', episodes: 1000, seed: 42, device: 'cpu', threads: 2, resume_path: null } }
  return { env, train, activeRun, prefs, presets, checkpoints, health, loadPresets, loadCheckpoints, checkHealth, resetEnv, resetTrain }
})
