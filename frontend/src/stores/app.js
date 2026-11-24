import { defineStore } from 'pinia'
import { computed, toRaw, ref, watch } from 'vue'
import api from '../api/client'
import { storage } from '../utils/storage'

const nodeConfigKeys = [
  'n_edges', 'n_terminals', 'edge_speed_matrix', 'cloud_speed', 'service_deployment',
  'node_memory_gb', 'node_storage_gb', 'node_queue_capacity', 'terminal_radio_rates_mbps',
]

export const useAppStore = defineStore('app', () => {
  const env = ref(storage.get('env', null)), train = ref(storage.get('train', { model_type: 'gatppo', episodes: 1000, seed: 42, device: 'cpu', threads: 2, resume_path: null }))
  const activeRun = ref(storage.get('run', null)), prefs = ref(storage.get('prefs', { collapsed: false, speed: 1 })), presets = ref({}), checkpoints = ref([]), health = ref(null)
  const dispatchScenario = ref(storage.get('dispatchScenario', 'default'))
  const dispatchHistory = ref(storage.get('completedDispatch', []))
  const completedDispatch = computed(() => dispatchHistory.value.at(-1)?.runs ?? {})
  // 节点草稿只提供硬件配置，任务流及链路条件由所选场景确定。
  const dispatchEnv = computed(() => {
    const scene = presets.value[dispatchScenario.value]
    if (!scene) return null
    const nodes = scene.enable_dag_mode ? {} : env.value.config
    const nodeConfig = Object.fromEntries(nodeConfigKeys.filter(key => key in nodes).map(key => [key, nodes[key]]))
    return { preset_name: dispatchScenario.value, config: { ...scene, ...nodeConfig }, seed: scene.seed }
  })
  watch(dispatchScenario, value => storage.set('dispatchScenario', value))
  function recordDispatchRun(run) {
    const previous = dispatchHistory.value.find(group => group.environment === run.environment)
    // 只有环境和种子相同的已完成推演才能一起比较。
    const group = {
      environment: run.environment,
      runs: { ...previous?.runs, [run.algorithm]: { checkpointPath: run.checkpointPath, samples: run.samples } },
    }
    dispatchHistory.value = [...dispatchHistory.value.filter(item => item.environment !== run.environment), group]
    storage.set('completedDispatch', dispatchHistory.value)
  }
  if (train.value.model_type === 'gat') train.value.model_type = 'gatppo'
  if (train.value.model_type === 'flat') train.value.model_type = 'ppo'
  watch(env, v => storage.set('env', v), { deep: true }); watch(train, v => storage.set('train', v), { deep: true }); watch(activeRun, v => storage.set('run', v)); watch(prefs, v => storage.set('prefs', v), { deep: true })
  async function loadPresets() { const data = await api.get('/env/presets', { params: { include_dag: true } }); presets.value = data.presets; if (!env.value) env.value = { preset_name: 'default', config: structuredClone(data.presets.default) } }
  async function loadCheckpoints() { checkpoints.value = await api.get('/training/checkpoints') }
  async function checkHealth() { try { health.value = await api.get('/health') } catch { health.value = null } }
  function resetEnv() { env.value = { preset_name: 'default', config: structuredClone(toRaw(presets.value.default)) } }
  function resetTrain() { train.value = { model_type: 'gatppo', episodes: 1000, seed: 42, device: 'cpu', threads: 2, resume_path: null } }
  return { env, train, activeRun, prefs, presets, checkpoints, health, dispatchScenario, dispatchEnv, completedDispatch, recordDispatchRun, loadPresets, loadCheckpoints, checkHealth, resetEnv, resetTrain }
})
