export const keys = { env: 'st_env_draft', train: 'st_train_draft', run: 'st_active_run_id', layout: 'st_topology_layout', prefs: 'st_ui_prefs', dispatchScenario: 'st_dispatch_scenario', completedDispatch: 'st_completed_dispatch' }
export const storage = {
  get(key, fallback) { const value = localStorage.getItem(keys[key]); return value === null ? fallback : JSON.parse(value) },
  set(key, value) { localStorage.setItem(keys[key], JSON.stringify(value)) },
  remove(key) { localStorage.removeItem(keys[key]) }
}
