import { createRouter, createWebHistory } from 'vue-router'
import TrainingWorkbench from '../views/TrainingWorkbench.vue'
import SimulationSandbox from '../views/SimulationSandbox.vue'
import CompareLab from '../views/CompareLab.vue'
import ExperimentHistory from '../views/ExperimentHistory.vue'
export default createRouter({ history: createWebHistory(), routes: [
  { path: '/', component: TrainingWorkbench }, { path: '/sandbox', component: SimulationSandbox },
  { path: '/compare', component: CompareLab }, { path: '/history', component: ExperimentHistory }
] })
