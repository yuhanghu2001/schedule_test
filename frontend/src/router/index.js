import { createRouter, createWebHistory } from 'vue-router'
import TopologyMonitor from '../views/TopologyMonitor.vue'
import NodeManagement from '../views/NodeManagement.vue'
import OverviewDashboard from '../views/OverviewDashboard.vue'
import TrainingWorkbench from '../views/TrainingWorkbench.vue'
import SimulationSandbox from '../views/SimulationSandbox.vue'
import CompareLab from '../views/CompareLab.vue'
import ExperimentHistory from '../views/ExperimentHistory.vue'
export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: OverviewDashboard },
    { path: '/logs', component: () => import('../views/TrainingLogs.vue') },
    { path: '/dispatch-logs', component: () => import('../views/DispatchLogs.vue') },
    { path: '/nodes', component: NodeManagement },
    { path: '/topology', component: TopologyMonitor },
    { path: '/resources', component: OverviewDashboard },
    { path: '/training', component: TrainingWorkbench },
    { path: '/sandbox', component: SimulationSandbox },
    { path: '/compare', component: CompareLab },
    { path: '/history', component: ExperimentHistory },
  ],
})
