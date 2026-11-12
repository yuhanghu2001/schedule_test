import unittest

import numpy as np

from src.continuous import allocate_compute, choose_uplink_power, uplink_rate
from src.config import Config
from src.environment import EdgeEnv
from src.tasks import Task


class ContinuousOptimizationTest(unittest.TestCase):
    def test_kkt_allocation_with_lower_bound(self):
        allocation = allocate_compute(
            compute_gflops=[1.0, 4.0, 9.0],
            priorities=[1.0, 1.0, 1.0],
            capacity_gflops=12.0,
            minimum_gflops=[4.0, 1.0, 1.0],
        )
        np.testing.assert_allclose(allocation, [4.0, 3.2, 4.8])
        free_lambda = np.array([4.0, 9.0]) / allocation[1:] ** 2
        np.testing.assert_allclose(free_lambda[0], free_lambda[1])

    def test_kkt_detects_infeasible_minimum_capacity(self):
        allocation = allocate_compute(
            compute_gflops=[1.0, 1.0],
            priorities=[1.0, 1.0],
            capacity_gflops=5.0,
            minimum_gflops=[3.0, 3.0],
        )
        self.assertIsNone(allocation)

    def test_dinkelbach_residual_and_energy_efficiency(self):
        solution = choose_uplink_power(
            bandwidth_mhz=20.0,
            channel_gain_to_noise=50.0,
            circuit_power_w=0.1,
            max_power_w=0.5,
            min_rate_mbps=40.0,
            tolerance=1e-8,
            max_iterations=50,
        )
        self.assertLessEqual(abs(solution.residual), 1e-8)
        self.assertGreaterEqual(solution.rate_mbps, 40.0)
        for power in (solution.power_w - 1e-4, solution.power_w + 1e-4):
            efficiency = uplink_rate(power, 20.0, 50.0) / (power + 0.1)
            self.assertLessEqual(efficiency, solution.efficiency_mbit_per_j)


class EnvironmentContinuousIntegrationTest(unittest.TestCase):
    @staticmethod
    def local_task():
        return Task(
            source=0, access_edge=0, kind=0, data_mb=0.5, output_mb=0.05,
            compute_gflop=1.0, deadline_s=0.5, memory_gb=0.5,
            storage_gb=0.5, priority=1, allow_offload=False,
        )

    def test_local_group_reallocation_updates_existing_metrics(self):
        env = EdgeEnv(Config(n_tasks=2, tasks_per_slot=2, seed=3))
        env.tasks = [self.local_task(), self.local_task()]
        env.idx = 0
        env.observe()
        first_id = next(i for i, action in enumerate(env.actions)
                        if action.destination == -1 and action.resource == 0)
        env.step(first_id)
        first_assignment = env.slot_assignments[0]
        first_delay_before = first_assignment.action.delay
        second_id = next(i for i, action in enumerate(env.actions)
                         if action.destination == -1 and action.resource == 0)
        env.step(second_id)
        self.assertGreater(first_assignment.action.delay, first_delay_before)
        np.testing.assert_allclose(
            [assignment.action.compute_share for assignment in env.slot_assignments],
            [6.0, 6.0],
        )
        expected_delay = sum(item.action.delay for item in env.slot_assignments) / 2
        self.assertAlmostEqual(env.metrics()["avg_delay_s"], expected_delay)
        self.assertAlmostEqual(env.slot_local_mem_used[0], 1.0)

    def test_return_traffic_uses_reverse_capacity_and_cloud_is_not_edge_hop(self):
        env = EdgeEnv(Config(n_tasks=1, tasks_per_slot=1, seed=4))
        task = Task(
            source=0, access_edge=0, kind=0, data_mb=0.1, output_mb=0.05,
            compute_gflop=0.1, deadline_s=0.5, memory_gb=0.2,
            storage_gb=0.2, priority=1, allow_offload=True,
        )
        env.tasks = [task]
        env.idx = 0
        action = env._candidate(task, env.cloud, 0, (0, env.cloud))
        env.actions = [action]
        env.step(0)
        self.assertAlmostEqual(env.slot_link_used[0, env.cloud], task.data_mb)
        self.assertAlmostEqual(env.slot_link_used[env.cloud, 0], task.output_mb)
        self.assertEqual(env.total_hops, 0)
        self.assertGreater(action.uplink_power, 0.0)


if __name__ == "__main__":
    unittest.main()
