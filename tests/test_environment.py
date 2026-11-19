import unittest
from dataclasses import replace

import numpy as np

from src.config import load_config
from src.environment import EdgeEnv
from src.tasks import Task


class EnvironmentDynamicsTest(unittest.TestCase):
    def test_sparse_edge_profiles_and_custom_node_counts(self):
        services = ((0, 1), (2, 3), (3, 4), (0,), (0, 1), (2, 4),
                    (1, 3), (2,), (0, 2), (3,), (1, 4), (4,))
        speeds = ((150, 520, 150), (150, 150, 520), (420, 180, 180), (120, 120, 120))
        for count in (3, 12, 15):
            cfg = replace(load_config("small_smoke"), n_edges=count)
            env = EdgeEnv(cfg)
            self.assertEqual(env.service_deployment.shape, (count, 5))
            self.assertEqual(env.speed.shape, (count + 1, 3))
            for edge in range(count):
                self.assertEqual(tuple(np.flatnonzero(env.service_deployment[edge])),
                                 services[edge % 12])
                np.testing.assert_array_equal(env.speed[edge], speeds[edge % 4])
            np.testing.assert_array_equal(env.speed[env.cloud], cfg.cloud_speed)

    def test_realized_congestion_uses_each_directed_arcs_current_load(self):
        env = EdgeEnv(load_config("small_smoke"))
        task = Task(0, 0, 0, .1, .01, .1, 1., .1, .1, 1)
        env.tasks = [task]
        env.observe()
        cloud_action = env._candidate(task, env.cloud, 0, (0, env.cloud))
        env.background[0, env.cloud] = .2
        env.background[env.cloud, 0] = .8
        env.execution_rng = np.random.default_rng(7)
        noise = np.random.default_rng(7).uniform(-1, 1, 2)
        realized, _, _, _ = env._sample_uncertainty(task, cloud_action)
        for arc, draw in zip(((0, env.cloud), (env.cloud, 0)), noise):
            rho = env.background[arc]
            expected = np.clip(rho + draw * env.cfg.congestion_noise * (.5 + rho),
                               0, env.cfg.congestion_realized_max)
            self.assertAlmostEqual(realized[arc], expected)

    def test_service_deployment_forces_remote_edge_candidates(self):
        cfg = replace(load_config("small_smoke"), episode_slots=1)
        env = EdgeEnv(cfg)
        env.tasks = [
            Task(0, 0, 3, 1.0, 0.1, 1.0, 10.0, 0.5, 0.5, 1, True, True),
        ]
        env.idx = 0
        env.observe()

        edge_actions = [
            action for action in env.actions
            if 0 <= action.destination < env.cloud
        ]
        self.assertTrue(edge_actions)
        self.assertTrue(all(action.destination != 0 for action in edge_actions))
        self.assertTrue(all(
            env.service_deployment[action.destination, 3]
            for action in edge_actions
        ))
        self.assertTrue(all(len(action.path) > 1 for action in edge_actions))
        self.assertTrue(any(action.destination == env.cloud for action in env.actions))
        self.assertFalse(any(action.destination == -1 for action in env.actions))

    def test_kkt_reallocation_updates_previous_task(self):
        cfg = replace(
            load_config("small_smoke"), episode_slots=1,
            compute_degradation_max=0.0,
        )
        env = EdgeEnv(cfg)
        env.tasks = [
            Task(0, 0, 0, 1.0, 0.1, 1.0, 0.5, 0.5, 0.5, 1),
            Task(0, 0, 0, 1.0, 0.1, 1.0, 0.5, 0.5, 0.5, 1),
        ]
        env.idx = 0
        env.observe()
        first_action = next(
            index for index, action in enumerate(env.actions)
            if action.destination == 0 and action.resource == 0
        )
        _, first_reward, _, _ = env.step(first_action)
        previous_share = env.slot_assignments[0].action.compute_share
        previous_delay = env.slot_assignments[0].action.delay

        second_action = next(
            index for index, action in enumerate(env.actions)
            if action.destination == 0 and action.resource == 0
        )
        _, second_reward, _, _ = env.step(second_action)

        first = env.slot_assignments[0].action
        self.assertLess(first.compute_share, previous_share)
        self.assertGreater(first.delay, previous_delay)
        self.assertAlmostEqual(env.slot_resource_used[0, 0], env.speed[0, 0])
        self.assertAlmostEqual(
            env.sum_delay,
            sum(assignment.action.delay for assignment in env.slot_assignments),
        )
        self.assertAlmostEqual(
            env.total_energy,
            sum(assignment.action.energy for assignment in env.slot_assignments),
        )
        self.assertAlmostEqual(
            env.metrics()["avg_energy_j"], env.total_energy / env.accepted,
        )
        metrics = env.metrics()
        self.assertAlmostEqual(first_reward + second_reward, metrics["episode_reward"])
        self.assertAlmostEqual(metrics["reward_sum_check"], metrics["objective_score"])

    def test_generated_actions_keep_core_constraints(self):
        env = EdgeEnv(load_config("small_smoke"))
        np.testing.assert_array_equal(
            env.speed[:env.cloud], np.array(env.cfg.edge_speed_matrix),
        )
        env.reset()
        done = False
        while not done:
            task = env.tasks[env.idx]
            for action in env.actions[:-1]:
                self.assertLessEqual(action.delay, task.deadline_s + 1e-12)
                self.assertTrue(env.support[action.resource, task.kind])
                self.assertGreaterEqual(action.destination, 0)
                self.assertLessEqual(action.destination, env.cloud)
                self.assertLessEqual(action.uplink_power, env.cfg.uplink_max_power_w)
                if action.destination < env.cloud:
                    self.assertTrue(env.service_deployment[action.destination, task.kind])
                    self.assertLessEqual(len(action.path) - 1, env.cfg.max_hops)
                    self.assertEqual(len(set(action.path)), len(action.path))
                self.assertEqual(action.path[0], task.access_edge)
                self.assertEqual(action.path[-1], action.destination)
                for source, target in zip(action.path, action.path[1:]):
                    self.assertTrue(env.adj[source, target])
            action_id = max(
                range(len(env.actions)),
                key=lambda index: (env.actions[index].destination != -2,
                                   len(env.actions[index].path)),
            )
            _, _, done, _ = env.step(action_id)
            self.assertTrue(np.all(env.slot_resource_used <= env.speed + 1e-5))
            self.assertTrue(np.all(env.slot_mem_used <= env.mem_cap + 1e-5))
            for source in range(env.n_graph_nodes):
                for target in range(env.n_graph_nodes):
                    if source != target and env.adj[source, target]:
                        self.assertLessEqual(
                            env.slot_link_used[source, target],
                            env._slot_capacity_mb(source, target) + 1e-5,
                        )
            for assignment in env.slot_assignments:
                self.assertLessEqual(
                    assignment.action.estimated_delay,
                    env.tasks[assignment.task_index].deadline_s + 1e-12,
                )

    def test_no_success_delay_is_null(self):
        cfg = replace(load_config("small_smoke"), episode_slots=2)
        env = EdgeEnv(cfg)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(len(env.actions) - 1)
        metrics = env.metrics()
        self.assertIsNone(metrics["avg_delay_s"])
        self.assertIsNone(metrics["avg_energy_j"])
        self.assertIsNone(metrics["priority_weighted_delay_s"])
        self.assertIsNone(metrics["mean_hops_accepted"])
        self.assertGreater(metrics["load_balance_degree"], 0.0)
        self.assertLessEqual(metrics["load_balance_degree"], 1.0)

    def test_carryover_resource_is_reserved_then_released(self):
        cfg = replace(
            load_config("small_smoke"), episode_slots=3,
            compute_degradation_max=0.0,
        )
        env = EdgeEnv(cfg)
        env.tasks = [
            Task(3, 3, 0, .1, .01, 150., 3., .5, .5, 1, slot=0),
            Task(0, 0, 0, .1, .01, .1, 1., 150., .1, 1, slot=1),
            Task(0, 0, 0, .1, .01, .1, 1., 150., .1, 1, slot=2),
        ]
        env.idx = 0
        env.observe()
        action = next(
            index for index, item in enumerate(env.actions)
            if item.destination == 3 and item.resource == 0
        )
        env.step(action)
        self.assertEqual(env.slot, 1)
        self.assertAlmostEqual(env.carryover_resource_used[3, 0], 120.0)
        self.assertAlmostEqual(env.slot_mem_used[3], .5)

        env.step(len(env.actions) - 1)
        self.assertEqual(env.slot, 2)
        self.assertAlmostEqual(env.carryover_resource_used[3, 0], 0.0)
        self.assertAlmostEqual(env.slot_mem_used[3], 0.0)


if __name__ == "__main__":
    unittest.main()
