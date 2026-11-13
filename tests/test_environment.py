import unittest
from dataclasses import replace

import numpy as np

from src.config import load_config
from src.environment import EdgeEnv
from src.tasks import Task


class EnvironmentDynamicsTest(unittest.TestCase):
    def test_service_deployment_forces_remote_edge_candidates(self):
        cfg = replace(load_config("configs/small_smoke.json"), n_tasks=1)
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
        cfg = replace(load_config("configs/small_smoke.json"), n_tasks=2, tasks_per_slot=2)
        env = EdgeEnv(cfg)
        env.tasks = [
            Task(1, 1, 0, 1.0, 0.1, 1.0, 0.5, 0.5, 0.5, 1, False, False),
            Task(1, 1, 0, 1.0, 0.1, 1.0, 0.5, 0.5, 0.5, 1, False, False),
        ]
        env.idx = 0
        env.observe()
        first_action = next(
            index for index, action in enumerate(env.actions)
            if action.destination == -1 and action.resource == 0
        )
        _, first_reward, _, _ = env.step(first_action)
        previous_share = env.slot_assignments[0].action.compute_share
        previous_delay = env.slot_assignments[0].action.delay

        second_action = next(
            index for index, action in enumerate(env.actions)
            if action.destination == -1 and action.resource == 0
        )
        _, second_reward, _, _ = env.step(second_action)

        first = env.slot_assignments[0].action
        self.assertLess(first.compute_share, previous_share)
        self.assertGreater(first.delay, previous_delay)
        self.assertAlmostEqual(env.slot_local_resource_used[1, 0], env.local_speed[0])
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
        env = EdgeEnv(load_config("configs/small_smoke.json"))
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
                if action.destination == -1:
                    self.assertEqual(action.path, ())
                else:
                    self.assertLessEqual(action.uplink_power, env.cfg.uplink_max_power_w)
                    if action.destination < env.cloud:
                        self.assertTrue(
                            env.service_deployment[action.destination, task.kind]
                        )
                    for source, target in zip(action.path, action.path[1:]):
                        self.assertTrue(env.adj[source, target])
            action_id = max(
                range(len(env.actions)),
                key=lambda index: (env.actions[index].destination != -2,
                                   len(env.actions[index].path)),
            )
            _, _, done, _ = env.step(action_id)
            self.assertTrue(np.all(env.slot_resource_used <= env.speed + 1e-5))
            self.assertTrue(np.all(env.slot_local_resource_used <= env.local_speed + 1e-5))
            self.assertTrue(np.all(env.slot_mem_used <= env.mem_cap + 1e-5))
            self.assertTrue(np.all(env.slot_local_mem_used <= env.local_mem_cap + 1e-5))
            for source in range(env.n_graph_nodes):
                for target in range(env.n_graph_nodes):
                    if source != target and env.adj[source, target]:
                        self.assertLessEqual(
                            env.slot_link_used[source, target],
                            env._slot_capacity_mb(source, target) + 1e-5,
                        )
            for assignment in env.slot_assignments:
                self.assertLessEqual(
                    assignment.action.delay,
                    env.tasks[assignment.task_index].deadline_s + 1e-12,
                )

    def test_no_success_delay_is_null(self):
        cfg = replace(load_config("configs/small_smoke.json"), n_tasks=2)
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


if __name__ == "__main__":
    unittest.main()
