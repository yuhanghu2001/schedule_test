import unittest
from dataclasses import replace

import numpy as np

from src.baselines import shortest_fastest_action
from src.config import load_config
from src.environment import EdgeEnv
from src.tasks import Task


def run_episode(cfg, selector):
    env = EdgeEnv(cfg)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(selector(env))
    return env.metrics()


class AdditiveRewardTest(unittest.TestCase):
    def test_all_reject_is_zero(self):
        cfg = replace(load_config("configs/small_smoke.json"), n_tasks=5, tasks_per_slot=2)
        metrics = run_episode(cfg, lambda env: len(env.actions) - 1)
        self.assertEqual(metrics["task_success_rate"], 0.0)
        self.assertEqual(metrics["episode_reward"], 0.0)
        self.assertEqual(metrics["objective_score"], 0.0)
        self.assertIsNone(metrics["avg_delay_s"])
        self.assertIsNone(metrics["avg_energy_j"])

    def test_reward_matches_objective_for_three_policies(self):
        selectors = (
            lambda env: 0 if len(env.actions) > 1 else len(env.actions) - 1,
            lambda env: shortest_fastest_action(env.actions),
            lambda env: int(np.random.default_rng(env.idx).integers(len(env.actions))),
        )
        for seed in (3, 7):
            cfg = replace(
                load_config("configs/small_smoke.json"),
                seed=seed, n_tasks=13, tasks_per_slot=5,
            )
            for selector in selectors:
                metrics = run_episode(cfg, selector)
                self.assertAlmostEqual(
                    metrics["episode_reward"] / metrics["tasks"],
                    metrics["objective_score"], places=6,
                )

    def test_low_cost_accept_is_better_than_reject(self):
        cfg = replace(load_config("configs/small_smoke.json"), n_tasks=1, tasks_per_slot=1)
        task = Task(0, 0, 0, .1, .01, .1, .5, .1, .1, 1, False, False)
        accept_env = EdgeEnv(cfg)
        accept_env.tasks = [task]
        accept_env.idx = 0
        accept_env.observe()
        accept_action = next(
            index for index, action in enumerate(accept_env.actions)
            if action.destination == -1 and action.resource == 0
        )
        _, accept_reward, _, _ = accept_env.step(accept_action)

        reject_env = EdgeEnv(cfg)
        reject_env.tasks = [task]
        reject_env.idx = 0
        reject_env.observe()
        _, reject_reward, _, _ = reject_env.step(len(reject_env.actions) - 1)
        self.assertGreater(accept_reward, reject_reward)

    def test_group_reallocation_uses_total_deltas(self):
        cfg = replace(load_config("configs/small_smoke.json"), n_tasks=2, tasks_per_slot=2)
        env = EdgeEnv(cfg)
        env.tasks = [
            Task(1, 1, 0, 1., .1, 1., .5, .5, .5, 1, False, False),
            Task(1, 1, 0, 1., .1, 1., .5, .5, .5, 1, False, False),
        ]
        env.idx = 0
        env.observe()
        action = next(i for i, item in enumerate(env.actions)
                      if item.destination == -1 and item.resource == 0)
        env.step(action)
        before_delay = env.sum_delay
        before_energy = env.total_energy
        action = next(i for i, item in enumerate(env.actions)
                      if item.destination == -1 and item.resource == 0)
        _, _, _, info = env.step(action)
        delta_delay = -info["reward_delay_part"] * cfg.reward_delay_ref / cfg.reward_delay
        delta_energy = -info["reward_energy_part"] * cfg.reward_energy_ref / cfg.reward_energy
        self.assertAlmostEqual(delta_delay, env.sum_delay - before_delay)
        self.assertAlmostEqual(delta_energy, env.total_energy - before_energy)
        self.assertGreater(delta_delay, info["delay"])

    def test_partial_last_slot_balance_telescopes(self):
        cfg = replace(load_config("configs/small_smoke.json"), n_tasks=5, tasks_per_slot=2)
        env = EdgeEnv(cfg)
        env.reset()
        balance_reward = 0.0
        done = False
        while not done:
            action = 0 if len(env.actions) > 1 else len(env.actions) - 1
            _, _, done, info = env.step(action)
            balance_reward += info["reward_balance_part"]
        slot_count = 3
        expected = (
            -cfg.reward_balance * (cfg.n_tasks / slot_count)
            * env.total_load_variance / cfg.reward_variance_ref
        )
        self.assertAlmostEqual(balance_reward, expected)


if __name__ == "__main__":
    unittest.main()
