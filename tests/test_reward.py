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
    def test_conditional_reject_uses_omega_3(self):
        cfg = replace(load_config("small_smoke"), n_tasks=1, tasks_per_slot=1)
        env = EdgeEnv(cfg)
        env.tasks = [Task(0, 0, 0, .1, .01, .1, .5, .1, .1, 1, False, False)]
        env.idx = 0
        env.observe()
        _, reward, _, info = env.step(len(env.actions) - 1)
        self.assertEqual(info["feasible_count"] > 0, True)
        self.assertEqual(info["reward_constraint_part"], -cfg.reward_penalty_conditional_reject)
        self.assertEqual(reward, -cfg.reward_penalty_conditional_reject)

    def test_forced_reject_has_no_constraint_penalty(self):
        cfg = replace(load_config("small_smoke"), n_tasks=1, tasks_per_slot=1)
        env = EdgeEnv(cfg)
        env.tasks = [Task(0, 0, 0, .1, .01, .1, .5, 1e9, 1e9, 1, True, True)]
        env.idx = 0
        env.observe()
        self.assertEqual(len(env.actions), 1)
        _, reward, _, info = env.step(0)
        self.assertEqual(info["reward_constraint_part"], 0.0)
        self.assertEqual(reward, 0.0)

    def test_invalid_action_uses_omega_2(self):
        cfg = replace(load_config("small_smoke"), n_tasks=1, tasks_per_slot=1)
        env = EdgeEnv(cfg)
        env.reset()
        _, reward, done, info = env.step(-1)
        self.assertTrue(done)
        self.assertTrue(info["invalid_action"])
        self.assertEqual(info["reward_constraint_part"], -cfg.reward_penalty_invalid)
        self.assertEqual(reward, -cfg.reward_penalty_invalid)

    def test_reward_matches_objective_for_three_policies(self):
        selectors = (
            lambda env: 0 if len(env.actions) > 1 else len(env.actions) - 1,
            lambda env: shortest_fastest_action(env.actions),
            lambda env: int(np.random.default_rng(env.idx).integers(len(env.actions))),
        )
        for seed in (3, 7):
            cfg = replace(
                load_config("small_smoke"),
                seed=seed, n_tasks=13, tasks_per_slot=5,
            )
            for selector in selectors:
                metrics = run_episode(cfg, selector)
                self.assertAlmostEqual(
                    metrics["episode_reward"] / metrics["tasks"],
                    metrics["objective_score"], places=6,
                )

    def test_low_cost_accept_is_better_than_reject(self):
        cfg = replace(load_config("small_smoke"), n_tasks=1, tasks_per_slot=1)
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

    def test_accept_all_objective_beats_rejecting_long_delay(self):
        cfg = replace(load_config("default"), seed=1042)

        def accept_all(env):
            return shortest_fastest_action(env.actions)

        def reject_long_delay(env):
            action = shortest_fastest_action(env.actions)
            if action < len(env.actions) - 1 and env.actions[action].delay > 0.9:
                return len(env.actions) - 1
            return action

        accept_metrics = run_episode(cfg, accept_all)
        reject_metrics = run_episode(cfg, reject_long_delay)
        self.assertGreater(accept_metrics["objective_score"], 0.0)
        self.assertGreater(
            accept_metrics["objective_score"], reject_metrics["objective_score"],
        )

    def test_reward_uses_current_task_delay_and_energy(self):
        cfg = replace(load_config("small_smoke"), n_tasks=2, tasks_per_slot=2)
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
        action = next(i for i, item in enumerate(env.actions)
                      if item.destination == -1 and item.resource == 0)
        _, _, _, info = env.step(action)
        rewarded_delay = (
            -info["reward_delay_part"] * cfg.reward_delay_ref / cfg.reward_beta_delay
        )
        rewarded_energy = (
            -info["reward_energy_part"] * cfg.reward_energy_ref / cfg.reward_beta_energy
        )
        self.assertAlmostEqual(rewarded_delay, info["delay"])
        self.assertAlmostEqual(rewarded_energy, info["energy"])

    def test_load_penalty_uses_current_normalized_variance(self):
        cfg = replace(load_config("small_smoke"), n_tasks=5, tasks_per_slot=2)
        env = EdgeEnv(cfg)
        env.reset()
        done = False
        while not done:
            action = 0 if len(env.actions) > 1 else len(env.actions) - 1
            _, _, done, info = env.step(action)
            expected = (
                -cfg.reward_beta_load * info["load_variance"] / cfg.reward_variance_ref
            )
            self.assertAlmostEqual(info["reward_load_part"], expected)


if __name__ == "__main__":
    unittest.main()
