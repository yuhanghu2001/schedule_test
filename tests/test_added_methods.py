from dataclasses import replace
import unittest

import numpy as np
import torch

from src.algorithms.ddpg import DDPG, DDPGTrainer
from src.algorithms.dqn import DQN, DQNTrainer
from src.baselines import evaluate_heuristic, slot_end, slot_model
from src.config import load_config
from src.environment import EdgeEnv
from src.evaluation import evaluate_policy


class RecordingEnv(EdgeEnv):
    def reset(self):
        self.actions_taken = []
        self.topologies = []
        return super().reset()

    def step(self, action):
        self.actions_taken.append((action, len(self.actions)))
        if self.slot_step_count == 0:
            self.topologies.append((self.adj.copy(), self.rates.copy(), self.background.copy()))
        return super().step(action)


class AddedMethodsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg = replace(
            load_config("small_smoke"),
            gat_hidden_dim=16, actor_hidden_dim=16, critic_hidden_dim=16,
            dqn_warmup_steps=0, dqn_batch_size=4, dqn_target_sync=4, dqn_double=True,
            ddpg_warmup_steps=0, ddpg_batch_size=4,
            ga_population=4, ga_generations=2, aco_ants=4, aco_iterations=2,
        )

    def test_off_policy_training_and_evaluation(self):
        for name, model_type, trainer_type in (
            ("dqn", DQN, DQNTrainer), ("ddpg", DDPG, DDPGTrainer),
        ):
            with self.subTest(method=name):
                torch.manual_seed(42)
                np.random.seed(42)
                env = RecordingEnv(self.cfg)
                model = model_type(self.cfg)
                trainer = trainer_type(model, self.cfg)
                for episode in range(2):
                    updates, metrics = trainer.train_episode(env, episode, 2)
                    self.assertGreater(updates["updates"], 0)
                    self.assertIn("episode_reward", metrics)
                    self.assertTrue(all(0 <= action < count for action, count in env.actions_taken))
                evaluation = evaluate_policy(model, self.cfg, [2042], name)
                self.assertIn("task_success_rate_mean", evaluation)

    def test_planners_use_valid_actions_and_share_the_topology_trace(self):
        reference = RecordingEnv(self.cfg)
        evaluate_heuristic(reference, "greedy_estimate")
        for method in ("ga", "aco"):
            with self.subTest(method=method):
                env = RecordingEnv(self.cfg)
                metrics = evaluate_heuristic(env, method)
                self.assertEqual(env.tasks, reference.tasks)
                self.assertEqual(metrics["tasks"], len(env.actions_taken))
                self.assertIn("objective_score", metrics)
                self.assertTrue(all(0 <= action < count for action, count in env.actions_taken))
                self.assertEqual(len(env.topologies), len(reference.topologies))
                for actual, expected in zip(env.topologies, reference.topologies):
                    for actual_array, expected_array in zip(actual, expected):
                        np.testing.assert_array_equal(actual_array, expected_array)

    def test_slot_model_uses_independent_noise_and_stops_at_slot_end(self):
        env = EdgeEnv(self.cfg)
        end = slot_end(env)
        model = slot_model(env, end, 123)
        self.assertIsNot(model.execution_rng, env.execution_rng)
        self.assertNotEqual(model.execution_rng.bit_generator.state,
                            env.execution_rng.bit_generator.state)
        env.execution_rng.random()
        repeated = slot_model(env, end, 123)
        self.assertEqual(model.execution_rng.bit_generator.state,
                         repeated.execution_rng.bit_generator.state)
        for _ in range(end - env.idx):
            _, _, done, _ = model.step(len(model.actions) - 1)
        self.assertTrue(done)
        self.assertEqual(model.slot, env.slot)
        self.assertEqual(model.tasks, env.tasks[:end])
        np.testing.assert_array_equal(model.adj, env.adj)


if __name__ == "__main__":
    unittest.main()
