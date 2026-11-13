import unittest
from dataclasses import replace

import numpy as np
import torch

from src.algorithms.ppo import PPOTrainer
from src.config import load_config
from src.environment import EdgeEnv
from src.models.policy import GATPPO


class PPOTrainingTest(unittest.TestCase):
    def test_default_parameters_match_experiment_table(self):
        cfg = load_config("configs/default.json")
        self.assertEqual(cfg.n_tasks, 200)
        self.assertEqual(cfg.gat_hidden_dim, 256)
        self.assertEqual(cfg.actor_hidden_dim, 128)
        self.assertEqual(cfg.critic_hidden_dim, 128)
        self.assertEqual(cfg.actor_learning_rate, 1e-4)
        self.assertEqual(cfg.critic_learning_rate, 1e-3)
        self.assertEqual(cfg.ppo_epochs, 5)
        self.assertEqual(cfg.dinkelbach_max_iterations, 4)
        self.assertEqual(cfg.dinkelbach_tolerance, 1e-3)

    def test_minibatch_update_changes_parameters(self):
        cfg = replace(
            load_config("configs/small_smoke.json"),
            n_tasks=12,
            tasks_per_slot=6,
            ppo_epochs=2,
            ppo_batch_size=4,
        )
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        model = GATPPO()
        trainer = PPOTrainer(model, cfg)
        self.assertEqual(trainer.optimizer.param_groups[0]["name"], "actor")
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 1e-4)
        self.assertEqual(trainer.optimizer.param_groups[1]["name"], "critic")
        self.assertEqual(trainer.optimizer.param_groups[1]["lr"], 1e-3)
        episode, _ = trainer.collect_episode(EdgeEnv(cfg))
        for obs, action, old_logprob, _, _, _ in episode:
            distribution, _ = model(obs)
            current = distribution.log_prob(torch.tensor(action)).item()
            self.assertAlmostEqual(current, old_logprob, places=6)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        update_metrics = trainer.update(episode)
        self.assertTrue(np.isfinite(update_metrics["loss"]))
        self.assertTrue(any(
            not torch.equal(previous, current)
            for previous, current in zip(before, model.parameters())
        ))


if __name__ == "__main__":
    unittest.main()
