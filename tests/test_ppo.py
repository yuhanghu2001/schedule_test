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
        cfg = load_config("default")
        self.assertEqual(cfg.episode_slots, 20)
        self.assertEqual(cfg.arrival_rate, 18.0)
        self.assertEqual(cfg.gat_hidden_dim, 256)
        self.assertEqual(cfg.actor_hidden_dim, 128)
        self.assertEqual(cfg.critic_hidden_dim, 128)
        self.assertEqual(cfg.actor_learning_rate, 3e-4)
        self.assertEqual(cfg.critic_learning_rate, 1e-3)
        self.assertEqual(cfg.ppo_epochs, 5)
        self.assertEqual(cfg.dinkelbach_max_iterations, 4)
        self.assertEqual(cfg.dinkelbach_tolerance, 1e-3)
        high_congestion = load_config("high_congestion")
        self.assertEqual(high_congestion.congestion_min, 0.45)
        self.assertEqual(high_congestion.congestion_max, 0.8)
        small_smoke = load_config("small_smoke")
        self.assertEqual(small_smoke.episode_slots, 4)
        self.assertEqual(small_smoke.ppo_batch_size, 16)

    def test_linear_schedule(self):
        trainer = PPOTrainer(GATPPO(), load_config("small_smoke"))
        actor, critic = trainer.optimizer.param_groups
        self.assertEqual((actor["lr"], critic["lr"], trainer.ent_coef), (3e-4, 1e-3, 0.01))
        for episode_index, episodes, actor_lr, critic_lr, ent_coef in (
            (0, 3, 3e-4, 1e-3, 0.01),
            (1, 3, 1.65e-4, 5.5e-4, 0.0055),
            (2, 3, 3e-5, 1e-4, 0.001),
            (0, 1, 3e-4, 1e-3, 0.01),
        ):
            trainer.set_schedule(episode_index, episodes)
            self.assertAlmostEqual(actor["lr"], actor_lr, places=12)
            self.assertAlmostEqual(critic["lr"], critic_lr, places=12)
            self.assertAlmostEqual(trainer.ent_coef, ent_coef, places=12)

    def test_minibatch_update_changes_parameters(self):
        cfg = replace(
            load_config("small_smoke"),
            episode_slots=2,
            arrival_rate=4.0,
            ppo_epochs=2,
            ppo_batch_size=4,
        )
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        model = GATPPO()
        trainer = PPOTrainer(model, cfg)
        self.assertEqual(trainer.optimizer.param_groups[0]["name"], "actor")
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 3e-4)
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
