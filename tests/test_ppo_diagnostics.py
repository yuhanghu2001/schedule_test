import unittest
from dataclasses import replace

import numpy as np
import torch

from src.algorithms.ppo import PPOTrainer
from src.config import load_config
from src.environment import EdgeEnv
from src.models.policy import GATPPO
from src.tasks import Task


class PPODiagnosticsTest(unittest.TestCase):
    def test_update_returns_finite_diagnostics(self):
        cfg = replace(
            load_config("small_smoke"),
            n_tasks=8, tasks_per_slot=4, ppo_epochs=2, ppo_batch_size=4,
        )
        torch.manual_seed(cfg.seed)
        model = GATPPO()
        trainer = PPOTrainer(model, cfg)
        episode, _ = trainer.collect_episode(EdgeEnv(cfg))
        for obs, action, old_logprob, _, _, _ in episode:
            distribution, _ = model(obs)
            self.assertAlmostEqual(
                distribution.log_prob(torch.tensor(action)).item(), old_logprob, places=6,
            )
        before = [parameter.detach().clone() for parameter in model.parameters()]
        diagnostics = trainer.update(episode)
        expected = {
            "loss", "policy_loss", "value_loss", "entropy", "normalized_entropy",
            "approx_kl", "clip_fraction", "explained_variance", "grad_norm_preclip",
            "value_mean", "return_mean",
        }
        self.assertEqual(set(diagnostics), expected)
        self.assertTrue(all(np.isfinite(value) for value in diagnostics.values()))
        self.assertTrue(any(
            not torch.equal(previous, current)
            for previous, current in zip(before, model.parameters())
        ))

    def test_only_reject_action_has_zero_entropy(self):
        cfg = replace(load_config("small_smoke"), n_tasks=1)
        env = EdgeEnv(cfg)
        env.tasks = [Task(0, 0, 0, .1, .1, .1, .5, 100., .1, 1, False, False)]
        env.idx = 0
        obs = env.observe()
        distribution, _ = GATPPO()(obs)
        self.assertEqual(len(env.actions), 1)
        self.assertEqual(distribution.entropy().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
