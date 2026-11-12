import unittest
from dataclasses import replace

import numpy as np

from src.config import load_config
from src.environment import EdgeEnv
from src.models.policy import GATPPO


class PolicyObservationTest(unittest.TestCase):
    def test_global_features_change_with_history(self):
        cfg = replace(load_config("configs/small_smoke.json"), n_tasks=2)
        env = EdgeEnv(cfg)
        initial = env.reset()["global_features"].copy()
        env.accepted = 1
        env.sum_delay = .2
        env.total_energy = 1.0
        changed = env.observe()["global_features"]
        self.assertFalse(np.array_equal(initial, changed))

    def test_actor_and_critic_accept_global_features(self):
        env = EdgeEnv(load_config("configs/small_smoke.json"))
        obs = env.reset()
        self.assertEqual(obs["global_features"].shape, (9,))
        for use_gat in (True, False):
            distribution, value = GATPPO(use_gat=use_gat)(obs)
            self.assertEqual(len(distribution.probs), len(env.actions))
            self.assertTrue(np.all(np.isfinite(distribution.probs.detach().numpy())))
            self.assertTrue(np.isfinite(value.item()))

    def test_global_features_are_decision_time_statistics(self):
        env = EdgeEnv(load_config("configs/small_smoke.json"))
        obs = env.reset()
        self.assertEqual(obs["global_features"][0], 0.0)
        self.assertEqual(obs["global_features"][1], 0.0)
        self.assertEqual(obs["global_features"][3], 0.0)
        self.assertEqual(obs["global_features"][4], 0.0)


if __name__ == "__main__":
    unittest.main()
