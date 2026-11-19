import unittest
from dataclasses import replace

import numpy as np

from src.config import load_config
from src.environment import EdgeEnv
from src.models.policy import GATPPO


class PolicyObservationTest(unittest.TestCase):
    def test_global_features_change_with_history(self):
        cfg = replace(load_config("small_smoke"), episode_slots=1)
        env = EdgeEnv(cfg)
        initial = env.reset()["global_features"].copy()
        env.accepted = 1
        env.sum_delay = .2
        env.total_energy = 1.0
        changed = env.observe()["global_features"]
        self.assertFalse(np.array_equal(initial, changed))

    def test_edge_features_use_current_bandwidth_reservations_and_signed_trend(self):
        cfg = replace(load_config("small_smoke"), link_rate_max_mbps=1600.0)
        env = EdgeEnv(cfg)
        while env.slot == 0:
            env.step(len(env.actions) - 1)
        a, b = np.argwhere(np.triu(env.rates[:env.cloud, :env.cloud], k=1) > 0)[0]
        env.slot_link_used[a, b] = env._slot_capacity_mb(a, b) / 4
        obs = env.observe()
        edges = obs["edges"]
        np.testing.assert_allclose(edges[:, :, 0], env.rates / cfg.link_rate_max_mbps)
        self.assertAlmostEqual(edges[a, b, 1], env.background[a, b] + .25)
        np.testing.assert_allclose(edges[:, :, 2], env.prop / .003)
        np.testing.assert_array_equal(edges[:, :, 3], env.congestion_delta)
        self.assertLess(edges[:, :, 3].min(), 0)
        self.assertGreater(edges[:, :, 3].max(), 0)
        self.assertTrue(np.all(obs["adjacency"][env.cloud, :env.cloud] == 0))
        self.assertEqual(obs["adjacency"][env.cloud, env.cloud], 1)

    def test_actor_and_critic_accept_global_features(self):
        env = EdgeEnv(load_config("small_smoke"))
        obs = env.reset()
        self.assertEqual(obs["global_features"].shape, (9,))
        self.assertEqual(obs["edges"].shape, (13, 13, 4))
        for use_gat in (True, False):
            model = GATPPO(use_gat=use_gat)
            self.assertEqual(model.gat.layer1.transform.out_features, 256)
            self.assertEqual(model.gat.layer2.transform.out_features, 256)
            self.assertEqual(model.actor[0].out_features, 128)
            self.assertEqual(model.critic[0].out_features, 128)
            distribution, value = model(obs)
            self.assertEqual(len(distribution.probs), len(env.actions))
            self.assertTrue(np.all(np.isfinite(distribution.probs.detach().numpy())))
            self.assertTrue(np.isfinite(value.item()))

    def test_global_features_are_decision_time_statistics(self):
        env = EdgeEnv(load_config("small_smoke"))
        obs = env.reset()
        self.assertEqual(obs["global_features"][0], 0.0)
        self.assertEqual(obs["global_features"][1], 0.0)
        self.assertEqual(obs["global_features"][3], 0.0)
        self.assertEqual(obs["global_features"][4], 0.0)


if __name__ == "__main__":
    unittest.main()
