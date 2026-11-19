import unittest
from dataclasses import replace

import numpy as np

from src.config import load_config
from src.environment import EdgeEnv


class DynamicTopologyTest(unittest.TestCase):
    def test_flaps_preserve_connectivity_and_rebuild_resets_state(self):
        cfg = replace(load_config("default"), link_flap_prob=1.0)
        env = EdgeEnv(cfg)
        n = cfg.n_edges
        disconnected = set()
        for slot in range(1, cfg.topology_period_slots):
            previous = env.rates > 0
            old_background = env.background.copy()
            env._advance_to_slot(slot)
            active = env.rates > 0
            added = active & ~previous
            removed = previous & ~active
            self.assertTrue(added.any() and removed.any())
            np.testing.assert_array_equal(active, active.T)
            self.assertTrue(active[n, :n].all())
            self.assertLessEqual(np.count_nonzero(active[:n, :n]) // 2, cfg.undirected_edge_links)
            # 删除后仍有旧边连接全部边缘，新增链路和云均不参与连通性判断。
            retained = (active & previous)[:n, :n]
            reachable = np.arange(n) == 0
            for _ in range(n - 1):
                reachable |= retained[reachable].any(axis=0)
            self.assertTrue(reachable.all())

            disconnected.update(zip(*np.nonzero(np.triu(removed, 1))))
            disconnected.difference_update(zip(*np.nonzero(np.triu(added, 1))))
            self.assertEqual(set(env.flapped_edges), disconnected)
            for state in (env.base_rates, env.base_background, env.prop, env.forward_power):
                self.assertTrue((state[added] > 0).all())
                self.assertTrue((state[~active] == 0).all())
            np.testing.assert_array_equal(env.background[added], env.base_background[added])
            self.assertTrue((env.background[~active] == 0).all())
            continuous = active & previous
            expected_delta = np.where(continuous, env.background - old_background, 0)
            np.testing.assert_array_equal(env.congestion_delta, expected_delta)
            a, b = np.nonzero(np.triu(active[:n, :n], 1))
            ratio = env.rates[a, b] / env.base_rates[a, b]
            self.assertTrue((np.abs(ratio - 1) <= cfg.link_rate_fluctuation).all())
            self.assertTrue(np.any(ratio != 1))
            np.testing.assert_array_equal(env.rates, env.rates.T)
            np.testing.assert_array_equal(env.rates[n], env.base_rates[n])
        env._advance_to_slot(cfg.topology_period_slots)
        self.assertEqual(np.count_nonzero(env.base_rates[:n, :n]) // 2, cfg.undirected_edge_links)
        self.assertEqual(env.flapped_edges, [])
        self.assertFalse(env.congestion_delta.any())
        np.testing.assert_array_equal(env.background, env.base_background)

    def test_ar_mean_reversion_spike_and_clipping_include_cloud_arcs(self):
        cfg = replace(load_config("small_smoke"), link_flap_prob=0, congestion_step_noise=0,
                      link_spike_prob=1, link_spike_min=.15, link_spike_max=.15)
        env = EdgeEnv(cfg)
        active = env.rates > 0
        env.background[active] = .55
        expected = np.clip(cfg.congestion_ar_coef * .55
                           + (1 - cfg.congestion_ar_coef) * env.base_background[active] + .15,
                           cfg.congestion_min, cfg.congestion_max)
        env._advance_to_slot(1)
        np.testing.assert_allclose(env.background[active], expected, atol=np.finfo(np.float32).eps)
        np.testing.assert_allclose(env.congestion_delta[active], expected - .55, atol=1e-7)

    def test_actions_do_not_change_same_seed_topology_trajectory(self):
        cfg = load_config("small_smoke")
        accept, reject = EdgeEnv(cfg), EdgeEnv(cfg)
        for _ in accept.tasks:
            accept.step(0)
            reject.step(len(reject.actions) - 1)
            np.testing.assert_array_equal(accept.rates, reject.rates)
            np.testing.assert_array_equal(accept.background, reject.background)
        self.assertGreater(accept.accepted, reject.accepted)


if __name__ == "__main__":
    unittest.main()
