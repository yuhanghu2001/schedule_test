import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from src.algorithms.ppo import PPOTrainer
from src.baselines import baseline_action
from src.config import load_config
from src.environment import DAGAction, EdgeEnv, model_environment
from src.models.gat import DAGGraphEncoder
from src.models.policy import GATPPO


VARIANTS = ("dual_gatppo", "single_gatppo", "ppo", "shuffled_gat")


class DAGPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg = replace(load_config("hetero_dag_smoke"), gat_hidden_dim=16,
                          actor_hidden_dim=16, critic_hidden_dim=16, ppo_epochs=1)

    def assert_finite_gradients(self, model):
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_collect_and_update(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(self.cfg.seed)
                model = GATPPO.from_config(self.cfg, variant)
                self.assertEqual(model.use_gat, variant != "ppo")
                self.assertEqual(model.dag_encoder is not None, variant in ("dual_gatppo", "shuffled_gat"))
                trainer = PPOTrainer(model, self.cfg)
                env = model_environment(EdgeEnv(self.cfg), variant, self.cfg.seed)
                episode, _ = trainer.collect_episode(env)
                self.assertTrue(episode[-1][-1])
                for obs, action, old_logprob, _, _, _ in episode:
                    dist, _ = model(obs)
                    mask = torch.from_numpy(obs["action_mask"])
                    self.assertTrue(mask[action])
                    self.assertTrue(torch.isneginf(dist.logits[~mask]).all())
                    self.assertTrue((dist.probs[~mask] == 0).all())
                    self.assertAlmostEqual(dist.log_prob(torch.tensor(action)).item(), old_logprob)
                before = model.actor[0].weight.detach().clone()
                metrics = trainer.update(episode)
                self.assertTrue(all(np.isfinite(x) for x in metrics.values()))
                self.assertFalse(torch.equal(before, model.actor[0].weight))
                self.assert_finite_gradients(model)

    def test_forced_reject_and_entropy_denominator(self):
        obs = EdgeEnv(self.cfg).observe()
        reject = len(obs["action_mask"]) - 1
        mask = np.zeros(reject + 1, dtype=bool)
        mask[-1] = True
        model = GATPPO.from_config(self.cfg, "dual_gatppo")
        trainer = PPOTrainer(model, self.cfg)
        # 两个等概率合法动作的归一化熵为 1，强制拒绝步为 0。
        torch.nn.init.zeros_(model.actor[-1].weight)
        episode = []
        for valid_count in (1, 2):
            obs = dict(obs, action_mask=mask.copy())
            action, logprob, value = model.act(obs)
            dist, critic_value = model(obs)
            if valid_count == 1:
                self.assertEqual((action, logprob, dist.entropy().item()), (reject, 0., 0.))
                loss = critic_value.square() - dist.log_prob(torch.tensor(action)) - dist.entropy()
                loss.backward()
                self.assert_finite_gradients(model)
            episode.append((obs, action, logprob, value, 0., valid_count == 2))
            mask[0] = True
        self.assertAlmostEqual(trainer.update(episode)["normalized_entropy"], .5, places=6)

    def test_padding_does_not_enter_attention_or_mean(self):
        encoder = DAGGraphEncoder(3, hidden=8)
        nodes = torch.randn(4, 3)
        adj = torch.ones(4, 4)
        edges = torch.randn(4, 4, 1)
        # 填充行即便含特征和边也不得参与；真实节点索引不要求连续。
        mask = torch.tensor([True, False, True, True])
        current, pooled = encoder(nodes, adj, edges, mask, 2)
        real_nodes = encoder.layer(nodes[mask], adj[mask][:, mask], edges[mask][:, mask])
        torch.testing.assert_close(current, real_nodes[1])
        torch.testing.assert_close(pooled, real_nodes.mean(0))

    def test_shuffled_observation_only_permutes_network(self):
        dual = GATPPO.from_config(self.cfg, "dual_gatppo")
        shuffled = GATPPO.from_config(self.cfg, "shuffled_gat")
        shuffled.load_state_dict(dual.state_dict(), strict=True)
        env = EdgeEnv(self.cfg)
        wrapped = shuffled.rollout_environment(env, self.cfg.seed)
        obs = wrapped.reset()
        original = env.observe()
        perm = wrapped.permutation
        self.assertFalse(np.array_equal(perm, np.arange(len(perm))))
        np.testing.assert_array_equal(obs["nodes"], original["nodes"][perm])
        for key in ("adj", "adjacency", "gat_adj", "edges"):
            np.testing.assert_array_equal(obs[key], original[key][perm][:, perm])
        for key in original.keys() - {"nodes", "adj", "adjacency", "gat_adj", "edges"}:
            np.testing.assert_array_equal(obs[key], original[key])

    def test_masked_baseline_episodes(self):
        for name in ("heft_hetero", "greedy", "shortest_fastest", "random", "cloud_first"):
            with self.subTest(name=name):
                env = EdgeEnv(self.cfg)
                done = False
                while not done:
                    action = baseline_action(name, env)
                    self.assertTrue(env.action_mask[action])
                    self.assertEqual(action == len(env.actions) - 1, not env.action_mask[:-1].any())
                    _, _, done, _ = env.step(action)

    def test_heft_uses_delivery_then_energy_then_hops(self):
        def candidate(finish, delivery, energy, path):
            return DAGAction(path[-1], 0, path, delivery, energy, 1.,
                             finish_time_s=finish, delivery_time_s=delivery)

        actions = [candidate(0., 0., 0., (0,)), candidate(1., 5., 1., (0,)),
                   candidate(2., 4., 3., (0,)), candidate(2., 4., 2., (0, 1)),
                   candidate(2., 4., 2., (0,)), DAGAction(-2, -1, (), 0., 0., 0.)]
        env = SimpleNamespace(actions=actions, cloud=3,
                              action_mask=np.array([False, True, True, True, True, True]))
        self.assertEqual(baseline_action("heft_hetero", env), 4)
        env.action_mask[:-1] = False
        self.assertEqual(baseline_action("heft_hetero", env), 5)

    def test_v4_checkpoint_structure_and_aliases(self):
        cfg = load_config("small_smoke")
        obs = EdgeEnv(cfg).observe()
        keys = {f"gat.layer{i}.{param}" for i in (1, 2) for param in
                ("transform.weight", "attention.weight", "norm.weight", "norm.bias")}
        keys.update(f"{layer}.{param}"
                    for layer in ("gat.skip", "flat_encoder.0", "actor.0", "actor.2", "critic.0", "critic.2")
                    for param in ("weight", "bias"))
        for name, canonical in (("gatppo", "gatppo"), ("shuffled_gat", "shuffled_gat"), ("ppo", "ppo"),
                                ("gat", "gatppo"), ("gatpppo", "gatppo"), ("flat", "ppo")):
            with self.subTest(name=name):
                legacy = GATPPO(use_gat=canonical != "ppo")
                model = GATPPO.from_config(cfg, name)
                self.assertEqual(model.model_type, canonical)
                self.assertEqual(set(model.state_dict()), keys)
                self.assertEqual(model.gat.layer1.transform.in_features, 10)
                self.assertEqual(model.actor[0].in_features, 3 * cfg.gat_hidden_dim + 27)
                self.assertEqual(model.critic[0].in_features, 2 * cfg.gat_hidden_dim + 17)
                model.load_state_dict(legacy.state_dict(), strict=True)
                expected, expected_value = legacy(obs)
                actual, value = model(obs)
                torch.testing.assert_close(actual.logits, expected.logits)
                torch.testing.assert_close(value, expected_value)
