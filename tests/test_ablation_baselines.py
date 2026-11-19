import unittest

import numpy as np

from src.baselines import greedy_estimate_action, shortest_fastest_action
from src.config import load_config
from src.environment import Action, shuffle_graph_observation


class AblationBaselineTest(unittest.TestCase):
    def test_shuffle_changes_graph_only(self):
        obs = {
            "nodes": np.arange(12).reshape(4, 3),
            "adjacency": np.arange(16).reshape(4, 4),
            "edges": np.arange(64).reshape(4, 4, 4),
            "candidate_targets": np.array([0, 2]),
        }
        permutation = np.array([2, 0, 1, 3])
        shuffled = shuffle_graph_observation(obs, permutation)
        np.testing.assert_array_equal(shuffled["nodes"], obs["nodes"])
        np.testing.assert_array_equal(
            shuffled["adjacency"], obs["adjacency"][permutation][:, permutation])
        np.testing.assert_array_equal(
            shuffled["edges"], obs["edges"][permutation][:, permutation])
        np.testing.assert_array_equal(
            shuffled["candidate_targets"], obs["candidate_targets"])

    def test_shortest_path_then_fastest_resource(self):
        actions = [
            Action(2, 0, (0, 1, 2), .1, 1., 100.),
            Action(0, 0, (0,), .2, 2., 80.),
            Action(0, 1, (0,), .3, 3., 120.),
            Action(-2, -1, (), 0., 0., 0.),
        ]
        self.assertEqual(shortest_fastest_action(actions), 2)
        self.assertEqual(shortest_fastest_action(actions[-1:]), 0)

    def test_greedy_estimate_uses_candidate_estimates(self):
        actions = [
            Action(0, 0, (0,), .2, 3., 100.),
            Action(1, 0, (0, 1), .4, .3, 100.),
            Action(-2, -1, (), 0., 0., 0.),
        ]
        self.assertEqual(greedy_estimate_action(actions, load_config("default")), 1)
        self.assertEqual(
            greedy_estimate_action(actions[-1:], load_config("default")), 0,
        )


if __name__ == "__main__":
    unittest.main()
