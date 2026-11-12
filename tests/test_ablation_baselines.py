import unittest

import numpy as np

from src.ablation import shuffle_graph_observation
from src.baselines import shortest_fastest_action
from src.environment import Action


class AblationBaselineTest(unittest.TestCase):
    def test_shuffle_changes_graph_only(self):
        obs = {
            "nodes": np.arange(12).reshape(4, 3),
            "adjacency": np.arange(16).reshape(4, 4),
            "edges": np.arange(32).reshape(4, 4, 2),
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


if __name__ == "__main__":
    unittest.main()
