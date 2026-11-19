import unittest
from dataclasses import replace

from src.baselines import evaluate_heuristic
from src.config import load_config
from src.environment import EdgeEnv


class MultihopActivationTest(unittest.TestCase):
    def test_feasibility_hops_and_completion_targets(self):
        cfg = load_config("default")
        evaluation_cfg = replace(cfg, seed=cfg.evaluation_seed)

        low_arrival_metrics = evaluate_heuristic(EdgeEnv(replace(
            evaluation_cfg, arrival_rate=1.0,
        )))
        full_metrics = evaluate_heuristic(EdgeEnv(evaluation_cfg))

        self.assertGreaterEqual(low_arrival_metrics["feasible_task_rate"], 0.9)
        self.assertGreater(full_metrics["feasible_task_rate"], 0.8)
        self.assertEqual(full_metrics["acceptance_rate"], full_metrics["feasible_task_rate"])
        self.assertGreaterEqual(full_metrics["mean_path_length"], 1.0)
        self.assertLessEqual(full_metrics["mean_path_length"], cfg.max_hops + 2)
        self.assertEqual(full_metrics["local_execution_rate"], 0.0)
        self.assertGreater(full_metrics["mean_hops_accepted"], 0.0)
        expected_path_length = (
            full_metrics["mean_hops_accepted"]
            + full_metrics["edge_execution_rate"] / full_metrics["acceptance_rate"]
            + 2 * full_metrics["cloud_execution_rate"] / full_metrics["acceptance_rate"]
        )
        self.assertAlmostEqual(full_metrics["mean_path_length"], expected_path_length)


if __name__ == "__main__":
    unittest.main()
