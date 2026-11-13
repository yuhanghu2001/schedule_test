import unittest
from dataclasses import replace

from src.baselines import evaluate_heuristic
from src.config import load_config
from src.environment import EdgeEnv


class MultihopActivationTest(unittest.TestCase):
    def test_feasibility_hops_and_completion_targets(self):
        cfg = load_config("configs/default.json")
        evaluation_cfg = replace(cfg, seed=cfg.evaluation_seed)

        empty_batch_metrics = evaluate_heuristic(EdgeEnv(replace(
            evaluation_cfg, tasks_per_slot=1,
        )))
        full_metrics = evaluate_heuristic(EdgeEnv(evaluation_cfg))

        self.assertGreaterEqual(empty_batch_metrics["feasible_task_rate"], 0.95)
        self.assertGreaterEqual(full_metrics["task_success_rate"], 0.95)
        self.assertGreaterEqual(full_metrics["mean_path_length"], 0.5)
        self.assertLessEqual(full_metrics["mean_path_length"], 1.5)
        self.assertGreater(full_metrics["mean_hops_accepted"], 0.0)
        expected_path_length = (
            full_metrics["mean_hops_accepted"]
            + full_metrics["edge_execution_rate"] / full_metrics["task_success_rate"]
            + 2 * full_metrics["cloud_execution_rate"] / full_metrics["task_success_rate"]
        )
        self.assertAlmostEqual(full_metrics["mean_path_length"], expected_path_length)


if __name__ == "__main__":
    unittest.main()
