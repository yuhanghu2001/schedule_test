import unittest

import numpy as np

from src.tasks import sample_tasks


class TaskSamplingTest(unittest.TestCase):
    def test_deadline_scales_with_reference_processing_time(self):
        radio_rates = np.linspace(60.0, 180.0, 20)
        min_edge_compute = 150.0
        slack_min = 1.8
        slack_max = 3.0
        tasks = sample_tasks(
            np.random.default_rng(42),
            100,
            20,
            6,
            radio_rates,
            min_edge_compute,
            slack_min,
            slack_max,
        )

        for task in tasks:
            radio_rate = radio_rates[task.source]
            reference_time = (
                task.data_mb * 8 / radio_rate
                + task.output_mb * 8 / radio_rate
                + task.compute_gflop / min_edge_compute
            )
            slack = task.deadline_s / reference_time
            self.assertGreaterEqual(slack, slack_min)
            self.assertLessEqual(slack, slack_max)
        self.assertGreater(max(task.deadline_s for task in tasks), 0.95)


if __name__ == "__main__":
    unittest.main()
