import unittest

import numpy as np

from src.tasks import sample_tasks


class TaskSamplingTest(unittest.TestCase):
    def test_deadline_scales_with_reference_processing_time(self):
        max_uplink_rate = 180.0
        min_edge_compute = 50.0
        slack_min = 1.2
        slack_max = 2.5
        tasks = sample_tasks(
            np.random.default_rng(42),
            100,
            20,
            6,
            max_uplink_rate,
            min_edge_compute,
            slack_min,
            slack_max,
        )

        for task in tasks:
            reference_time = (
                task.data_mb * 8 / max_uplink_rate
                + task.compute_gflop / min_edge_compute
            )
            slack = task.deadline_s / reference_time
            self.assertGreaterEqual(slack, slack_min)
            self.assertLessEqual(slack, slack_max)


if __name__ == "__main__":
    unittest.main()
