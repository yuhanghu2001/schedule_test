import unittest

import numpy as np

from src.tasks import sample_tasks


class TaskSamplingTest(unittest.TestCase):
    def test_deadline_scales_with_reference_processing_time(self):
        radio_rates = np.linspace(60.0, 180.0, 24)
        min_edge_compute = 120.0
        slack_min = 1.8
        slack_max = 3.0
        tasks = sample_tasks(
            np.random.default_rng(42),
            episode_slots=20,
            arrival_rate=5.0,
            burst_prob=0.15,
            burst_multiplier=2.5,
            hotspot_fraction=0.3,
            hotspot_share=0.6,
            topology_period_slots=5,
            terminals=24,
            edges=12,
            radio_rates=radio_rates,
            min_edge_compute_gflops=min_edge_compute,
            deadline_slack_min=slack_min,
            deadline_slack_max=slack_max,
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
        self.assertGreater(max(task.deadline_s for task in tasks), 1.0)
        self.assertTrue(all(task.allow_offload and task.must_offload for task in tasks))
        self.assertTrue(all(task.access_edge == task.source % 12 for task in tasks))

    def test_every_episode_slot_has_an_arrival(self):
        tasks = sample_tasks(
            np.random.default_rng(7), 4, 0.01, 0.0, 2.5, 0.3, 0.6, 2,
            24, 12, np.full(24, 100.0), 120.0, 1.8, 3.0,
        )
        self.assertEqual({task.slot for task in tasks}, set(range(4)))
        self.assertEqual(len(tasks), 4)


if __name__ == "__main__":
    unittest.main()
