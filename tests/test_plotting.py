import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.plotting import save_run_plots


class InterruptedPlottingTest(unittest.TestCase):
    def test_training_plot_is_saved_before_first_evaluation(self):
        record = {
            "episode": 1,
            "episode_reward": 1.0,
            "task_success_rate": 0.8,
            "avg_delay_s": 0.5,
            "avg_energy_j": 1.2,
            "load_balance_degree": 0.9,
        }
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "train_metrics.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8",
            )
            (run_dir / "evaluation_metrics.jsonl").write_text("", encoding="utf-8")

            save_run_plots(run_dir)

            self.assertTrue((run_dir / "training_metrics.png").is_file())
            self.assertFalse((run_dir / "evaluation_metrics.png").exists())


if __name__ == "__main__":
    unittest.main()
