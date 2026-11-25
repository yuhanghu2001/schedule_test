from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.config import load_config
from src.environment import EdgeEnv
from src.evaluation import DAG_EVALUATION_METRICS, evaluate_heuristic, evaluate_policy, summarize_metrics
from src.plotting import load_jsonl
from src.trainer import load_checkpoint, make_model, train_model


class DAGPipelineTest(unittest.TestCase):
    def test_training_checkpoint_inference_and_resume(self):
        cfg = replace(load_config("hetero_dag_smoke"), gat_hidden_dim=16,
                      actor_hidden_dim=16, critic_hidden_dim=16, ppo_epochs=1,
                      evaluation_seed_count=1)
        legacy_cfg = load_config("small_smoke")
        model_type = "dual_gatppo"
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            training = directory / "training"
            train_model(cfg, model_type, training, episodes=2, threads=1)
            checkpoint_path = training / "last_dual_gatppo.pt"
            checkpoint, saved_cfg, saved_model_type = load_checkpoint(checkpoint_path)
            self.assertEqual(saved_cfg, cfg)
            self.assertEqual(saved_model_type, model_type)
            self.assertEqual(checkpoint["config"], asdict(cfg))
            self.assertEqual(checkpoint["observation_version"], cfg.observation_version)
            self.assertEqual(checkpoint["reward_version"], cfg.reward_version)
            self.assertTrue(checkpoint["enable_dag_mode"])
            self.assertEqual(checkpoint["episode"], 2)
            self.assertTrue((training / "best_dual_gatppo.pt").is_file())
            for filename in ("training_metrics.png", "evaluation_metrics.png"):
                self.assertTrue((training / filename).is_file())
            records = load_jsonl(training / "train_metrics.jsonl")
            self.assertEqual([record["episode"] for record in records], [1, 2])
            self.assertTrue(set(DAG_EVALUATION_METRICS) <= records[-1].keys())
            evaluation, = load_jsonl(training / "evaluation_metrics.jsonl")
            model = make_model(saved_cfg, saved_model_type)
            model.load_state_dict(checkpoint["model"])
            self.assertEqual({key: value for key, value in evaluation.items() if key != "episode"},
                             evaluate_policy(model, saved_cfg, [cfg.evaluation_seed], saved_model_type))
            self.assertTrue({f"{field}_mean" for field in DAG_EVALUATION_METRICS} <= evaluation.keys())

            resumed = directory / "resumed"
            train_model(None, "gatppo", resumed, episodes=3, resume=checkpoint_path, threads=1)
            resumed_checkpoint, _, _ = load_checkpoint(resumed / checkpoint_path.name)
            self.assertEqual(resumed_checkpoint["episode"], 3)
            record, = load_jsonl(resumed / "train_metrics.jsonl")
            self.assertEqual(record["episode"], 3)
            resume_config = json.loads((resumed / "train_config.json").read_text(encoding="utf-8"))
            self.assertEqual(resume_config["resume"], str(checkpoint_path))

            with self.assertRaisesRegex(ValueError, r"observation_version.*v5.*v4"):
                load_checkpoint(checkpoint_path, observation_version=legacy_cfg.observation_version)
            with self.assertRaisesRegex(ValueError, r"observation_version.*v5.*v4"):
                train_model(legacy_cfg, "gatppo", directory / "incompatible", episodes=3,
                            resume=checkpoint_path, threads=1)

    def test_heuristics_and_undefined_metrics(self):
        cfg = load_config("hetero_dag_smoke")
        for method in ("heft_hetero", "greedy"):
            with self.subTest(method=method):
                metrics = evaluate_heuristic(EdgeEnv(cfg), method)
                self.assertTrue(set(DAG_EVALUATION_METRICS) <= metrics.keys())
                self.assertGreater(metrics["dag_completion_rate"], 0)
        field = "avg_dag_makespan_s"
        missing = [{field: None}, {field: float("nan")}]
        self.assertEqual(summarize_metrics(missing)[field], {"mean": None, "std": None, "count": 0})
        self.assertEqual(summarize_metrics([*missing, {field: 1.5}])[field],
                         {"mean": 1.5, "std": 0.0, "count": 1})
