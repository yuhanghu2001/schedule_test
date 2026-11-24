from contextlib import ExitStack
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient
import torch

from backend.app import app
from backend.services.sim_service import SimManager
from backend.services.train_service import TrainManager
from src.config import Config, load_config
from src.models.policy import GATPPO


PRESET = "hetero_dag_smoke"
CONFIG = {
    "n_edges": 3,
    "n_terminals": 3,
    "episode_slots": 2,
    "arrival_rate": 0,
    "eval_interval": 1,
    "evaluation_seed_count": 1,
    "gat_hidden_dim": 8,
    "actor_hidden_dim": 8,
    "critic_hidden_dim": 8,
    "ppo_epochs": 1,
}
ENV = {"preset_name": PRESET, "config": CONFIG}


class BackendDAGTests(TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.temp_dir = Path(stack.enter_context(TemporaryDirectory()))
        self.manager = TrainManager(results_dir=self.temp_dir)
        self.sim_manager = SimManager()
        stack.enter_context(patch.object(app.state, "train_manager", self.manager))
        stack.enter_context(patch.object(app.state, "sim_manager", self.sim_manager))
        self.client = stack.enter_context(TestClient(app, backend_options={"use_uvloop": True}))

    def test_presets_opt_in(self):
        legacy = {"default", "high_congestion", "small_smoke"}
        dag = {"hetero_dag_default", "hetero_dag_high_congestion", "hetero_dag_smoke"}
        for query, expected in (("", legacy), ("?include_dag=true", legacy | dag)):
            with self.subTest(query=query):
                response = self.client.get(f"/api/env/presets{query}")
                self.assertEqual(response.status_code, 200)
                presets = response.json()["presets"]
                self.assertEqual(set(presets), expected)
                self.assertEqual({name for name, cfg in presets.items() if cfg["enable_dag_mode"]},
                                 expected & dag)

    def test_heft_simulation_through_episode_end(self):
        response = self.client.post("/api/sim/sessions", json={**ENV, "policy": "heft_hetero"})
        self.assertEqual(response.status_code, 201, response.text)
        session_id = response.json()["session_id"]
        initial = response.json()["snapshot"]
        url = f"/api/sim/sessions/{session_id}"
        self.assertTrue(initial["enable_dag_mode"])
        self.assertEqual(initial["observation_version"], "v5_dual_graph_dag")
        self.assertEqual(initial["reward_version"], "v9_hetero_dag")
        self.assertEqual(initial["nodes"][-1]["tier"], "cloud")
        for node in initial["nodes"]:
            self.assertIn(node["tier"], {"gateway", "middle", "access", "cloud"})
            self.assertEqual(len(node["speed"]), 3)
            self.assertEqual(len(node["resource_used"]), 3)
            for resource in ("cpu_cores", "ram_gb", "npu_vram_gb", "fpga_lut",
                             "fpga_ff", "fpga_bram", "fpga_dsp"):
                self.assertEqual(node[f"{resource}_cap"], node[f"{resource}_free"])
            self.assertEqual(node["npu_resident_op"], -1)
            self.assertEqual(node["fpga_resident_op"], -1)

        task = initial["current_task"]
        subtasks = {row["subtask_id"]: row for row in task["subtasks"]}
        self.assertEqual(subtasks[task["subtask_id"]]["status"], "ready")
        self.assertEqual(subtasks[task["subtask_id"]]["op_type"], task["op_type"])
        for edge in task["edges"]:
            self.assertIn(edge["src"], subtasks[edge["dst"]]["preds"])
            self.assertGreater(edge["data_mb"], 0)
        self.assertTrue(task["edges"])
        valid = [action["valid"] for action in task["candidates"]]
        self.assertTrue(valid[-1])
        self.assertIn(False, valid)
        self.assertEqual(task["feasible_action_count"], sum(valid[:-1]))
        self.assertEqual([action["action_id"] for action in task["candidates"]],
                         list(range(len(valid))))

        snapshot = self.client.post(f"{url}/step").json()
        self.assertEqual(snapshot["task_index"], 1)
        trace = snapshot["recent_tasks"][0]
        self.assertEqual((trace["dag_id"], trace["subtask_id"]),
                         (task["dag_id"], task["subtask_id"]))
        self.assertIn(trace["status"], {"running", "completed"})
        self.assertGreater(trace["finish_time_s"], trace["start_time_s"])
        for field in ("setup_delay_s", "setup_energy_j", "comm_tier", "dag_status"):
            self.assertIn(field, trace)
        self.assertTrue(trace["flows"])
        self.assertEqual(set(trace["flows"][0]), {"pred_id", "path", "comm_tier", "data_mb"})

        for _ in range(CONFIG["episode_slots"] + 1):
            old_slot, old_index = snapshot["slot"], snapshot["task_index"]
            snapshot = self.client.post(f"{url}/step-slot").json()
            self.assertGreater(snapshot["task_index"], old_index)
            if snapshot["done"]:
                break
            self.assertGreater(snapshot["slot"], old_slot)
        self.assertTrue(snapshot["done"])
        self.assertIsNone(snapshot["current_task"])
        self.assertEqual(snapshot["task_index"], len(snapshot["recent_tasks"]))
        self.assertEqual(snapshot["metrics"]["tasks"], len(snapshot["dags"]))
        self.assertIn("dag_completion_rate", snapshot["metrics"])
        self.assertTrue(all(row["status"] in {"completed", "failed"} for row in snapshot["dags"]))
        json.dumps(self.sim_manager.get_snapshot(session_id), allow_nan=False)
        self.assertEqual(self.client.post(f"{url}/step").json(), snapshot)
        self.assertEqual(self.client.post(f"{url}/reset").json(), initial)

    def test_dag_training_resume_and_evaluation(self):
        def train(body):
            response = self.client.post("/api/training/runs", json={**body, "threads": 1})
            self.assertEqual(response.status_code, 202, response.text)
            with self.client.websocket_connect(
                f"/ws/training/{response.json()['run_id']}"
            ) as socket:
                event = socket.receive_json()
                while event["type"] != "terminal":
                    event = socket.receive_json()
            self.assertEqual(event["status"], "completed", event)
            return event["run"]

        run = train({**ENV, "episodes": 1, "model_type": "ppo"})
        path = run["last_checkpoint"]
        checkpoints = self.client.get("/api/training/checkpoints").json()
        self.assertEqual(len(checkpoints), 2)
        for checkpoint in checkpoints:
            self.assertEqual(checkpoint["model_type"], "ppo")
            self.assertTrue(checkpoint["enable_dag_mode"])
            self.assertEqual(checkpoint["observation_version"], "v5_dual_graph_dag")
            self.assertEqual(checkpoint["reward_version"], "v9_hetero_dag")

        resumed = train({"episodes": 2, "resume_path": path})
        cfg_snapshot = resumed["config_json"]
        for key in ("observation_version", "reward_version", "enable_dag_mode"):
            self.assertEqual(cfg_snapshot[key], checkpoints[0][key])
        self.assertEqual(resumed["current_episode"], 2)
        self.assertEqual(resumed["model_type"], "ppo")
        response = self.client.post("/api/experiments/evaluate", json={
            **ENV, "seeds": [11, 12],
            "strategies": [{"method": "heft_hetero"}, {"method": "ppo", "checkpoint_path": path}],
        })
        self.assertEqual(response.status_code, 200, response.text)
        for strategy in response.json()["strategies"]:
            self.assertEqual([record["seed"] for record in strategy["runs"]], [11, 12])
            self.assertIn("dag_completion_rate_mean", strategy["summary"])
            self.assertIn("mean_load_variance_mean", strategy["summary"])
        with self.assertRaisesRegex(ValueError, "observation_version"):
            self.sim_manager.create_session(replace(Config(), **CONFIG), "checkpoint", path)

    def test_named_graph_checkpoint_policies(self):
        cfg = replace(load_config(PRESET), **CONFIG)
        for model_type in ("dual_gatppo", "single_gatppo", "shuffled_gat"):
            with self.subTest(model_type=model_type):
                path = self.temp_dir / f"{model_type}.pt"
                torch.save({"config": asdict(cfg), "model_type": model_type,
                            "model": GATPPO.from_config(cfg, model_type).state_dict()}, path)
                response = self.client.post("/api/sim/sessions", json={
                    **ENV, "policy": model_type, "checkpoint_path": str(path),
                })
                self.assertEqual(response.status_code, 201, response.text)
                url = f"/api/sim/sessions/{response.json()['session_id']}"
                step = self.client.post(f"{url}/step")
                self.assertEqual(step.status_code, 200, step.text)
                self.assertEqual(step.json()["task_index"], 1)
