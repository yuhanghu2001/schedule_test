from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.dialects.mysql import dialect
from sqlalchemy.schema import CreateTable
import torch

from backend.app import app
from backend.config import REPO_ROOT
from backend.db import Base, engine
from backend.services.sim_service import SimManager, config_from_dict
from backend.services.train_service import TrainManager


CONFIG = {
    "n_edges": 2,
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


class BackendTests(TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.temp_dir = Path(stack.enter_context(TemporaryDirectory()))
        self.manager = TrainManager(results_dir=self.temp_dir)
        stack.enter_context(patch.object(app.state, "train_manager", self.manager))
        stack.enter_context(patch.object(app.state, "sim_manager", SimManager()))
        self.client = stack.enter_context(TestClient(app, backend_options={"use_uvloop": True}))

    def test_health_schema_and_presets(self):
        health = self.client.get("/api/health").json()
        self.assertEqual(health["status"], "ok")
        self.assertIsInstance(health["db_connected"], bool)
        self.assertEqual(health["results_dir"], str(self.temp_dir))
        self.assertEqual(engine.dialect.name, "mysql")
        origin = "http://localhost:8888"
        cors = self.client.options("/api/env/preview", headers={
            "Origin": origin, "Access-Control-Request-Method": "POST",
        })
        self.assertEqual(cors.headers["access-control-allow-origin"], origin)

        expected = ";\n".join(
            str(CreateTable(table, if_not_exists=True).compile(dialect=dialect())).strip()
            for table in Base.metadata.sorted_tables
        ) + ";"
        schema = (REPO_ROOT / "backend/schema.sql").read_text()
        self.assertEqual(schema.split(), expected.split())
        self.assertEqual(len(Base.metadata.tables), 5)

        response = self.client.get("/api/env/presets")
        self.assertEqual(response.status_code, 200)
        presets = response.json()
        self.assertEqual(set(presets["presets"]), {"default", "high_congestion", "small_smoke"})
        self.assertEqual(presets["resource_names"], ["CPU", "NPU", "FPGA"])
        self.assertEqual(len(presets["task_kind_names"]), 5)
        preset = config_from_dict(presets["presets"]["default"])
        self.assertIsInstance(preset.service_deployment[0], tuple)
        self.assertIsInstance(preset.edge_speed_matrix[0], tuple)

    def test_preview_simulation_and_link_perturbation(self):
        response = self.client.post("/api/env/preview", json=CONFIG)
        self.assertEqual(response.status_code, 200)
        preview = response.json()
        self.assertEqual([node["kind"] for node in preview["nodes"]], ["edge", "edge", "cloud"])
        self.assertEqual([t["access_edge"] for t in preview["terminals"]], [0, 0, 1])
        self.assertEqual(preview["metrics"]["tasks"], 0)
        self.assertEqual(preview["current_task"]["candidates"][-1]["destination"], -2)

        response = self.client.post("/api/sim/sessions", json={"config": CONFIG, "policy": "random"})
        self.assertEqual(response.status_code, 201)
        session = response.json()
        self.assertEqual(session["snapshot"], preview)
        url = f"/api/sim/sessions/{session['session_id']}"
        self.assertEqual(self.client.get(url).json(), preview)

        def edge_link(snapshot):
            return next(link for link in snapshot["links"] if link["kind"] == "edge")

        patch_link = {"source": 0, "target": 1}
        response = self.client.post(f"{url}/perturb", json={**patch_link, "congestion": 0.5})
        congested = edge_link(response.json())
        self.assertEqual(congested["background_fwd"], 0.5)
        self.assertEqual(congested["background_rev"], 0.5)
        response = self.client.post(f"{url}/perturb", json={**patch_link, "active": False})
        flapped = edge_link(response.json())
        self.assertEqual(flapped["status"], "flapped")
        self.assertEqual(flapped["capacity_fwd_mb"], 0)
        response = self.client.post(f"{url}/perturb", json={**patch_link, "active": True})
        self.assertEqual(edge_link(response.json())["rate_mbps"], congested["rate_mbps"])

        reject = preview["current_task"]["candidates"][-1]["action_id"]
        snapshot = self.client.post(f"{url}/step", json={"action_id": reject}).json()
        self.assertEqual(snapshot["task_index"], 1)
        self.assertEqual(snapshot["metrics"]["tasks"], 1)
        self.assertFalse(snapshot["recent_tasks"][0]["accepted"])
        self.assertEqual(snapshot["last_decision"]["destination"], -2)
        snapshot = self.client.post(f"{url}/step-slot").json()
        self.assertTrue(snapshot["done"])
        self.assertIsNone(snapshot["current_task"])
        self.assertEqual(len(snapshot["recent_tasks"]), 2)
        self.assertEqual(self.client.post(f"{url}/reset").json(), preview)
        self.assertEqual(self.client.delete(url).status_code, 204)
        self.assertEqual(self.client.get(url).status_code, 404)

        preview = self.client.post("/api/env/preview", json={"config": {**CONFIG, "n_edges": 1}})
        self.assertEqual(len(preview.json()["nodes"]), 2)

    def _training_events(self, run_id, *, stop=False):
        with self.client.websocket_connect(f"/ws/training/{run_id}") as socket:
            events = []
            while True:
                event = socket.receive_json()
                events.append(event)
                if stop and event["type"] == "metric":
                    self.assertEqual(self.client.post(f"/api/training/runs/{run_id}/stop")
                                     .status_code, 202)
                    stop = False
                if event["type"] == "terminal":
                    return events

    def test_train_resume_stop_and_evaluate(self):
        response = self.client.post("/api/training/runs", json={
            "config": CONFIG, "episodes": 1, "model_type": "flat", "threads": 1,
        })
        self.assertEqual(response.status_code, 202)
        run = response.json()
        run_id = run["run_id"]
        events = self._training_events(run_id)
        self.assertEqual(events[-1]["status"], "completed", events[-1])
        self.assertEqual([e["type"] for e in events], [
            "status", "status", "metric", "evaluation", "terminal"
        ])

        run = self.client.get(f"/api/training/runs/{run_id}").json()
        run_dir = Path(run["results_dir"])
        self.assertEqual(run_dir.parent, self.temp_dir)
        self.assertRegex(run_dir.name, r"^\d{8}_\d{6}$")
        self.assertEqual({p.name for p in run_dir.iterdir()}, {
            "train_config.json", "train_metrics.jsonl", "evaluation_metrics.jsonl",
            "best_ppo.pt", "last_ppo.pt", "training_metrics.png", "evaluation_metrics.png",
        })
        self.assertEqual(self.client.get("/api/training/runs").json()[0]["id"], run_id)
        self.assertEqual(self.client.get("/api/training/fs-runs").json()[0]["current_episode"], 1)
        self.assertEqual(len(self.client.get("/api/training/checkpoints").json()), 2)
        metrics = self.client.get(f"/api/training/runs/{run_id}/metrics").json()
        self.assertEqual(metrics["training"][0]["episode"], 1)
        self.assertEqual(metrics["evaluation"][0]["episode"], 1)

        checkpoint = run["last_checkpoint"]
        response = self.client.post("/api/experiments/evaluate", json={
            "config": CONFIG, "seeds": [11, 12],
            "strategies": [
                {"method": "shortest_fastest"}, {"method": "greedy_estimate"},
                {"method": "checkpoint", "checkpoint_path": checkpoint},
            ],
        })
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(len(result["strategies"]), 3)
        for strategy in result["strategies"]:
            self.assertEqual([entry["seed"] for entry in strategy["runs"]], [11, 12])
            self.assertIn("mean_load_variance_mean", strategy["summary"])
        self.assertEqual(result["strategies"][-1]["checkpoint_path"], checkpoint)

        response = self.client.post("/api/sim/sessions", json={
            "config": CONFIG, "policy": "checkpoint", "checkpoint_path": checkpoint,
        })
        self.assertEqual(response.status_code, 201)
        session_id = response.json()["session_id"]
        self.assertEqual(
            self.client.post(f"/api/sim/sessions/{session_id}/step").json()["task_index"], 1
        )

        resumed = self.manager.start_run(config_from_dict(CONFIG), episodes=2, threads=1,
                                         resume_path=checkpoint, run_dir=self.temp_dir / "resume")
        terminal = self._training_events(resumed["id"])[-1]
        self.assertEqual(terminal["status"], "completed", terminal)
        self.assertEqual(self.manager.metrics(resumed["id"])["training"][0]["episode"], 2)

        stopped = self.client.post("/api/training/runs", json={
            "config": CONFIG, "episodes": 100, "model_type": "flat", "threads": 1,
        }).json()["run_id"]
        terminal = self._training_events(stopped, stop=True)[-1]
        self.assertEqual(terminal["status"], "stopped", terminal)
        details = terminal["run"]
        saved = torch.load(details["last_checkpoint"], weights_only=True)
        self.assertEqual(saved["episode"], details["current_episode"])
        self.assertTrue((Path(details["results_dir"]) / "training_metrics.png").is_file())
