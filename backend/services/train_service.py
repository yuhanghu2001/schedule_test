import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
from threading import Event, Lock
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
from sqlalchemy import select
import torch

from src.algorithms.ppo import PPOTrainer
from src.environment import EdgeEnv, model_environment
from src.evaluation import evaluate_policy
from src.models.policy import MODEL_NAMES
from src.plotting import load_jsonl, save_run_plots

from ..config import REPO_ROOT, RESULTS_DIR
from ..db import EnvConfig, EvaluationRecord, TrainingMetric, TrainingRun, row_dict
from .sim_service import build_model, checkpoint_config, checkpoint_metadata, load_checkpoint

BEIJING = ZoneInfo("Asia/Shanghai")
RUN_TIMESTAMP = "%Y%m%d_%H%M%S"
ACTIVE = {"queued", "running"}
METRIC_COLUMNS = set(TrainingMetric.__table__.columns.keys()) - {"id", "run_id", "created_at"}


def save_checkpoint(path, model, trainer, env, cfg, model_type, episode, best_score):
    # CLI checkpoint 格式。根目录 train.py 是命令行入口，src.trainer 提供训练引擎。
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    torch.save({
        "model_type": model_type,
        "config": asdict(cfg),
        "reward_version": cfg.reward_version,
        "observation_version": cfg.observation_version,
        "enable_dag_mode": cfg.enable_dag_mode,
        "model": model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "episode": episode,
        "best_evaluation_score": best_score,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(trainer.device) if trainer.device.type == "cuda"
        else None,
        "numpy_rng_state": {
            "name": name,
            "keys": torch.from_numpy(keys),
            "position": position,
            "has_gauss": has_gauss,
            "cached_gaussian": cached_gaussian,
        },
        "env_rng_state": env.rng.bit_generator.state,
    }, path)


def restore_training(trainer, env, checkpoint):
    optimizer_state = checkpoint["optimizer"]
    if len(optimizer_state["param_groups"]) == 1:
        # 早期权重将 Actor 和 Critic 的 Adam 状态保存在同一组。
        saved_group = optimizer_state["param_groups"][0]
        actor, critic = trainer.optimizer.state_dict()["param_groups"]
        split = len(actor["params"])
        optimizer_state = {**optimizer_state, "param_groups": [
            {**saved_group, "name": actor["name"], "params": saved_group["params"][:split]},
            {**saved_group, "name": critic["name"], "params": saved_group["params"][split:]},
        ]}
    trainer.optimizer.load_state_dict(optimizer_state)
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if trainer.device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], trainer.device)
    state = checkpoint["numpy_rng_state"]
    np.random.set_state((state["name"], state["keys"].numpy(), state["position"],
                         state["has_gauss"], state["cached_gaussian"]))
    env.rng.bit_generator.state = checkpoint["env_rng_state"]


def metric_row(run_id, record):
    return TrainingMetric(run_id=run_id, **{
        key: value for key, value in record.items() if key in METRIC_COLUMNS
    })


def evaluation_row(run, record):
    return EvaluationRecord(
        name=f"验证 · 第 {record['episode']} 轮",
        method=run["model_type"],
        run_id=run["id"],
        checkpoint_path=record.get("checkpoint_path"),
        env_config_id=run["env_config_id"],
        seeds_json=record.get("evaluation_seeds", []),
        summary_json=record,
    )


def _read_records(run_dir, filename):
    path = Path(run_dir) / filename
    return load_jsonl(path) if path.is_file() else []


def read_run_metrics(run_dir):
    return {
        "training": _read_records(run_dir, "train_metrics.jsonl"),
        "evaluation": _read_records(run_dir, "evaluation_metrics.jsonl"),
    }


def _append_record(file, record):
    file.write(json.dumps(record, ensure_ascii=False) + "\n")
    file.flush()


def _run_paths(results_dir):
    return sorted(results_dir.rglob("train_config.json"), reverse=True)


def _disk_run(config_path, config, metrics, scores):
    run_dir = config_path.parent.resolve()
    model_type = config["model_type"]
    current = metrics[-1]["episode"] if metrics else config.get("start_episode", 0)
    best = run_dir / Path(config["best_output"]).name
    last = run_dir / Path(config["last_output"]).name
    status = config.get("status")
    if status is None:
        status = "completed" if current >= config["episodes"] else "stopped"
    return {
        "run_name": config.get("run_name", run_dir.name),
        "model_type": model_type,
        "env_config_id": config.get("env_config_id"),
        "config_json": config,
        "episodes": config["episodes"],
        "current_episode": current,
        "seed": config["config"]["seed"],
        "device": config.get("device", "cpu"),
        "threads": config.get("torch_threads", 2),
        "status": status,
        "best_score": max(scores, default=None),
        "results_dir": str(run_dir),
        "best_checkpoint": str(best) if best.is_file() else None,
        "last_checkpoint": str(last) if last.is_file() else None,
        "error_message": config.get("error_message"),
    }


def _read_config(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_config(path, config):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _scores(records):
    return [record["objective_score_mean"] for record in records
            if record.get("objective_score_mean") is not None]


def list_checkpoints(results_dir=RESULTS_DIR):
    return [
        {"name": path.name, "path": str(path.resolve()),
         "run_dir": path.parent.relative_to(results_dir).as_posix(),
         **checkpoint_metadata(load_checkpoint(path.resolve()))}
        for path in sorted(results_dir.rglob("*.pt"), reverse=True)
    ]


def list_fs_runs(results_dir=RESULTS_DIR):
    runs = []
    for path in _run_paths(results_dir):
        config = _read_config(path)
        records = read_run_metrics(path.parent)
        runs.append({
            "run_id": path.parent.relative_to(results_dir).as_posix(),
            **_disk_run(path, config, records["training"], _scores(records["evaluation"])),
            "latest_metrics": records["training"][-1] if records["training"] else None,
        })
    return runs


def sync_results_to_db(db, results_dir=RESULTS_DIR):
    known = set(db.scalars(select(TrainingRun.results_dir)))
    configs = set(db.scalars(select(EnvConfig.id)))
    imported = 0
    skipped = 0
    for path in _run_paths(results_dir):
        run_dir = path.parent.resolve()
        if str(run_dir) in known:
            skipped += 1
            continue
        config = _read_config(path)
        if config.get("status") in ACTIVE:
            skipped += 1
            continue
        records = read_run_metrics(run_dir)
        run = {
            "id": config.get("run_id") or uuid4().hex,
            **_disk_run(path, config, records["training"], _scores(records["evaluation"])),
            "created_at": datetime.strptime(run_dir.name, RUN_TIMESTAMP).replace(
                tzinfo=BEIJING
            ).astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
            "finished_at": datetime.utcfromtimestamp(
                (run_dir / "train_metrics.jsonl").stat().st_mtime
                if (run_dir / "train_metrics.jsonl").is_file() else path.stat().st_mtime
            ),
        }
        if run["env_config_id"] not in configs:
            run["env_config_id"] = None
        db.add(TrainingRun(**run))
        db.flush()
        db.add_all(metric_row(run["id"], record) for record in records["training"])
        db.add_all(evaluation_row(run, record) for record in records["evaluation"])
        imported += 1
    db.commit()
    return {"imported": imported, "skipped": skipped}


@dataclass
class Run:
    data: dict
    stop_event: Event = field(default_factory=Event)
    events: list = field(default_factory=list)
    subscribers: set = field(default_factory=set)
    lock: Lock = field(default_factory=Lock)

    def emit(self, event_type, **payload):
        event = {"type": event_type, "run_id": self.data["id"], **payload}
        with self.lock:
            self.events.append(event)
            for loop, queue in self.subscribers:
                loop.call_soon_threadsafe(queue.put_nowait, event)

    def subscribe(self):
        subscription = (asyncio.get_running_loop(), asyncio.Queue())
        with self.lock:
            queue = subscription[1]
            for event in self.events:
                queue.put_nowait(event)
            self.subscribers.add(subscription)
        return subscription

    def unsubscribe(self, subscription):
        with self.lock:
            self.subscribers.remove(subscription)


class TrainManager:
    def __init__(self, db_factory=None, results_dir=RESULTS_DIR):
        self.db_factory = db_factory
        self.results_dir = Path(results_dir)
        # PyTorch 的线程数和随机数状态是进程共享的。
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training")
        self.runs = {}
        self.lock = Lock()

    def _new_run_dir(self):
        timestamp = datetime.now(BEIJING)
        run_dir = self.results_dir / timestamp.strftime(RUN_TIMESTAMP)
        while run_dir.exists():
            timestamp += timedelta(seconds=1)
            run_dir = self.results_dir / timestamp.strftime(RUN_TIMESTAMP)
        return run_dir

    def start_run(self, cfg, *, episodes, model_type="gatppo", device="cpu", threads=2,
                  resume_path=None, run_name=None, env_config_id=None, run_dir=None):
        resume_path = str(REPO_ROOT / resume_path) if resume_path else None
        checkpoint = load_checkpoint(resume_path) if resume_path else None
        start_episode = 0
        if checkpoint is not None:
            cfg = checkpoint_config(checkpoint)
            model_type = checkpoint["model_type"]
            start_episode = checkpoint["episode"]

        model_type = MODEL_NAMES[model_type]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        run_id = uuid4().hex
        custom_dir = run_dir is not None
        with self.lock:
            run_dir = Path(run_dir).resolve() if custom_dir else self._new_run_dir()
            if custom_dir and run_dir.is_dir() and any(run_dir.iterdir()):
                raise FileExistsError(run_dir)
            best = run_dir / f"best_{model_type}.pt"
            last = run_dir / f"last_{model_type}.pt"
            config = {
                "run_id": run_id,
                "run_name": run_name or run_dir.name,
                "env_config_id": env_config_id,
                "config": asdict(cfg),
                "config_source": resume_path or "api",
                "model_type": model_type,
                "episodes": episodes,
                "start_episode": start_episode,
                "resume": resume_path,
                "reward_version": cfg.reward_version,
                "observation_version": cfg.observation_version,
                "enable_dag_mode": cfg.enable_dag_mode,
                "validation_seeds": list(range(cfg.evaluation_seed,
                                               cfg.evaluation_seed + cfg.evaluation_seed_count)),
                "best_output": str(best),
                "last_output": str(last),
                "device": device,
                "torch_threads": threads,
            }
            data = {
                "id": run_id,
                "run_name": config["run_name"],
                "model_type": model_type,
                "env_config_id": env_config_id,
                "config_json": config,
                "episodes": episodes,
                "current_episode": start_episode,
                "seed": cfg.seed,
                "device": device,
                "threads": threads,
                "status": "queued",
                "best_score": None,
                "results_dir": str(run_dir),
                "best_checkpoint": None,
                "last_checkpoint": None,
                "error_message": None,
                "created_at": datetime.utcnow(),
                "finished_at": None,
            }
            if self.db_factory is not None:
                with self.db_factory() as db, db.begin():
                    db.add(TrainingRun(**data))
                    db.flush()
                    run_dir.mkdir(parents=True, exist_ok=custom_dir)
            else:
                run_dir.mkdir(parents=True, exist_ok=custom_dir)
            run = Run(data)
            self.runs[run_id] = run
            run.emit("status", status="queued")
            queued = data.copy()
            self.executor.submit(self._worker, run, cfg, checkpoint)
        return queued

    def list_runs(self, offset, limit):
        if self.db_factory is None:
            with self.lock:
                return [run.data.copy() for run in reversed(self.runs.values())][offset:offset + limit]
        with self.db_factory() as db:
            query = select(TrainingRun).order_by(TrainingRun.created_at.desc()).offset(offset).limit(limit)
            return [row_dict(row) for row in db.scalars(query)]

    def get_run(self, run_id):
        run = self.runs.get(run_id)
        if run is not None:
            return run.data.copy()
        if self.db_factory is not None:
            with self.db_factory() as db:
                row = db.get(TrainingRun, run_id)
                if row is not None:
                    return row_dict(row)
        raise KeyError(run_id)

    def stop_run(self, run_id):
        run = self.runs[run_id]
        run.stop_event.set()
        return run.data.copy()

    def metrics(self, run_id):
        run = self.runs.get(run_id)
        if run is None:
            return read_run_metrics(self.get_run(run_id)["results_dir"])
        with run.lock:
            training = [event["metrics"] for event in run.events if event["type"] == "metric"]
            evaluation = [event["metrics"] for event in run.events if event["type"] == "evaluation"]
        return {"training": training, "evaluation": evaluation}

    def active_count(self):
        with self.lock:
            return sum(run.data["status"] in ACTIVE for run in self.runs.values())

    def _update(self, run, extra_row=None, **changes):
        if self.db_factory is not None:
            with self.db_factory() as db, db.begin():
                row = db.get(TrainingRun, run.data["id"])
                for key, value in changes.items():
                    setattr(row, key, value)
                if extra_row is not None:
                    db.add(extra_row)
        run.data.update(changes)

    def _worker(self, run, cfg, checkpoint):
        run_dir = Path(run.data["results_dir"])
        config_path = run_dir / "train_config.json"
        config = {"status": "running", **run.data["config_json"]}
        try:
            _write_config(config_path, config)
            self._update(run, status="running")
            run.emit("status", status="running")
            self._train(run, cfg, checkpoint, run_dir, config)
            status = "stopped" if run.data["current_episode"] < config["episodes"] else "completed"
            config["status"] = status
            _write_config(config_path, config)
            self._update(run, status=status, finished_at=datetime.utcnow())
        except Exception as error:
            changes = {"status": "failed", "error_message": str(error),
                       "finished_at": datetime.utcnow()}
            config.update(status="failed", error_message=str(error))
            run.data.update(changes)
            try:
                _write_config(config_path, config)
                self._update(run, **changes)
            finally:
                run.emit("terminal", status="failed", run=run.data.copy())
        else:
            run.emit("terminal", status=status, run=run.data.copy())

    def _train(self, run, cfg, checkpoint, run_dir, config):
        model_type = config["model_type"]
        torch.set_num_threads(config["torch_threads"])
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        model = build_model(cfg, model_type, checkpoint).to(config["device"])
        trainer = PPOTrainer(model, cfg)
        env = model_environment(EdgeEnv(cfg), model_type, cfg.seed)
        if checkpoint is not None:
            restore_training(trainer, env, checkpoint)
        episode = config["start_episode"]
        best_score = -float("inf")
        with ((run_dir / "train_metrics.jsonl").open("x", encoding="utf-8") as metrics_file,
              (run_dir / "evaluation_metrics.jsonl").open("x", encoding="utf-8") as eval_file):
            for episode_number in range(episode + 1, config["episodes"] + 1):
                if run.stop_event.is_set():
                    break
                rollout, metrics = trainer.collect_episode(env)
                record = {"episode": episode_number, "seed": cfg.seed,
                          **trainer.update(rollout), **metrics}
                episode = episode_number
                _append_record(metrics_file, record)
                self._update(run, metric_row(run.data["id"], record), current_episode=episode)
                run.emit("metric", metrics=record)
                if episode % cfg.eval_interval == 0 or episode == config["episodes"]:
                    evaluation = {
                        "episode": episode,
                        **evaluate_policy(model, cfg, config["validation_seeds"], model_type),
                    }
                    changes = {}
                    if evaluation["objective_score_mean"] > best_score:
                        best_score = evaluation["objective_score_mean"]
                        save_checkpoint(config["best_output"], model, trainer, env, cfg,
                                        model_type, episode, best_score)
                        changes = {"best_score": best_score, "best_checkpoint": config["best_output"]}
                    _append_record(eval_file, evaluation)
                    self._update(run, evaluation_row(run.data, evaluation), **changes)
                    run.emit("evaluation", metrics=evaluation)

        save_checkpoint(config["last_output"], model, trainer, env, cfg,
                        model_type, episode, best_score)
        self._update(run, last_checkpoint=config["last_output"])
        save_run_plots(run_dir)
