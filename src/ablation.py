"""消融实验入口：直接对比完整 GAT-PPO 与单纯 PPO（去掉 GAT）。"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config

METRICS = (
    "objective_score",
    "episode_reward",
    "task_success_rate",
    "acceptance_rate",
    "deadline_miss_rate",
    "avg_delay_s",
    "avg_energy_j",
    "mean_load_variance",
    "load_balance_degree",
)

ABLATION_MODELS = (
    ("GAT-PPO", "gat"),
    ("PPO", "flat"),
)


def shuffle_graph_observation(obs, permutation):
    shuffled = dict(obs)
    shuffled["adjacency"] = obs["adjacency"][permutation][:, permutation]
    shuffled["edges"] = obs["edges"][permutation][:, permutation]
    return shuffled


class ShuffledGraphEnv:
    """保持真实动作与约束不变，只扰动策略看到的图连接。"""

    def __init__(self, env, seed):
        self.env = env
        edge_nodes = np.random.default_rng(seed).permutation(env.cloud)
        self.permutation = np.append(edge_nodes, env.cloud)

    @property
    def rng(self):
        return self.env.rng

    def reset(self):
        return shuffle_graph_observation(self.env.reset(), self.permutation)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not done:
            obs = shuffle_graph_observation(obs, self.permutation)
        return obs, reward, done, info

    def metrics(self):
        return self.env.metrics()


def model_environment(env, model_type, seed):
    if model_type == "shuffled_gat":
        return ShuffledGraphEnv(env, seed)
    return env


def new_run_dir():
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / timestamp


def run_ablation():
    parser = argparse.ArgumentParser(description="对比 GAT-PPO 与单纯 PPO")
    parser.add_argument("--config", choices=("default", "high_congestion", "small_smoke"),
                        default="default")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--run-dir", help="消融结果目录；默认创建北京时间结果目录")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else new_run_dir()
    run_dir.mkdir(parents=True, exist_ok=bool(args.run_dir))

    with open(run_dir / "ablation_config.json", "x", encoding="utf-8") as file:
        json.dump({
            "config": asdict(cfg),
            "config_source": args.config,
            "episodes": args.episodes,
            "seeds": args.seeds,
            "models": [name for name, _ in ABLATION_MODELS],
            "device": args.device,
            "torch_threads": args.threads,
        }, file, ensure_ascii=False, indent=2)

    method_metrics = {name: [] for name, _ in ABLATION_MODELS}
    for name, model_type in ABLATION_MODELS:
        for seed in args.seeds:
            seed_dir = run_dir / name / f"seed_{seed}"
            evaluation_seed = cfg.evaluation_seed + seed
            subprocess.run([
                sys.executable, str(REPO_ROOT / "src" / "train.py"),
                "--config", args.config,
                "--episodes", str(args.episodes),
                "--model", model_type,
                "--seed", str(seed),
                "--run-dir", str(seed_dir),
                "--device", args.device,
                "--threads", str(args.threads),
            ], check=True)
            checkpoint = seed_dir / f"best_{model_type}_ppo.pt"
            subprocess.run([
                sys.executable, "-m", "scripts.evaluate",
                "--checkpoint", str(checkpoint),
                "--config", args.config,
                "--seed", str(evaluation_seed),
                "--run-dir", str(seed_dir),
                "--device", args.device,
                "--threads", str(args.threads),
            ], check=True, cwd=REPO_ROOT)
            with open(seed_dir / "metrics.json", encoding="utf-8") as file:
                method_metrics[name].append(json.load(file))

    summary = {}
    for name, runs in method_metrics.items():
        summary[name] = {}
        for metric in METRICS:
            values = [run[metric] for run in runs if run[metric] is not None]
            summary[name][metric] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values)) if values else None,
                "count": len(values),
            }

    with open(run_dir / "summary.json", "x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_ablation()
