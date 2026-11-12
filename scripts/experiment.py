import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np

from src.config import load_config


METRICS = (
    "episode_reward",
    "task_success_rate",
    "avg_delay_s",
    "avg_energy_j",
    "load_balance_degree",
)


def new_run_dir():
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    return Path("results") / timestamp


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.json")
parser.add_argument("--evaluation-config", help="评估场景配置；默认与训练配置相同")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
parser.add_argument(
    "--methods", nargs="+",
    choices=("gat", "flat", "shuffled_gat", "heuristic"),
    default=("gat", "flat", "shuffled_gat", "heuristic"),
)
parser.add_argument("--run-dir", help="组合实验目录；默认创建北京时间结果目录")
args = parser.parse_args()

cfg = load_config(args.config)
evaluation_config_source = args.evaluation_config or args.config
evaluation_cfg = load_config(evaluation_config_source)
run_dir = Path(args.run_dir) if args.run_dir else new_run_dir()
run_dir.mkdir(parents=True, exist_ok=bool(args.run_dir))

experiment_config = {
    "config": asdict(cfg),
    "config_source": args.config,
    "evaluation_config": asdict(evaluation_cfg),
    "evaluation_config_source": evaluation_config_source,
    "episodes": args.episodes,
    "seeds": args.seeds,
    "methods": args.methods,
}
with open(run_dir / "experiment_config.json", "x", encoding="utf-8") as file:
    json.dump(experiment_config, file, ensure_ascii=False, indent=2)

method_metrics = {method: [] for method in args.methods}
for method in args.methods:
    for seed in args.seeds:
        seed_dir = run_dir / method / f"seed_{seed}"
        evaluation_seed = evaluation_cfg.evaluation_seed + seed
        if method == "heuristic":
            command = [
                sys.executable, "-m", "scripts.evaluate_heuristic",
                "--config", evaluation_config_source,
                "--seed", str(evaluation_seed),
                "--run-dir", str(seed_dir),
            ]
        else:
            subprocess.run([
                sys.executable, "src/train.py",
                "--config", args.config,
                "--episodes", str(args.episodes),
                "--model", method,
                "--seed", str(seed),
                "--run-dir", str(seed_dir),
            ], check=True)
            checkpoint = seed_dir / f"{method}_ppo.pt"
            command = [
                sys.executable, "-m", "scripts.evaluate",
                "--checkpoint", str(checkpoint),
                "--config", evaluation_config_source,
                "--seed", str(evaluation_seed),
                "--run-dir", str(seed_dir),
            ]
        subprocess.run(command, check=True)
        with open(seed_dir / "metrics.json", encoding="utf-8") as file:
            method_metrics[method].append(json.load(file))

summary = {}
for method, runs in method_metrics.items():
    summary[method] = {}
    for metric in METRICS:
        values = [run[metric] for run in runs if run[metric] is not None]
        summary[method][metric] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "count": len(values),
        }

with open(run_dir / "summary.json", "x", encoding="utf-8") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
