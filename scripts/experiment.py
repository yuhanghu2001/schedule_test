import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from src.config import load_config
from src.evaluation import DAG_EVALUATION_METRICS, summarize_metrics
from src.plotting import save_method_bar_plot
from src.trainer import PRESET_NAMES, TRAIN_MODEL_NAMES, require_observation_version


METRICS = (
    "objective_score",
    "episode_reward",
    "task_success_rate",
    "avg_delay_s",
    "avg_energy_j",
    "mean_load_variance",
    "load_balance_degree",
    "feasible_task_rate",
    "conditional_reject_rate",
    "forced_reject_rate",
    "mean_hops_accepted",
    "mean_path_length",
)
HEURISTICS = {
    "heuristic": "shortest_fastest", "shortest_fastest": "shortest_fastest",
    "greedy_estimate": "greedy_estimate", "greedy": "greedy", "heft_hetero": "heft_hetero",
}


def new_run_dir():
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    return Path("results") / timestamp


parser = argparse.ArgumentParser()
parser.add_argument("--config", "--preset", choices=PRESET_NAMES, default="default")
parser.add_argument("--evaluation-config",
                    choices=PRESET_NAMES,
                    help="评估场景配置；默认与训练配置相同")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
parser.add_argument(
    "--methods", nargs="+",
    choices=(*TRAIN_MODEL_NAMES, *HEURISTICS),
)
parser.add_argument("--run-dir", help="组合实验目录；默认创建北京时间结果目录")
parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
parser.add_argument("--threads", type=int, default=4)
args = parser.parse_args()
cfg = load_config(args.config)
evaluation_config_source = args.evaluation_config or args.config
evaluation_cfg = load_config(evaluation_config_source)
default_methods = ("gatppo", "ppo", "shuffled_gat", "heuristic")
if cfg.enable_dag_mode:
    default_methods = ("dual_gatppo", "single_gatppo", "ppo", "shuffled_gat", "heft_hetero", "greedy")
methods = list(dict.fromkeys(TRAIN_MODEL_NAMES.get(method, method)
                             for method in (args.methods or default_methods)))
if any(method in TRAIN_MODEL_NAMES for method in methods):
    require_observation_version(cfg.observation_version, evaluation_cfg.observation_version)
if "heft_hetero" in methods and not evaluation_cfg.enable_dag_mode:
    parser.error("heft_hetero 需要异构 DAG 评估预设")
run_dir = Path(args.run_dir) if args.run_dir else new_run_dir()
run_dir.mkdir(parents=True, exist_ok=bool(args.run_dir))

experiment_config = {
    "config": asdict(cfg),
    "config_source": args.config,
    "evaluation_config": asdict(evaluation_cfg),
    "evaluation_config_source": evaluation_config_source,
    "episodes": args.episodes,
    "seeds": args.seeds,
    "methods": methods,
    "device": args.device,
    "torch_threads": args.threads,
}
with open(run_dir / "experiment_config.json", "x", encoding="utf-8") as file:
    json.dump(experiment_config, file, ensure_ascii=False, indent=2)

method_metrics = {method: [] for method in methods}
for method in methods:
    for seed in args.seeds:
        seed_dir = run_dir / method / f"seed_{seed}"
        evaluation_seed = evaluation_cfg.evaluation_seed + seed
        if method in HEURISTICS:
            command = [
                sys.executable, "-m", "scripts.evaluate_heuristic",
                "--config", evaluation_config_source,
                "--method", HEURISTICS[method],
                "--seed", str(evaluation_seed),
                "--run-dir", str(seed_dir),
            ]
        else:
            subprocess.run([
                sys.executable, "src/trainer.py",
                "--config", args.config,
                "--episodes", str(args.episodes),
                "--model", method,
                "--seed", str(seed),
                "--run-dir", str(seed_dir),
                "--device", args.device,
                "--threads", str(args.threads),
            ], check=True)
            checkpoint = seed_dir / f"last_{method}.pt"
            command = [
                sys.executable, "-m", "scripts.evaluate",
                "--checkpoint", str(checkpoint),
                "--config", evaluation_config_source,
                "--seed", str(evaluation_seed),
                "--run-dir", str(seed_dir),
                "--device", args.device,
                "--threads", str(args.threads),
            ]
        subprocess.run(command, check=True)
        with open(seed_dir / "metrics.json", encoding="utf-8") as file:
            method_metrics[method].append(json.load(file))

fields = METRICS + (DAG_EVALUATION_METRICS if evaluation_cfg.enable_dag_mode else ())
summary = {method: summarize_metrics(runs, fields) for method, runs in method_metrics.items()}

with open(run_dir / "summary.json", "x", encoding="utf-8") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)
if evaluation_cfg.enable_dag_mode:
    save_method_bar_plot({
        method: {f"{field}_{stat}": values[stat] for field, values in metrics.items()
                 for stat in ("mean", "std")}
        for method, metrics in summary.items()
    }, run_dir / "method_metrics_bar.png")
print(json.dumps(summary, ensure_ascii=False, indent=2))
