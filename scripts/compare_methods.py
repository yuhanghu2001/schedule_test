"""在同一环境配置与测试种子上比较学习策略和启发式调度。"""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import evaluate_heuristic
from src.config import Config, load_config
from src.environment import EdgeEnv
from src.evaluation import EVALUATION_METRICS, evaluate_episode
from src.plotting import save_method_bar_plot, save_rl_reward_plot
from src.trainer import make_model


RL_METHODS = ("gatppo", "ppo", "dqn", "ddpg")
METHODS = (*RL_METHODS, "ga", "aco", "greedy_estimate", "shortest_fastest")


def summarize(runs):
    summary = {}
    for metric in EVALUATION_METRICS:
        # 成功任务的时延和能耗、接收任务的跳数可能没有定义。
        values = [run[metric] for run in runs if run[metric] is not None]
        summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
        summary[f"{metric}_std"] = float(np.std(values)) if values else None
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2042, 2052)))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", choices=("default", "high_congestion", "small_smoke"),
                        default="default")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = args.output_dir or run_dir / "method_comparison"
    cfg = load_config(args.config)
    torch.set_num_threads(2)
    summaries = {}
    details = {}
    for method in dict.fromkeys(args.methods):
        checkpoint_path = None
        if method in RL_METHODS:
            checkpoint_path = run_dir / method / f"best_{method}.pt"
            if not checkpoint_path.is_file():
                print(f"跳过 {method}，缺少权重 {checkpoint_path}", file=sys.stderr)
                continue
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model = make_model(Config(**checkpoint["config"]), method)
            model.load_state_dict(checkpoint["model"])
            del checkpoint
            model.eval()
            runs = [evaluate_episode(model, cfg, seed, method) for seed in args.seeds]
        else:
            runs = [
                evaluate_heuristic(EdgeEnv(replace(cfg, seed=seed)), method)
                for seed in args.seeds
            ]
        summaries[method] = summarize(runs)
        details[method] = {
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            **summaries[method],
            "runs": [{"seed": seed, **metrics} for seed, metrics in zip(args.seeds, runs)],
        }
        print(f"{method} 已完成 {len(runs)} 个测试回合", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    with (output / "method_comparison.json").open("x", encoding="utf-8") as file:
        json.dump({
            "config_source": args.config, "config": asdict(cfg),
            "test_seeds": args.seeds, "methods": details,
        }, file, ensure_ascii=False, indent=2)
    with (output / "method_comparison.csv").open("x", encoding="utf-8", newline="") as file:
        fields = [f"{metric}_{stat}" for metric in EVALUATION_METRICS for stat in ("mean", "std")]
        writer = csv.DictWriter(file, fieldnames=["method", "checkpoint", *fields])
        writer.writeheader()
        for method, summary in summaries.items():
            writer.writerow({"method": method, "checkpoint": details[method]["checkpoint"], **summary})
    save_method_bar_plot(summaries, output / "method_metrics_bar.png")
    save_rl_reward_plot(run_dir, RL_METHODS, output / "rl_training_reward.png")
    print(f"对比结果目录：{output}")


if __name__ == "__main__":
    main()
