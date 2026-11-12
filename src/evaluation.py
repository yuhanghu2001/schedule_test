"""固定种子、贪心动作的策略评估。"""
from dataclasses import replace

import numpy as np

from .ablation import model_environment
from .environment import EdgeEnv


EVALUATION_METRICS = (
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
    "mean_action_count",
    "local_execution_rate",
    "edge_execution_rate",
    "cloud_execution_rate",
    "mean_selected_hops",
)


def evaluate_episode(model, cfg, seed, model_type):
    env = model_environment(EdgeEnv(replace(cfg, seed=seed)), model_type, seed)
    obs = env.reset()
    done = False
    while not done:
        action, _, _ = model.act(obs, greedy=True)
        obs, _, done, _ = env.step(action)
    return env.metrics()


def evaluate_policy(model, cfg, seeds, model_type):
    was_training = model.training
    model.eval()
    runs = [evaluate_episode(model, cfg, seed, model_type) for seed in seeds]
    if was_training:
        model.train()
    summary = {"evaluation_seeds": list(seeds)}
    for field in EVALUATION_METRICS:
        values = [run[field] for run in runs if run[field] is not None]
        summary[f"{field}_mean"] = float(np.mean(values)) if values else None
        summary[f"{field}_std"] = float(np.std(values)) if values else None
    return summary
