"""固定种子、贪心动作的策略评估。"""
from dataclasses import replace

import numpy as np

from .baselines import evaluate_heuristic as evaluate_heuristic
from .environment import EdgeEnv, model_environment
from .models.policy import GATPPO


EVALUATION_METRICS = (
    "objective_score",
    "episode_reward",
    "task_success_rate",
    "acceptance_rate",
    "deadline_miss_rate",
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
    "mean_hops_accepted",
    "mean_path_length",
)

DAG_EVALUATION_METRICS = (
    "dag_completion_rate",
    "avg_dag_makespan_s",
    "deadline_violation_rate",
    "avg_dag_energy_j",
    "fpga_reconfig_rate",
    "fpga_reconfig_delay_ratio",
    "cpu_load_std",
    "npu_load_std",
    "fpga_load_std",
    "hetero_load_balance_degree",
    "comm_tier0_ratio",
    "comm_tier1_ratio",
    "comm_tier2_ratio",
)


def summarize_metrics(runs, fields=None):
    if fields is None:
        fields = tuple(field for field in (*EVALUATION_METRICS, *DAG_EVALUATION_METRICS)
                       if field in runs[0])
    summary = {}
    for field in fields:
        values = [run[field] for run in runs if run[field] is not None and np.isfinite(run[field])]
        summary[field] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "count": len(values),
        }
    return summary


def evaluate_episode(model, cfg, seed, model_type):
    if isinstance(model, GATPPO) and model.dag_mode != cfg.enable_dag_mode:
        expected = "v5 DAG" if model.dag_mode else "v4"
        raise ValueError(f"Incompatible observation_version: model expects {expected}, got {cfg.observation_version!r}")
    env = model_environment(EdgeEnv(replace(cfg, seed=seed)), model_type, seed)
    if isinstance(model, GATPPO) and model.dag_mode:
        env = model.rollout_environment(env, seed)
    obs = env.reset()
    done = False
    while not done:
        action, _, _ = model.act(obs, greedy=True)
        obs, _, done, _ = env.step(action)
    return env.metrics()


def evaluate_policy(model, cfg, seeds, model_type):
    seeds = list(seeds)
    was_training = model.training
    model.eval()
    runs = [evaluate_episode(model, cfg, seed, model_type) for seed in seeds]
    if was_training:
        model.train()
    fields = EVALUATION_METRICS + (DAG_EVALUATION_METRICS if cfg.enable_dag_mode else ())
    return {"evaluation_seeds": seeds, **{
        f"{field}_{stat}": values[stat]
        for field, values in summarize_metrics(runs, fields).items()
        for stat in ("mean", "std")
    }}
