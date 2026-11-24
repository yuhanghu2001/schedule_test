"""统一训练入口：GAT-PPO、PPO、DQN、DDPG 与对比绘图。"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.ddpg import DDPG, DDPGTrainer
from src.algorithms.dqn import DQN, DQNTrainer
from src.algorithms.ppo import PPOTrainer
from src.config import Config, load_config
from src.environment import EdgeEnv
from src.evaluation import evaluate_policy
from src.models.policy import GATPPO, MODEL_NAMES
from src.plotting import save_run_plots, save_comparison_plots


REPLAY_TRAINERS = {"dqn": DQNTrainer, "ddpg": DDPGTrainer}
TRAIN_MODEL_NAMES = {**MODEL_NAMES, "dqn": "dqn", "ddpg": "ddpg"}
PRESET_NAMES = (
    "default", "high_congestion", "small_smoke",
    "hetero_dag_default", "hetero_dag_high_congestion", "hetero_dag_smoke",
)


def make_model(cfg, model_type):
    model_type = TRAIN_MODEL_NAMES[model_type]
    if model_type in REPLAY_TRAINERS and cfg.enable_dag_mode:
        raise ValueError(f"{model_type} requires a v4 environment")
    if model_type in ("dual_gatppo", "single_gatppo") and not cfg.enable_dag_mode:
        raise ValueError(f"{model_type} requires a v5 DAG preset")
    if model_type == "dqn":
        return DQN(cfg)
    if model_type == "ddpg":
        return DDPG(cfg)
    return GATPPO.from_config(cfg, model_type)


def new_run_dir():
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / timestamp


def repository_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def require_observation_version(expected, requested):
    if requested is not None and requested != expected:
        raise ValueError(f"Incompatible observation_version: expected {expected!r}, got {requested!r}")


def load_checkpoint(path, *, observation_version=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    values = dict(checkpoint["config"])
    values.update({key: checkpoint[key] for key in ("observation_version", "reward_version", "enable_dag_mode")
                   if key in checkpoint})
    cfg = Config(**values)
    require_observation_version(cfg.observation_version, observation_version)
    return checkpoint, cfg, TRAIN_MODEL_NAMES[checkpoint["model_type"]]


def save_checkpoint(path, model, trainer, env, cfg, model_type, episode, best_score):
    numpy_state = np.random.get_state()
    device = next(model.parameters()).device
    checkpoint = {
        "model_type": model_type,
        "config": asdict(cfg),
        "reward_version": cfg.reward_version,
        "observation_version": cfg.observation_version,
        "enable_dag_mode": cfg.enable_dag_mode,
        "model": model.state_dict(),
        "episode": episode,
        "best_evaluation_score": best_score,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if device.type == "cuda" else None,
        "numpy_rng_state": {
            "name": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1]),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "env_rng_state": env.rng.bit_generator.state,
    }
    if model_type in REPLAY_TRAINERS:
        checkpoint["trainer"] = trainer.state_dict()
    else:
        checkpoint["optimizer"] = trainer.optimizer.state_dict()
    torch.save(checkpoint, path)


def train_model(cfg, model_type, run_dir, *, episodes, resume=None, observation_version=None,
                device="cpu", threads=2, output=None, config_source=None, existing_dir=True):
    """训练到指定总回合数，续训时沿用 checkpoint 的配置和算法。"""
    resume_path = Path(resume).resolve() if resume else None
    resume_checkpoint = None
    if resume_path:
        resume_checkpoint, saved_cfg, model_type = load_checkpoint(
            resume_path, observation_version=observation_version,
        )
        if cfg is not None:
            require_observation_version(saved_cfg.observation_version, cfg.observation_version)
        cfg = saved_cfg
    else:
        require_observation_version(cfg.observation_version, observation_version)
        model_type = TRAIN_MODEL_NAMES[model_type]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.set_num_threads(threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = make_model(cfg, model_type).to(device)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=existing_dir)
    last_name = Path(output).name if output else f"last_{model_type}.pt"
    last_output = run_dir / last_name
    best_output = run_dir / f"best_{model_type}.pt"

    off_policy = model_type in REPLAY_TRAINERS
    trainer = REPLAY_TRAINERS.get(model_type, PPOTrainer)(model, cfg)
    env = EdgeEnv(cfg)
    start_episode = 0
    best_score = -np.inf
    if resume_checkpoint:
        model.load_state_dict(resume_checkpoint["model"])
        if off_policy:
            trainer.load_state_dict(resume_checkpoint["trainer"])
        else:
            trainer.optimizer.load_state_dict(resume_checkpoint["optimizer"])
        start_episode = resume_checkpoint["episode"]
        torch.set_rng_state(resume_checkpoint["torch_rng_state"])
        if device.type == "cuda" and resume_checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(resume_checkpoint["cuda_rng_state"])
        numpy_state = resume_checkpoint["numpy_rng_state"]
        np.random.set_state((numpy_state["name"], numpy_state["keys"].numpy(),
                             numpy_state["position"], numpy_state["has_gauss"],
                             numpy_state["cached_gaussian"]))
        env.rng.bit_generator.state = resume_checkpoint["env_rng_state"]

    run_config = {
        "config": asdict(cfg),
        "config_source": str(resume_path) if resume_path else config_source,
        "model_type": model_type,
        "episodes": episodes,
        "resume": str(resume_path) if resume_path else None,
        "reward_version": cfg.reward_version,
        "observation_version": cfg.observation_version,
        "enable_dag_mode": cfg.enable_dag_mode,
        "validation_seeds": [
            cfg.evaluation_seed + offset for offset in range(cfg.evaluation_seed_count)
        ],
        "best_output": str(best_output),
        "last_output": str(last_output),
        "device": str(device),
        "torch_threads": threads,
    }
    with open(run_dir / "train_config.json", "x", encoding="utf-8") as file:
        json.dump(run_config, file, ensure_ascii=False, indent=2)
    print(json.dumps({"算法": model_type, "训练设备": str(device), "CPU线程数": threads}, ensure_ascii=False))

    last_episode = start_episode
    validation_seeds = [
        cfg.evaluation_seed + offset for offset in range(cfg.evaluation_seed_count)
    ]
    interrupted = False
    try:
        with (open(run_dir / "train_metrics.jsonl", "x", encoding="utf-8") as metrics_file,
              open(run_dir / "evaluation_metrics.jsonl", "x", encoding="utf-8") as evaluation_file):
            for episode_index in range(start_episode, episodes):
                if off_policy:
                    update_metrics, metrics = trainer.train_episode(env, episode_index, episodes)
                else:
                    rollout, metrics = trainer.collect_episode(env)
                    trainer.set_schedule(episode_index, episodes)
                    update_metrics = trainer.update(rollout)
                last_episode = episode_index + 1
                record = {
                    "model_type": model_type,
                    "episode": last_episode,
                    "seed": cfg.seed,
                    **update_metrics,
                    **metrics,
                }
                metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics_file.flush()
                print(json.dumps({
                    "算法": model_type,
                    "episode": last_episode,
                    "累计奖励": metrics["episode_reward"],
                    "任务完成率": metrics["task_success_rate"],
                    "平均传输路径长度": metrics["mean_path_length"],
                    "任务平均完成时间": metrics["avg_delay_s"],
                    "任务平均能耗": metrics["avg_energy_j"],
                    "负载均衡度": metrics["load_balance_degree"],
                }, ensure_ascii=False))
                if last_episode % cfg.eval_interval == 0 or last_episode == episodes:
                    evaluation = {
                        "episode": last_episode,
                        **evaluate_policy(model, cfg, validation_seeds, model_type),
                    }
                    evaluation_file.write(json.dumps(evaluation, ensure_ascii=False) + "\n")
                    evaluation_file.flush()
                    evaluation_score = evaluation["objective_score_mean"]
                    if evaluation_score > best_score:
                        best_score = evaluation_score
                        save_checkpoint(
                            best_output, model, trainer, env, cfg,
                            model_type, last_episode, best_score,
                        )
    except KeyboardInterrupt:
        interrupted = True
        print(json.dumps({"训练状态": "手动中断", "已完成回合": last_episode}, ensure_ascii=False))

    save_run_plots(run_dir)
    save_checkpoint(
        last_output, model, trainer, env, cfg,
        model_type, last_episode, best_score,
    )
    if interrupted:
        print(json.dumps({"中断收尾": "已生成曲线并保存 last checkpoint"}, ensure_ascii=False))

    return interrupted


def train_one(args, model_type, run_dir):
    config_source = args.config or "default"
    cfg = load_config(config_source) if args.config or not args.resume else None
    if cfg is not None and args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    return train_model(
        cfg, model_type, run_dir, episodes=args.episodes,
        resume=repository_path(args.resume) if args.resume else None,
        observation_version=args.observation_version, device=args.device, threads=args.threads,
        output=args.output, config_source=config_source, existing_dir=bool(args.run_dir),
    )


def main():
    parser = argparse.ArgumentParser(description="顺序训练所选算法并生成对比图")
    parser.add_argument("--config", "--preset", choices=PRESET_NAMES,
                        help="训练场景；默认 default，续训时使用 checkpoint 配置")
    parser.add_argument("--observation-version", help="要求配置或续训 checkpoint 使用指定观测版本")
    parser.add_argument("--episodes", type=int, default=500, help="训练结束时的总回合数")
    parser.add_argument("--model", nargs="+", choices=(*TRAIN_MODEL_NAMES, "both", "and", "gatppo and ppo", "gatpppo and ppo"),
                        metavar="MODEL",
                        default=["gatppo"], help="gatppo dual_gatppo single_gatppo ppo shuffled_gat dqn ddpg both")
    parser.add_argument("--seed", type=int, help="覆盖新训练运行的配置种子")
    parser.add_argument("--run-dir", help="默认创建北京时间 results 目录；开发验证须指定临时目录")
    parser.add_argument("--output", help="单算法 last checkpoint 的文件名")
    parser.add_argument("--resume", help="单算法从 checkpoint 继续训练")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    models = []
    for name in " ".join(args.model).split():
        if name == "both":
            models.extend(("gatppo", "ppo"))
        elif name != "and":
            models.append(TRAIN_MODEL_NAMES[name])
    models = list(dict.fromkeys(models))
    if not models:
        parser.error("--model 至少需要一个算法")
    comparison = len(models) > 1
    if comparison and (args.resume or args.output):
        parser.error("--resume 和 --output 用于单算法模式")
    run_dir = repository_path(args.run_dir) if args.run_dir else new_run_dir()
    if comparison:
        run_dir.mkdir(parents=True, exist_ok=bool(args.run_dir))
        config_source = args.config or "default"
        cfg = load_config(config_source)
        if args.seed is not None:
            cfg = replace(cfg, seed=args.seed)
        with (run_dir / "comparison_config.json").open("x", encoding="utf-8") as file:
            json.dump({"models": models, "config_source": config_source, "config": asdict(cfg),
                       "episodes": args.episodes, "device": args.device,
                       "torch_threads": args.threads}, file, ensure_ascii=False, indent=2)
        try:
            for model_type in models:
                if train_one(args, model_type, run_dir / model_type):
                    break
        finally:
            save_comparison_plots(run_dir, models)
        print(f"对比结果目录：{run_dir}")
    else:
        train_one(args, models[0], run_dir)


if __name__ == "__main__":
    main()
