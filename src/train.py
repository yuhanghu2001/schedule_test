"""GAT-PPO 完整训练入口：训练、续训、日志、checkpoint 与五指标曲线。"""
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

from src.algorithms.ppo import PPOTrainer
from src.ablation import model_environment
from src.config import Config, load_config
from src.environment import EdgeEnv
from src.evaluation import evaluate_policy
from src.models.policy import GATPPO
from src.plotting import save_run_plots


def new_run_dir():
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / timestamp


def repository_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def save_checkpoint(path, model, trainer, env, cfg, model_type, episode, best_score):
    numpy_state = np.random.get_state()
    device = next(model.parameters()).device
    checkpoint = {
        "model_type": model_type,
        "config": asdict(cfg),
        "reward_version": cfg.reward_version,
        "observation_version": cfg.observation_version,
        "model": model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
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
    torch.save(checkpoint, path)


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.json")
parser.add_argument("--episodes", type=int, default=500,
                    help="训练结束时的总 episode 数")
parser.add_argument("--model", choices=("gat", "flat", "shuffled_gat"), default="gat")
parser.add_argument("--seed", type=int, help="覆盖新训练运行的配置种子")
parser.add_argument("--run-dir", help="配置和训练指标目录；默认创建北京时间结果目录")
parser.add_argument("--output", help="run-dir 内 last checkpoint 的文件名")
parser.add_argument("--resume", help="从 checkpoint 的模型、优化器和 episode 继续训练")
parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu",
                    help="模型训练设备；小图逐任务训练默认使用实测更快的 CPU")
parser.add_argument("--threads", type=int, default=4,
                    help="PyTorch CPU 线程数；小图模型默认 4 线程")
args = parser.parse_args()

config_path = repository_path(args.config)
resume_path = repository_path(args.resume) if args.resume else None
resume_checkpoint = None
if resume_path:
    resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
    cfg = Config(**resume_checkpoint["config"])
    model_type = resume_checkpoint["model_type"]
else:
    cfg = load_config(config_path)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    model_type = args.model
device_name = (
    "cuda" if args.device == "auto" and torch.cuda.is_available()
    else "cpu" if args.device == "auto"
    else args.device
)
device = torch.device(device_name)
torch.set_num_threads(args.threads)
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

run_dir = repository_path(args.run_dir) if args.run_dir else new_run_dir()
run_dir.mkdir(parents=True, exist_ok=bool(args.run_dir))
last_name = Path(args.output).name if args.output else f"last_{model_type}_ppo.pt"
last_output = run_dir / last_name
best_output = run_dir / f"best_{model_type}_ppo.pt"

model = GATPPO(use_gat=model_type != "flat").to(device)
trainer = PPOTrainer(model, cfg)
env = model_environment(EdgeEnv(cfg), model_type, cfg.seed)
start_episode = 0
best_score = -np.inf
if resume_checkpoint:
    model.load_state_dict(resume_checkpoint["model"])
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
    "config_source": str(resume_path or config_path),
    "model_type": model_type,
    "episodes": args.episodes,
    "resume": str(resume_path) if resume_path else None,
    "reward_version": cfg.reward_version,
    "observation_version": cfg.observation_version,
    "validation_seeds": [
        cfg.evaluation_seed + offset for offset in range(cfg.evaluation_seed_count)
    ],
    "best_output": str(best_output),
    "last_output": str(last_output),
    "device": str(device),
    "torch_threads": args.threads,
}
with open(run_dir / "train_config.json", "x", encoding="utf-8") as file:
    json.dump(run_config, file, ensure_ascii=False, indent=2)
print(json.dumps({"训练设备": str(device), "CPU线程数": args.threads}, ensure_ascii=False))

last_episode = start_episode
validation_seeds = [
    cfg.evaluation_seed + offset for offset in range(cfg.evaluation_seed_count)
]
interrupted = False
try:
    with (open(run_dir / "train_metrics.jsonl", "x", encoding="utf-8") as metrics_file,
          open(run_dir / "evaluation_metrics.jsonl", "x", encoding="utf-8") as evaluation_file):
        for episode_index in range(start_episode, args.episodes):
            rollout, metrics = trainer.collect_episode(env)
            update_metrics = trainer.update(rollout)
            last_episode = episode_index + 1
            record = {
                "episode": last_episode,
                "seed": cfg.seed,
                **update_metrics,
                **metrics,
            }
            metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps({
                "episode": last_episode,
                "累计奖励": metrics["episode_reward"],
                "任务完成率": metrics["task_success_rate"],
                "平均传输路径长度": metrics["mean_path_length"],
                "任务平均完成时间": metrics["avg_delay_s"],
                "任务平均能耗": metrics["avg_energy_j"],
                "负载均衡度": metrics["load_balance_degree"],
            }, ensure_ascii=False))
            if last_episode % cfg.eval_interval == 0 or last_episode == args.episodes:
                evaluation = {
                    "episode": last_episode,
                    **evaluate_policy(model, cfg, validation_seeds, model_type),
                }
                evaluation_file.write(json.dumps(evaluation, ensure_ascii=False) + "\n")
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
