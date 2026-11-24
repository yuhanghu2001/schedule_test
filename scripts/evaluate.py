import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from src.config import load_config
from src.evaluation import evaluate_episode
from src.train import PRESET_NAMES, load_checkpoint, make_model, require_observation_version


def new_run_dir():
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    return Path("results") / timestamp


parser = argparse.ArgumentParser()
parser.add_argument("--config", "--preset", choices=PRESET_NAMES,
                    help="评估场景配置；默认使用 checkpoint 配置")
parser.add_argument("--observation-version", help="要求 checkpoint 使用指定观测版本")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seed", type=int, help="覆盖评估种子")
parser.add_argument("--run-dir", help="评估配置和指标目录；默认创建北京时间结果目录")
parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
parser.add_argument("--threads", type=int, default=4)
args = parser.parse_args()

checkpoint_source = Path(args.checkpoint).resolve()
checkpoint, training_cfg, model_type = load_checkpoint(
    checkpoint_source, observation_version=args.observation_version,
)
cfg = load_config(args.config) if args.config else training_cfg
require_observation_version(training_cfg.observation_version, cfg.observation_version)
evaluation_seed = args.seed if args.seed is not None else cfg.evaluation_seed
device_name = (
    "cuda" if args.device == "auto" and torch.cuda.is_available()
    else "cpu" if args.device == "auto"
    else args.device
)
device = torch.device(device_name)
torch.set_num_threads(args.threads)
torch.manual_seed(evaluation_seed)
np.random.seed(evaluation_seed)

model = make_model(training_cfg, model_type).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

run_dir = Path(args.run_dir) if args.run_dir else new_run_dir()
run_dir.mkdir(parents=True, exist_ok=bool(args.run_dir))
evaluation = {
    "checkpoint_source": str(checkpoint_source),
    "checkpoint_episode": checkpoint["episode"],
    "model_type": model_type,
    "reward_version": training_cfg.reward_version,
    "observation_version": training_cfg.observation_version,
    "enable_dag_mode": cfg.enable_dag_mode,
    "evaluation_seed": evaluation_seed,
    "device": str(device),
    **evaluate_episode(model, cfg, evaluation_seed, model_type),
}
with open(run_dir / "evaluation_config.json", "x", encoding="utf-8") as file:
    json.dump({"evaluation_config": asdict(cfg),
               "config_source": args.config or str(checkpoint_source),
               "checkpoint_source": str(checkpoint_source),
               "training_config": asdict(training_cfg), "model_type": model_type,
               "device": str(device), "torch_threads": args.threads},
              file, ensure_ascii=False, indent=2)
with open(run_dir / "metrics.json", "x", encoding="utf-8") as file:
    json.dump(evaluation, file, ensure_ascii=False, indent=2)
print(json.dumps(evaluation, ensure_ascii=False, indent=2))
