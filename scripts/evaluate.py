import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from src.config import load_config
from src.config import Config
from src.evaluation import evaluate_episode
from src.models.policy import GATPPO


def new_run_dir():
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    return Path("results") / timestamp


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="评估场景配置；默认使用 checkpoint 配置")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seed", type=int, help="覆盖评估种子")
parser.add_argument("--run-dir", help="评估配置和指标目录；默认创建北京时间结果目录")
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
model_type = checkpoint["model_type"]
training_cfg = Config(**checkpoint["config"])
cfg = load_config(args.config) if args.config else training_cfg
evaluation_seed = args.seed if args.seed is not None else cfg.evaluation_seed
eval_cfg = replace(cfg, seed=evaluation_seed)
torch.set_num_threads(1)
torch.manual_seed(evaluation_seed)
np.random.seed(evaluation_seed)

run_dir = Path(args.run_dir) if args.run_dir else new_run_dir()
run_dir.mkdir(parents=True, exist_ok=bool(args.run_dir))

model = GATPPO(use_gat=model_type != "flat")
model.load_state_dict(checkpoint["model"])
model.eval()

evaluation = {
    "checkpoint_source": args.checkpoint,
    "checkpoint_episode": checkpoint["episode"],
    "model_type": model_type,
    "reward_version": checkpoint["reward_version"],
    "observation_version": checkpoint["observation_version"],
    "evaluation_seed": evaluation_seed,
    **evaluate_episode(model, eval_cfg, evaluation_seed, model_type),
}
with open(run_dir / "evaluation_config.json", "x", encoding="utf-8") as file:
    json.dump({"evaluation_config": asdict(cfg),
               "config_source": args.config or args.checkpoint,
               "training_config": checkpoint["config"], "model_type": model_type},
              file, ensure_ascii=False, indent=2)
with open(run_dir / "metrics.json", "x", encoding="utf-8") as file:
    json.dump(evaluation, file, ensure_ascii=False, indent=2)
print(json.dumps(evaluation, ensure_ascii=False, indent=2))
