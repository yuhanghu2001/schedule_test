import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from src.baselines import evaluate_heuristic
from src.config import load_config
from src.environment import EdgeEnv


def new_run_dir():
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    return Path("results") / timestamp


parser = argparse.ArgumentParser()
parser.add_argument("--config", choices=("default", "high_congestion", "small_smoke"),
                    default="default")
parser.add_argument("--seed", type=int, help="覆盖评估种子")
parser.add_argument("--run-dir", help="评估配置和指标目录；默认创建北京时间结果目录")
args = parser.parse_args()

cfg = load_config(args.config)
evaluation_seed = args.seed if args.seed is not None else cfg.evaluation_seed
eval_cfg = replace(cfg, seed=evaluation_seed)
np.random.seed(evaluation_seed)

run_dir = Path(args.run_dir) if args.run_dir else new_run_dir()
run_dir.mkdir(parents=True, exist_ok=bool(args.run_dir))

metrics = {
    "method": "shortest_path_fastest_resource",
    "evaluation_seed": evaluation_seed,
    **evaluate_heuristic(EdgeEnv(eval_cfg)),
}
with open(run_dir / "evaluation_config.json", "x", encoding="utf-8") as file:
    json.dump({"config": asdict(cfg), "config_source": args.config},
              file, ensure_ascii=False, indent=2)
with open(run_dir / "metrics.json", "x", encoding="utf-8") as file:
    json.dump(metrics, file, ensure_ascii=False, indent=2)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
