import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from src.config import load_config
from src.environment import EdgeEnv
from src.evaluation import evaluate_heuristic
from src.train import PRESET_NAMES


def new_run_dir():
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    return Path("results") / timestamp


parser = argparse.ArgumentParser()
parser.add_argument("--config", "--preset", choices=PRESET_NAMES, default="default")
parser.add_argument("--seed", type=int, help="覆盖评估种子")
parser.add_argument(
    "--method", choices=("shortest_fastest", "greedy_estimate", "greedy", "heft_hetero"),
    default="shortest_fastest",
)
parser.add_argument("--run-dir", help="评估配置和指标目录；默认创建北京时间结果目录")
args = parser.parse_args()

cfg = load_config(args.config)
if args.method == "heft_hetero" and not cfg.enable_dag_mode:
    parser.error("heft_hetero 需要异构 DAG 预设")
evaluation_seed = args.seed if args.seed is not None else cfg.evaluation_seed
eval_cfg = replace(cfg, seed=evaluation_seed)
np.random.seed(evaluation_seed)

run_dir = Path(args.run_dir) if args.run_dir else new_run_dir()
run_dir.mkdir(parents=True, exist_ok=bool(args.run_dir))

metrics = {
    "method": args.method,
    "evaluation_seed": evaluation_seed,
    **evaluate_heuristic(EdgeEnv(eval_cfg), args.method),
}
with open(run_dir / "evaluation_config.json", "x", encoding="utf-8") as file:
    json.dump({"config": asdict(cfg), "config_source": args.config},
              file, ensure_ascii=False, indent=2)
with open(run_dir / "metrics.json", "x", encoding="utf-8") as file:
    json.dump(metrics, file, ensure_ascii=False, indent=2)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
