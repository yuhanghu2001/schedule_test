import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from src.config import load_config
from src.environment import EdgeEnv


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/small_smoke.json")
parser.add_argument("--run-dir", help="配置和指标目录；默认创建北京时间结果目录")
args = parser.parse_args()
cfg = load_config(args.config)
timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
run_dir = Path(args.run_dir) if args.run_dir else Path("results") / timestamp
run_dir.mkdir(parents=True, exist_ok=bool(args.run_dir))
with open(run_dir / "smoke_config.json", "x", encoding="utf-8") as file:
    json.dump({"config": asdict(cfg), "config_source": args.config},
              file, ensure_ascii=False, indent=2)

env = EdgeEnv(cfg)
obs = env.reset()
rng = np.random.default_rng(10)
done = False
while not done:
    # 在可行候选集合内采样，拒绝动作保留为最后一个。
    a = int(rng.integers(len(env.actions)))
    obs, reward, done, info = env.step(a)
metrics = env.metrics()
with open(run_dir / "smoke_metrics.json", "x", encoding="utf-8") as file:
    json.dump(metrics, file, ensure_ascii=False, indent=2)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print("saved", run_dir)
