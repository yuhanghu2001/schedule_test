import argparse
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from src.config import load_config
from src.environment import EdgeEnv
from src.evaluation import DAG_EVALUATION_METRICS
from src.plotting import CORE_METRICS, load_jsonl
from src.trainer import PRESET_NAMES, TRAIN_MODEL_NAMES, train_model


parser = argparse.ArgumentParser()
parser.add_argument("--config", "--preset", choices=PRESET_NAMES, default="small_smoke")
parser.add_argument("--model", choices=TRAIN_MODEL_NAMES, help="运行两回合训练、固定评估和绘图")
args = parser.parse_args()
cfg = load_config(args.config)
with TemporaryDirectory(prefix="schedule_test_smoke_") as temporary:
    run_dir = Path(temporary)
    with open(run_dir / "smoke_config.json", "x", encoding="utf-8") as file:
        json.dump({"config": asdict(cfg), "config_source": args.config},
                  file, ensure_ascii=False, indent=2)

    if args.model:
        train_model(cfg, args.model, run_dir, episodes=2, config_source=args.config)
        evaluation = load_jsonl(run_dir / "evaluation_metrics.jsonl")[-1]
        fields = DAG_EVALUATION_METRICS if cfg.enable_dag_mode else tuple(field[0] for field in CORE_METRICS)
        metrics = {"episode": evaluation["episode"],
                   **{f"{field}_mean": evaluation[f"{field}_mean"]
                      for field in ("objective_score", *fields)}}
    else:
        env = EdgeEnv(cfg)
        obs = env.reset()
        rng = np.random.default_rng(10)
        done = False
        while not done:
            # v5 候选包含掩码无效项；拒绝动作仍参与抽样。
            action = (rng.choice(np.flatnonzero(obs["action_mask"])) if cfg.enable_dag_mode
                      else rng.integers(len(env.actions)))
            obs, _, done, _ = env.step(int(action))
        metrics = env.metrics()
    with open(run_dir / "smoke_metrics.json", "x", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
print("临时 smoke 产物已自动删除")
