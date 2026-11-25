"""直接查看一张拓扑和当前任务可用的前几条调度路线。"""
import argparse
import json

from src.baselines import feasible_indices
from src.config import load_config
from src.environment import EdgeEnv
from src.evaluation import DAG_EVALUATION_METRICS, evaluate_heuristic
from src.trainer import PRESET_NAMES


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", "--preset", choices=PRESET_NAMES, default="small_smoke")
args = parser.parse_args()
cfg = load_config(args.config)
env = EdgeEnv(cfg)
obs = env.observe()
print(f"网络邻接矩阵 (边缘 e0..e{env.cloud - 1}，最后一个为云 c0)：")
print(env.adj.astype(int))
if cfg.enable_dag_mode:
    print("当前 DAG 与子任务：", env.current_dag.dag_id, env.current_subtask)
else:
    print("当前任务：", env.tasks[env.idx])
feasible = feasible_indices(env.actions, obs.get("action_mask"))
print("联合动作数（含拒绝）：", len(env.actions))
print("有效接收候选数：", len(feasible))
print("观测版本与维度：", cfg.observation_version,
      {key: value.shape for key, value in obs.items() if hasattr(value, "shape")})
for index in feasible[:12]:
    action = env.actions[index]
    print("目标", action.destination, "资源", action.resource,
          "路径", action.path, "预测时延(秒)", round(action.delay, 4),
          "能耗(J)", round(action.energy, 4))
if cfg.enable_dag_mode:
    metrics = evaluate_heuristic(env, "heft_hetero")
    print("HEFT 单回合指标：", json.dumps({key: metrics[key] for key in DAG_EVALUATION_METRICS}))
