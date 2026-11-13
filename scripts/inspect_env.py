"""直接查看一张拓扑和当前任务可用的前几条调度路线。"""
from src.config import load_config
from src.environment import EdgeEnv


env = EdgeEnv(load_config("small_smoke"))
print("网络邻接矩阵 (边缘 e0..e5，最后一个为云 c0)：")
print(env.adj.astype(int))
print("当前任务：", env.tasks[env.idx])
print("可行联合动作数：", len(env.actions))
for action in env.actions[:12]:
    print("目标", action.destination, "资源", action.resource,
          "路径", action.path, "预测时延(秒)", round(action.delay, 4),
          "能耗(J)", round(action.energy, 4))
