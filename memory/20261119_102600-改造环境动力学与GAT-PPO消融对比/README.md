# 20261119_102600 — 改造环境动力学与 GAT-PPO 消融对比

## 1. 对话目标

1. 验证本机通过 CLI 调用 Codex 协作开发的工作流是否畅通。
2. 全面审查仓库中 GAT-PPO 建模、奖励函数、环境动力学与实验基线设计，定位关键缺陷并给出可视化前端架构设计。
3. 按“先改数学文档、再由 Codex 改代码、最后人工/AI 逐行审核”的流程，完成环境动力学 `v5` 改造（跨时隙资源延续、执行不确定性、非平稳到达与边缘负载增量奖励）。
4. 实测 CPU 与 GPU（RTX 5060）训练耗时，确定默认运行设备、线程数与 conda 环境。
5. 调整 Actor 学习率并分析 1000 轮训练收敛情况；改造 `src/ablation.py` 为直接对比 `GAT-PPO` 与单纯 `PPO` 的脚本，对比最新两组 1000 轮正式实验指标并推送仓库。

## 2. 已完成事项

1. **Codex 协作链路验证**：验证 `codex exec --sandbox read-only/workspace-write` 正常运行，并完成整个 `v5` 环境改造与审核闭环。
2. **环境动力学与奖励重构（`v5_carryover_uncertainty`）**：
   - 解决了旧环境“接收即完成（完成率退化为 0.981）”、“时隙结束资源清零无跨时隙耦合”、“负载方差混入 20 个空闲终端且重复按步累计”等核心问题。
   - 引入非平稳泊松突发到达与热点终端、跨时隙在途任务资源冻结/释放、链路拥塞与算力退化执行不确定性、仅统计边缘节点的负载增量奖励与并发组 KKT 重分配奖励修正，并按随机非拒绝策略标定 `reward_variance_ref = 0.012`。
3. **新增基线与简化消融脚本**：
   - 在 `src/baselines.py` 新增一步估计奖励贪心基线 `greedy_estimate`。
   - 重写 `src/ablation.py` 为直接对比 `GAT-PPO`（`gat`）与单纯 `PPO`（`flat`）的运行脚本（同时保留原有 `model_environment` 兼容接口）。
4. **训练设备基准与超参调优**：
   - 实测默认配置下 CPU 2 线程约 `1.99 s/ep`，CUDA（RTX 5060）约 `12.09 s/ep`（小图逐样本循环导致 GPU 慢 6 倍）。
   - 将 `src/train.py` 默认线程数改为 `2`（默认设备 `cpu`），明确默认使用 `gatppo` conda 环境。
   - 根据 1000 轮 `lr=1e-4` 曲线尚未完全收敛、KL 偏小（`0.0014`）的现象，将 `actor_learning_rate` 调整为 `3e-4`。
5. **1000 轮正式实验分析与对比**：
   - 分析并对比了最新两组 `default` 场景 1000 轮正式实验：`results/20261118_195306`（完整 `GAT-PPO`）与 `results/20261118_204605`（单纯 `PPO`），以及传统启发式基线 `shortest_fastest` 和 `greedy_estimate`。

## 3. 关键代码和文档修改

- `docs/02_数学模型与C1-C24约束.md`：新增 0.1 非平稳到达、0.2 跨时隙占用、0.3 执行不确定性；同步修订 C13（$F_{jq}^{avail}(t)$）、C14–C16、C18、C22（$\mathcal N_t$）及第 5 节边缘负载增量奖励与 `D_ref` 标定公式。
- `README.md`、`docs/01`、`docs/03`、`docs/04`、`docs/05`：同步 `v5` 动力学、学习率 `3e-4`、默认 `gatppo` 环境与 CPU 2 线程说明。
- `src/config.py`：移除 `n_tasks`/`tasks_per_slot`，新增 `episode_slots=20`、到达/突发/热点/扰动参数，`reward_variance_ref=0.012`，`actor_learning_rate=3e-4`，更新版本标识为 `v5_carryover_uncertainty` / `v3_slot_arrivals_carryover`。
- `src/tasks.py`：`Task` 新增 `slot` 字段，实现泊松突发到达与周期热点终端抽样。
- `src/environment.py`：实现按 `Task.slot` 推进、在途任务资源冻结与释放、执行扰动固化、估计/实际分离、KKT 组奖励修正、边缘负载增量奖励、新指标（`acceptance_rate`、`deadline_miss_rate`）及边特征容量归一化。
- `src/baselines.py`、`scripts/evaluate_heuristic.py`：新增 `greedy_estimate` 基线与 `--method` 选项。
- `src/evaluation.py`：新增 `acceptance_rate` 与 `deadline_miss_rate` 聚合。
- `src/train.py`：默认 `--threads` 改为 `2`（默认 `--device cpu`）。
- `src/ablation.py`：新增直接对比 `GAT-PPO` 与单纯 `PPO` 的命令行入口与 `summary.json` 汇总。
- `tests/`：更新受影响测试用例，新增在途释放、实际超时触发 $\omega_1$、增量奖励分解、到达覆盖及贪心基线测试（共 34 项）。

## 4. 验证命令与结果证据

- **全量单元测试**（`gatppo` 环境）：
  ```bash
  MPLCONFIGDIR=/tmp/schedule_test_mpl PYTHONPATH=. /home/hyh/miniconda3/envs/gatppo/bin/python -m unittest discover tests
  ```
  结果：34 项全部通过（约 3.0s）。
- **Smoke 验证**：`scripts/smoke.py` 及 `src/ablation.py --config small_smoke --episodes 2` 在系统临时目录跑通并自动清理，无临时文件写入 `results/`。
- **CPU vs GPU 基准实测（RTX 5060 Laptop GPU）**：
  - `cpu, threads=2`：`collect=0.367s, update=1.623s, total=1.990s/ep`（最快）
  - `cuda, threads=1`：`collect=1.248s, update=10.839s, total=12.087s/ep`
- **最新正式实验对比（`default` 场景、`seed=42`、`episodes=1000`、`actor_lr=3e-4`，验证种子 `1042–1046` 均值）**：

| 方法 / 来源 | 综合得分 (`obj`) | 完成率 (`succ`) | 超时率 (`miss`) | 平均时延 (s) | 平均能耗 (J) | 边缘负载方差 | 平均跳数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `shortest_fastest` | -0.0064 | 0.9008 | 0.0857 | 0.6168 | 1.5949 | 0.07751 | — |
| `greedy_estimate` | 0.7212 | 0.9668 | 0.0204 | 0.6037 | 0.9174 | 0.04099 | — |
| `GAT-PPO` Best (`20261118_195306`, ep=800) | 0.8056 | **0.9356** | **0.0219** | **0.5962** | **0.7770** | 0.02850 | **0.5842** |
| `PPO` Best (`20261118_204605`, ep=700) | **0.8098** | 0.9296 | 0.0310 | 0.6086 | 0.8189 | **0.02020** | 0.7368 |
| `GAT-PPO` 后5次均值 (ep=900–1000) | 0.7118 | 0.9265 | 0.0233 | 0.5945 | 0.7987 | 0.03780 | 0.5803 |
| `PPO` 后5次均值 (ep=900–1000) | 0.7483 | 0.9222 | 0.0305 | 0.6067 | 0.7938 | 0.02840 | 0.7098 |

## 5. 尚未解决的事项和实现边界

1. 在 `default` 场景下，拓扑仅 6 个边缘节点且 20 时隙才刷新一次（等于整个 episode 长度），导致 `GAT-PPO` 虽在完成率、超时率（-29%）、时延、Best 能耗与跳数上优于单纯 `PPO`，但综合得分与单纯 `PPO` 接近（后者通过多跳分散压低了负载方差）。
2. 目前仅完成了单种子（`seed=42`）对比，正式论文结论仍需多种子（3–5 个 seed）及 `high_congestion` 场景验证。
3. 可视化前端调度系统仅完成了离线回放 + 在线交互的架构方案设计，尚未进入开发实现。

## 6. 后续建议与常用运行命令

- 在高拥塞动态拓扑场景（每 8 槽刷新拓扑）对比 `GAT-PPO` 与单纯 `PPO`：
  ```bash
  conda run -n gatppo python src/ablation.py --config high_congestion --episodes 1000 --seeds 0 1 2
  ```
- 单次训练完整 GAT-PPO 或单纯 PPO（默认 `cpu` + 2 线程）：
  ```bash
  conda run -n gatppo python src/train.py --episodes 1000
  conda run -n gatppo python src/train.py --model flat --episodes 1000
  ```
- 评估传统基线：
  ```bash
  conda run -n gatppo python -m scripts.evaluate_heuristic --method greedy_estimate --config default
  ```
