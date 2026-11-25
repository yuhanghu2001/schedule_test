# v8 动态瓶颈调优、PPO 线性退火与 8 方法全面对比

## 1. 对话目标

1. 解决 `results/20261119_220735` 中 46.6% 主动拒绝套利塌缩问题，保留真实多跳与边路瓶颈，并拉开 `GAT-PPO` 相对 `PPO` 的优势。
2. 诊断 `results/20261120_093641` 中训练累计奖励波动与负载均衡度起伏的根因，通过退火消融提升 `GAT-PPO` 收敛稳定性。
3. 补齐 `DQN`、`DDPG`（Wolpertinger 离散映射）、遗传算法（`GA`）与蚁群算法（`ACO`）对比基线，建立统一 8 方法评估与出图入口，完成 4 个 RL 算法 600 回合正式对比训练及 8 方法独立测试评估。

## 2. 已完成事项

1. **Plan B / Plan C 受控筛选与 `v8_dynamic_bottleneck` 落地**
   - 在 `/tmp` 完成 Plan B 与 Plan C 的 200 回合 `GAT-PPO` vs `PPO` 筛选及 8 组共享噪声配对，锁定 Plan C（`link_rate 200–500 Mbps`、`deadline_slack 2.2–3.5`、`link_flap_prob=0.20`、`link_spike_prob=0.25`、`omega_3=2.5`、`D_ref=0.0045`）。
   - 先同步 `docs/02`、`docs/04`、`docs/05`，再更新 `src/config.py`，完成 600 回合正式对比训练 `results/20261120_093641`（主动拒绝率由 46.6% 降至 2.80%，完成率升至 82.81%，8 组共享噪声配对 `GAT-PPO` 综合分与负载方差 8/8 胜 `PPO`）。
2. **累计奖励与负载均衡度根因诊断**
   - 基于 `results/20261120_093641` 日志与最佳权重复跑定量验证 6 项假设：
     - 确认训练累计奖励波动主要源于外生场景差异（任务数 327–623）、负载项单步方差贡献达 83.3%，以及学习率与熵系数恒定无退火（后期 `approx_kl≈0.009`、`clip_fraction≈9.3%`）。
     - 确认负载均衡度中段下探（20 回合滑窗降幅 0.0145）是用均衡换完成率的结构性权衡，同时受窄纵轴放大。
3. **V1 线性退火受控实验与正式接入**
   - 在 `/tmp` 同口径 600 回合对比 V0（恒定）、V1（线性退火）、V2（余弦退火）、V3（100 回合预热 + 线性退火），V1 胜出：后 100 回合 `approx_kl` 由 0.00918 降至 0.00150、`clip_fraction` 由 8.79% 降至 1.74%，验证综合分显著跃升。
   - 正式接入 `PPOTrainer.set_schedule()` 与 `src/train.py`（学习率线性衰减至 10%，`ent_coef` 从 `0.01` 线性衰减至 `0.001`）。
4. **新增 DQN、DDPG、GA、ACO 与统一多方法评估**
   - 新增 `src/algorithms/dqn.py`（动态变长候选打分 DQN，支持可选 Double DQN）与 `src/algorithms/ddpg.py`（Wolpertinger 10 维连续原型映射到最近可行候选）。
   - 在 `src/baselines.py` 新增时隙级 `GA`、`ACO` 规划器，回放副本截断至当前时隙并替换独立 `execution_rng`，确保不窥视未来任务与真实执行噪声。
   - 扩展 `src/train.py` 与 `src/plotting.py` 支持任意多 RL 算法顺序训练与叠加对比；新增 `scripts/compare_methods.py` 统一在独立测试种子上评测全部 8 种方法并输出 JSON/CSV、四核心指标柱状图及 RL 训练奖励对比图。
5. **4 RL 600 回合正式训练 + 8 方法统一对比归档**
   - 完成 `results/20261124_134034/`（35 个文件）：4 个 RL 算法（`gatppo`、`ppo`、`dqn`、`ddpg`）各 600 回合（`seed=42`）+ 全部 8 种方法在独立测试种子 `2042` 上的统一评估。
   - 本窗口改动已精准暂存（仅暂存 `src/config.py` 中退火与 4 基线片段，完全保留外部窗口的 DAG 与前端改动）并提交至本地 `main` 分支（commit `63dfb7d`）。

## 3. 关键代码和文档修改

- `src/config.py`：落地 `v8_dynamic_bottleneck` 默认参数，新增 `ppo_lr_end_factor=0.1`、`ppo_ent_coef_end=0.001` 及 `dqn_*`、`ddpg_*`、`ga_*`、`aco_*` 配置项。
- `src/algorithms/ppo.py`：新增 `PPOTrainer.set_schedule()` 并在 `update()` 中使用动态 `self.ent_coef`。
- `src/algorithms/dqn.py`（新增）：变长候选 Q 网络、经验回放、硬同步目标网络与 ε-greedy 训练器。
- `src/algorithms/ddpg.py`（新增）：Wolpertinger 离散映射 Actor/Critic、软更新目标网络与训练器。
- `src/baselines.py`：新增 `genetic_plan`、`ant_colony_plan`、`evaluate_slot_planner` 接入 `evaluate_heuristic`。
- `src/train.py`、`src/plotting.py`：支持多算法顺序训练、PPO 退火调用、多算法曲线图与四指标柱状图绘制。
- `scripts/compare_methods.py`（新增）：8 方法统一测试评估与图表生成脚本。
- `tests/test_added_methods.py`（新增）、`tests/test_ppo.py`：覆盖 DQN/DDPG 训练评估、GA/ACO 动作合法性与随机数隔离、PPO 线性退火调度。
- `docs/02_数学模型与C1-C24约束.md`、`docs/03_实验设计与指标.md`、`docs/04_数据字典与实现边界.md`、`docs/05_训练奖励与PPO对齐说明.md`：同步 `v8_dynamic_bottleneck`、线性退火及新增对比方法说明。
- `results/20261120_093641/`、`results/20261124_134034/`：两轮正式训练与评估产物归档。

## 4. 验证命令与结果证据

- 全量单元测试：`MPLCONFIGDIR=/tmp/schedule_test_mpl PYTHONPATH=. /home/hyh/miniconda3/envs/gatppo/bin/python -m unittest discover tests` → **`Ran 50 tests in 7.557s — OK`**。
- 快速冒烟：`MPLCONFIGDIR=/tmp/schedule_test_mpl PYTHONPATH=. /home/hyh/miniconda3/envs/gatppo/bin/python scripts/smoke.py` → 通过（16 任务，完成率 68.75%，临时目录自动清理）。
- 4 RL 固定验证对比（`results/20261124_134034`，种子 `1042–1046`，600 回合）：
  - `gatppo`：最佳 `-0.3687`（ep500），第 600 回合 `-0.3883`，**末 5 点均值 `-0.3899`（4 RL 第一）**。
  - `dqn`：最佳 `-0.3161`（ep100），第 600 回合回落至 `-0.4223`，末 5 点均值 `-0.4252`。
  - `ppo`：最佳 `-0.6427`（ep600），末 5 点均值 `-0.7091`。
  - `ddpg`：最佳 `-1.3227`（ep75），末 5 点均值 `-1.3252`。
- 8 方法独立测试对比（`results/20261124_134034/method_comparison/`，独立种子 `2042` 单回合）：
  - `gatppo`：完成率 80.31%、时延 0.7651 s、能耗 1.4251 J、均衡度 0.9446、综合分 -0.5023。
  - `ppo`：完成率 75.13%、时延 0.8015 s、能耗 1.6223 J、均衡度 0.9523、综合分 -0.6574。
  - `dqn`（ep100 峰值权重）：完成率 84.72%、时延 0.7437 s、能耗 1.3018 J、均衡度 0.9366、综合分 -0.3767。
  - `ddpg`：完成率 77.46%、时延 0.8117 s、能耗 1.6845 J、均衡度 0.8979、综合分 -1.4293。
  - `ga`：完成率 86.27%、时延 0.7610 s、能耗 1.4007 J、均衡度 0.9433、综合分 -0.3094（单回合耗时 ~161 s）。
  - `aco`：完成率 87.56%、时延 0.7596 s、能耗 1.3294 J、均衡度 0.9329、综合分 -0.3785（单回合耗时 ~144 s）。
  - `greedy_estimate`：完成率 88.60%、时延 0.7600 s、能耗 1.2975 J、均衡度 0.9012、综合分 -0.7779。
  - `shortest_fastest`：完成率 86.53%、时延 0.7613 s、能耗 1.4549 J、均衡度 0.8790、综合分 -1.2247。

## 5. 尚未解决的事项和实现边界

1. **Git 推送尚未执行**：本地提交 `63dfb7d` 已生成（未包含外部窗口的 DAG 与前端改动），但 `git push origin main` 权限被拒绝，待用户在终端手动执行推送。
2. **仅完成了单训练种子（`seed=42`）与单独立测试种子（`2042`）**：单回合测试存在随机波动（例如 `gatppo` 在单种子 `2042` 上低于其 5 种子验证均值 `-0.3883`），尚不足以作为最终论文统计显著性结论。
3. **GA / ACO 为高开销离线/准离线参考**：每回合需大量副本前向仿真（144–161 秒/回合），与 RL 的毫秒级在线推理不在同一实时性量级。
4. **工作区存在并发窗口改动**：另一窗口正在修改 DAG 相关文件（`src/config.py` DAG 段、`src/tasks.py`、`src/topology.py`、`src/environment.py`、`src/resources.py`、`docs/02`、`frontend/`、`schedule_test_dag.md`、`tests/test_dag_tasks.py`），本窗口已完全保留未触碰。

## 6. 后续建议与常用运行命令

1. **手动推送已提交成果**：
   ```bash
   git push origin main
   ```
2. **用现有权重直接补跑 5–10 个独立测试种子评估（无需重训）**：
   ```bash
   MPLCONFIGDIR=/tmp/schedule_test_mpl PYTHONPATH=. /home/hyh/miniconda3/envs/gatppo/bin/python \
     scripts/compare_methods.py --run-dir results/20261124_134034 --config default \
     --seeds 2042 2043 2044 2045 2046 --output-dir results/20261124_134034/method_comparison_5seeds
   ```
3. **4 算法顺序正式训练入口**：
   ```bash
   MPLCONFIGDIR=/tmp/schedule_test_mpl PYTHONPATH=. /home/hyh/miniconda3/envs/gatppo/bin/python \
     src/train.py --config default --model gatppo ppo dqn ddpg --episodes 600 --seed 42 --device cpu --threads 2
   ```
