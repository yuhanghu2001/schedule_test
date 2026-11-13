# 训练奖励与 PPO 对齐说明

## 1. 实验报告指标与训练标量

论文与实验分别报告任务完成率 `R_suc`、成功任务平均时延 `T_avg`、成功任务平均能耗 `E_avg` 和时隙平均负载方差 `D_load`。训练单步奖励与论文公式保持一致：

```text
r_i = beta_1*delta_i
    - y_i*(beta_2*t_i + beta_3*E_i)
    - beta_4*D_load
    - Omega_i
```

`beta_1:beta_2:beta_3:beta_4=2:1:1:1`。`t_i`、`E_i` 和 `D_load` 分别用 `1.0s`、`3.0J` 和 `0.05` 归一化。

## 2. 单步奖励不变量

```text
r_i = 2*completed
    - accepted*(delay/1.0 + energy/3.0)
    - load_variance/0.05
    - constraint_penalty
```

约束惩罚按图中分段定义：接收但未按期完成为 `omega_1`，不可行动作为 `omega_2`，存在可行动作时主动拒绝为 `omega_3`，默认均为 `1.0`。不可行动作不会进入候选集，无可行动作时拒绝不惩罚。未折扣奖励满足：

```text
episode_reward / N == objective_score
```

这里 `objective_score` 定义为 `episode_reward/N`，与带约束惩罚的训练奖励口径一致；四项原始报告指标仍分开输出。

## 3. 观测与价值估计

观测增加不包含未来信息的 `global_features`：任务进度、完成进度、当前完成率、累计时延/能耗、已结束与当前时隙负载、时隙进度和时隙内任务进度。Actor 和 Critic 都使用该向量，新观测结构与旧 checkpoint 不兼容，必须从零训练。

## 4. PPO 诊断和固定评估

每轮记录 `policy_loss,value_loss,entropy,normalized_entropy,approx_kl,clip_fraction,explained_variance,grad_norm_preclip,value_mean,return_mean`。训练期间的随机动作曲线不用于选择模型；每25轮在固定验证种子上贪心评估，按 `objective_score` 均值保存 best checkpoint，并始终保存 last checkpoint。checkpoint 写入 `reward_version` 和 `observation_version`，禁止将新旧 reward 数值直接比较。

PPO-Clip 在采样时保存旧策略的 `log_prob`，并在当前轨迹的5轮更新中固定它作为概率比分母；下一轮轨迹采样时直接使用更新后的策略。因此当前标准 PPO 实现不维护独立的目标策略网络，也不使用 `kappa=0.995` 的软更新；将该系数加入会改变为图中 MPPO 的旧策略网络变体，不再是当前 PPO-Clip 更新口径。
