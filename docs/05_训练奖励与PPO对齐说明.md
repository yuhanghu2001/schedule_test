# 训练奖励与 PPO 对齐说明

## 1. 实验报告指标与训练标量

论文与实验分别报告任务完成率 `R_suc`、成功任务平均时延 `T_avg`、成功任务平均能耗 `E_avg` 和时隙平均负载方差 `D_load`。训练单步奖励与论文公式保持一致：

```text
r_i = beta_1*delta_i
    - y_i*(beta_2*t_i + beta_3*E_i)
    - beta_4*D_load
    - Omega_i
```

`beta_1:beta_2:beta_3:beta_4=2:1:1:1`。任务被接收后才抽取实际链路拥塞和算力退化，`delta_i`、`t_i`、`E_i` 均使用最终实际值；候选过滤和策略输入只使用估计值。三项分别用 `1.0s`、`3.0J` 和 `D_ref=0.012` 归一化。

`D_ref` 按 default 场景、随机非拒绝策略、种子1042—1046标定：`D_bar_rand=0.083446`、`t_bar_rand=0.71145s`，原始比值为 `0.011729`，保留两位有效数字得到 `0.012`。

## 2. 单步奖励不变量

```text
r_i = current_task_actual_term
    + previous_group_task_term_delta
    - (edge_load_variance_after-edge_load_variance_before)/0.012
    - omega_2_or_omega_3
```

当前任务实际项为 `2*delta-delay/1.0-energy/3.0-omega_1*(1-delta)`。同一时隙 KKT 重分配改变旧任务最终结果时，差额在当前步结算；每时隙第一步的负载 `before=0`，其余步骤用动作前方差。`omega_2` 惩罚不可行动作，`omega_3` 惩罚存在可行动作时主动拒绝；无可行动作时拒绝不惩罚。未折扣奖励满足：

```text
episode_reward / N == objective_score
```

这里 `objective_score` 定义为 `episode_reward/N`，与带约束惩罚的训练奖励口径一致；四项原始报告指标仍分开输出。

## 3. 观测与价值估计

观测中的 `global_features` 不包含未来信息：任务进度、接收进度、当前接收率、累计实际时延/能耗、已结束与当前时隙边缘负载、episode 时隙进度和当前时隙已处理任务比例。节点资源、内存和队列占用包含在途任务。`observation_version=v3_slot_arrivals_carryover`，与旧 checkpoint 不兼容。

## 4. PPO 诊断和固定评估

每轮记录 `policy_loss,value_loss,entropy,normalized_entropy,approx_kl,clip_fraction,explained_variance,grad_norm_preclip,value_mean,return_mean`。训练期间的随机动作曲线不用于选择模型；每25轮在固定验证种子上贪心评估，按 `objective_score` 均值保存 best checkpoint，并始终保存 last checkpoint。checkpoint 写入 `reward_version=v5_carryover_uncertainty` 和 `observation_version`，禁止将新旧 reward 数值直接比较。

PPO-Clip 在采样时保存旧策略的 `log_prob`，并在当前轨迹的5轮更新中固定它作为概率比分母；下一轮轨迹采样时直接使用更新后的策略。因此当前标准 PPO 实现不维护独立的目标策略网络，也不使用 `kappa=0.995` 的软更新；将该系数加入会改变为图中 MPPO 的旧策略网络变体，不再是当前 PPO-Clip 更新口径。
