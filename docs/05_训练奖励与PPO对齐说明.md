# 训练奖励与 PPO 对齐说明

## 1. 实验报告指标与训练标量

论文与实验分别报告任务完成率 `R_suc`、成功任务平均时延 `T_avg`、成功任务平均能耗 `E_avg` 和时隙平均负载方差 `D_load`。训练不直接重复累加这些截至当前的均值，而使用固定任务数分母的可加目标：

```text
J = 4*A/N
  - 2*T_sum/(N*0.5)
  - E_sum/(N*3.0)
  - sum(V_s)/(S*0.05)
```

`4:2:1:1` 是首轮训练的工程初始权重，必须通过固定场景和多种子实验校准，不写成已证明最优权重。

## 2. 单步奖励不变量

```text
r_t = 4*accepted
    - 2*delta_delay/0.5
    - delta_energy/3.0
    - (N/S)*delta_variance/0.05
```

`delta_delay` 和 `delta_energy` 使用环境总统计的前后差，因此包含并发组 KKT 重分配对旧任务的修正。时隙内负载方差增量望远镜求和。未折扣奖励必须通过测试满足：

```text
episode_reward / N == objective_score
```

拒绝动作本身奖励为0；可行低成本接收动作应由正完成收益获得更高奖励。只在固定评估证明策略过度主动拒绝后，才增加单独拒绝惩罚。

## 3. 观测与价值估计

观测增加不包含未来信息的 `global_features`：任务进度、完成进度、当前完成率、累计时延/能耗、已结束与当前时隙负载、时隙进度和时隙内任务进度。Actor 和 Critic 都使用该向量，新观测结构与旧 checkpoint 不兼容，必须从零训练。

## 4. PPO 诊断和固定评估

每轮记录 `policy_loss,value_loss,entropy,normalized_entropy,approx_kl,clip_fraction,explained_variance,grad_norm_preclip,value_mean,return_mean`。训练期间的随机动作曲线不用于选择模型；每25轮在固定验证种子上贪心评估，按 `objective_score` 均值保存 best checkpoint，并始终保存 last checkpoint。checkpoint 写入 `reward_version` 和 `observation_version`，禁止将新旧 reward 数值直接比较。
