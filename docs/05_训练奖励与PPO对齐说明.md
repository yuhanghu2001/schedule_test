# 训练奖励与 PPO 对齐说明

## 1. 实验报告指标与训练标量

论文与实验分别报告任务完成率 `R_suc`、成功任务平均时延 `T_avg`、成功任务平均能耗 `E_avg` 和时隙平均负载方差 `D_load`。训练单步奖励与论文公式保持一致：

```text
r_i = beta_1*delta_i
    - y_i*(beta_2*t_i + beta_3*E_i)
    - beta_4*D_load
    - Omega_i
```

`beta_1:beta_2:beta_3:beta_4=2:1:1:1`。候选过滤和策略输入使用估计值；接收后，环境按各弧当前拥塞抽取实际扰动，并抽取任务级算力退化。任务已接收且实际时延满足 deadline 时 $\delta_i=1$，其余取零。奖励中的时延、能耗和边缘负载方差分别用 $t_{ref}=1.0\,\mathrm{s}$、$E_{ref}=3.0\,\mathrm{J}$ 和 $D_{ref}$ 归一化。

在 v6 `default` 场景下运行随机非拒绝策略（有可行接收动作时从中随机选择），用种子 1042–1046 统计时隙末边缘负载方差均值 $\bar D^{rand}$ 和成功任务平均实际时延 $\bar t^{rand}$。取基础到达率 $\lambda_0=18.0$，按下式计算并保留两位有效数字。

\[
D_{ref}=\frac{\bar D^{rand}}{\lambda_0\,\bar t^{rand}/t_{ref}}.
\]

v6 实测采用与固定评估相同的 episode 入口。每个种子单独构造 `EdgeEnv`，再显式调用一次 `reset()`。策略使用独立的 `np.random.default_rng(seed)`，每步在前 `len(env.actions)-1` 个可行接收动作中均匀抽取一个，只有拒绝动作时选它。每个种子运行 20 个时隙，episode 结束后按 KKT 重分配后的最终实际时延判定成功。

| 种子 | 到达任务数 | 成功任务数 | 时隙末边缘负载方差均值 | 成功任务平均实际时延（s） |
| ---: | ---: | ---: | ---: | ---: |
| 1042 | 492 | 374 | 0.08510734 | 0.83576435 |
| 1043 | 491 | 345 | 0.08707231 | 0.76133045 |
| 1044 | 509 | 366 | 0.08214011 | 0.85595540 |
| 1045 | 451 | 325 | 0.09258979 | 0.86269848 |
| 1046 | 443 | 311 | 0.08350115 | 0.78315209 |

合并 100 个时隙得到 $\bar D^{rand}=0.08608214164$，合并 1,721 个成功任务得到 $\bar t^{rand}=0.82071577979$ s。代入 $0.08608214164/(18.0\times0.82071577979/1.0)=0.00582703698$，两位有效数字为 **`D_ref=0.0058`**。

## 2. 单步奖励不变量

```text
r_i = current_task_actual_term
    + previous_group_task_term_delta
    - (edge_load_variance_after-edge_load_variance_before)/D_ref
    - omega_2_or_omega_3
```

当前任务接收后的实际项为 `2*delta-delay/1.0-energy/3.0-omega_1*(1-delta)`。同一时隙 KKT 重分配改变旧任务最终结果时，差额在当前步结算；每时隙第一步的负载 `before=0`，其余步骤用动作前方差。`omega_2` 惩罚不可行动作，`omega_3` 惩罚存在可行动作时主动拒绝；无可行动作时拒绝不惩罚。未折扣奖励满足：

```text
episode_reward / N_T == objective_score
```

这里 `N_T` 为本 episode 实际到达任务数，`objective_score` 定义为 `episode_reward/N_T`，与带约束惩罚的训练奖励口径一致；四项原始报告指标仍分开输出。

## 3. 观测与价值估计

观测中的 `global_features` 不包含未来信息：任务进度、接收进度、当前接收率、累计实际时延/能耗、已结束与当前时隙边缘负载、episode 时隙进度和当前时隙已处理任务比例。边缘与云的节点占用含在途任务。v6 的 GAT 图含 12 个边缘与 1 个云节点，端侧通过任务特征和接入边缘输入。边张量形状为 `[M+1,M+1,4]`，四个分量依次为归一化当前带宽、有效拥塞率、归一化传播时延、背景拥塞趋势 Δρ。

## 4. PPO 诊断和固定评估

每轮记录 `policy_loss,value_loss,entropy,normalized_entropy,approx_kl,clip_fraction,explained_variance,grad_norm_preclip,value_mean,return_mean`。训练期间的随机动作曲线不用于选择模型；每25轮在固定验证种子上贪心评估，按 `objective_score` 均值保存 best checkpoint，并始终保存 last checkpoint。checkpoint 记录 `reward_version` 和 `observation_version`。既有原型的观测为三维边特征，负载参考量按旧场景标定。v6 实验须随 checkpoint 记录四维观测版本与实测回填的奖励配置。

PPO-Clip 在采样时保存旧策略的 `log_prob`，并在当前轨迹的5轮更新中固定它作为概率比分母；下一轮轨迹采样时直接使用更新后的策略。因此当前标准 PPO 实现不维护独立的目标策略网络，也不使用 `kappa=0.995` 的软更新；将该系数加入会改变为图中 MPPO 的旧策略网络变体，不再是当前 PPO-Clip 更新口径。
