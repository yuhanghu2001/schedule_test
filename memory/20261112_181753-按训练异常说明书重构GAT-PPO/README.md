# 按训练异常说明书重构 GAT-PPO

## 基本信息

- 时间：2026-11-12 18:17:53（北京时间）
- 执行者：Codex
- 对话主题：根据用户提供的《GAT-PPO 训练异常问题详细修改说明书》完成 P0 与 P1 改造。

## 对话目标

1. 消除历史累计均值被每步重复奖励的错误。
2. 消除全拒绝 reward=0 优于正常接收策略的目标漏洞。
3. 使 Actor/Critic 能观测 reward 依赖的历史与进度。
4. 增加 PPO 稳定性诊断、固定场景验证和 best/last checkpoint。
5. 保留 GAT、Dinkelbach、KKT、合法候选动作和 C1–C24 约束主体。

## 完成事项

### P0：可加奖励

- 定义 `objective_score = 4*A/N - 2*T_sum/(N*0.5) - E_sum/(N*3.0) - sum(V_s)/(S*0.05)`。
- 单步 reward 使用接收指示、总时延/总能耗/负载方差增量，包含 KKT 重分配对旧任务的修正。
- 新增 `objective_score`、`reward_sum_check` 和四项 reward 分量。
- 新增可行任务率、条件拒绝率、强制拒绝率、动作数和执行位置比例。

### P1：观测、PPO与评估

- `observe()` 新增9维 `global_features`，Actor 和 Critic 共同使用。
- PPO 新增 `policy_loss`、`value_loss`、`entropy`、`normalized_entropy`、`approx_kl`、`clip_fraction`、`explained_variance`、`grad_norm_preclip`、`value_mean`、`return_mean`。
- 配置集中增加 reward 权重/参考量纲、`vf_coef`、`ent_coef`、`max_grad_norm`、评估间隔和版本号。
- 新增固定验证种子贪心评估，保存评估均值±标准差、`best_*_ppo.pt`、`last_*_ppo.pt` 和评估曲线。
- checkpoint 写入 `reward_version=v2_additive` 和 `observation_version=v2_global9`。
- 策略评估逻辑抽取至 `src/evaluation.py`，组合实验与启发式基线统一输出新指标。

### 文档和测试

- 先更新数学模型，再修改代码。
- 新增 `docs/05_训练奖励与PPO对齐说明.md`，并同步 README、AGENTS、开发说明、实验设计和数据字典。
- 新增 reward、观测和 PPO 诊断测试，完整测试数由11项增加到21项。

## 验证证据

- `results/20261112_175908/`：首轮21项单元测试全部通过。
- `results/20261112_175928/`：2 episode 小规模训练成功，生成训练/评估日志、两张曲线和 best/last checkpoint；reward恒等误差为 `2.2e-16` 和 `0`。
- `results/20261112_175955/`：best checkpoint 在独立种子2042上评估成功。
- `results/20261112_180006/`：同种子启发式基线评估成功。
- `results/20261112_180951/`：首次续训暴露新目录缺少 best checkpoint，如实保留证据。
- `results/20261112_181115/`：修正后续训成功，新目录同时含 best/last checkpoint。
- `results/20261112_181727/`：1种子的 GAT+启发式组合实验入口成功。
- `results/20261112_181753/`：最终21项回归测试全部通过。

## 未解决事项

- P2 GAT 结构未修改，按说明书要求应先用 P0+P1 新基线获得对照证据。
- 本轮仅完成小规模功能验证，没有伪造新500 episode 训练结果，不声称策略性能已提升。
- `4:2:1:1`、0.5s、3J、0.05 仍是工程初始值，需多种子敏感性实验。
- PPO 仍使用 `gamma=0.99`，折扣回报与最终未折扣 `J` 不严格等价。

## 后续建议

- 用新观测与新 reward 从零开始正式多种子训练，不续训旧 `v1` checkpoint。
- 先比较 GAT、flat、shuffled GAT 与启发式在固定验评集上的结果，再决定是否进入 P2 扩充图特征或调整 `vf_coef/gamma`。
