# 增加必须卸载约束并修复多跳统计

## 对话目标

修复启发式调度中本地执行抢占外部调度需求、`mean_selected_hops` 过低的问题，引入默认 60% 的必须卸载任务，并验证空批次可行率、完整启发式 episode 完成率和跳数。

## 已完成事项

- 先更新 `docs/02_数学模型与C1-C24约束.md` 的 C5，增加 `m_i^{off}` 外部调度标志与 `x_{i,u_i} <= 1-m_i^{off}`、`m_i^{off} <= a_i`，未改动 C1–C24 编号。
- `Task` 新增 `must_offload`，`sample_tasks` 按 `must_offload_rate` 抽样；默认三套配置均设为 `0.6`。
- 必须卸载任务必然同时设置 `allow_offload=True`，避免约束自相矛盾。
- `_make_candidates()` 对 `must_offload=True` 的任务不生成 `destination=-1` 的源端本地候选，边缘、云端和拒绝动作保持原有物理约束。
- 修正跳数指标口径：`mean_hops_accepted` 继续仅统计边缘—边缘跳数；`mean_selected_hops` 统计完整选中传输路径段数，本地0、接入边缘1、远端边缘为1+边缘跳数、云端2。
- 训练日志增加“平均选中跳数”，组合实验汇总增加 `mean_hops_accepted` 和 `mean_selected_hops`。
- 同步 README、环境说明、实验设计和数据字典。
- 增加回归测试，固化空批次可行率 `>=95%`、完成率 `>=95%`、`mean_selected_hops` 位于 `0.5–1.5` 且存在真实边缘间多跳。

## 验证证据

- 24 项单元测试全部通过。
- 固定验证种子 `1042..1046`的启发式完整 episode 结果：
  - 空批次单任务可行率均值：`0.991333`。
  - 任务完成率均值：`0.991333`。
  - `mean_selected_hops` 均值：`1.125797`。
  - `mean_hops_accepted` 均值：`0.153992`。
- 种子 1042 单轮结果：完成率 `0.991667`，`mean_selected_hops=1.134454`，纯边缘间平均跳数 `0.166387`。
- 所有验证直接运行内存中的环境或系统临时目录，未向 `results/` 写入开发产物。

## 未解决事项和实现边界

- 本次跳数与完成率来自未训练的启发式基线，用于验证环境动力学和候选动作是否激活多跳，不是 PPO 训练结论。
- `mean_selected_hops` 与 `mean_hops_accepted` 口径不同，后续论文和图表必须指明使用哪一个指标。

## 后续建议

- 下次正式训练时在配置快照中保留 `must_offload_rate`，并同时报告完成率、`mean_selected_hops`、`mean_hops_accepted`和本地/边缘/云执行比例。
