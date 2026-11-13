# 修正跳数与路径长度指标命名

## 对话目标

修正多跳指标口径，避免将完整传输路径长度误称为平均多跳数；保留纯边缘间转发跳数指标，并用 smoke 与固定种子单次评估验证两种口径。

## 已完成事项

- 保留 `mean_hops_accepted`，其语义为已接收任务的平均边缘—边缘转发跳数。
- 完整传输路径指标统一使用 `mean_path_length`，计算口径不变：本地0、接入边缘1、远端边缘1+边缘间跳数、云端2。
- 同步修改环境累计量、评估指标集、训练日志、组合实验汇总、测试、实验设计、数据字典和历史工作记录。
- 不保留旧指标名别名，避免新旧口径并存。
- `src/plotting.py`、`scripts/evaluate.py` 和其他评估脚本本来不直接引用旧名，因此无需增加无关修改。

## 验证证据

- `python -m unittest discover -s tests -v`：24 项测试全部通过。
- `python -m scripts.smoke`：完整运行 36 个任务，输出 `mean_hops_accepted=1.677419`、`mean_path_length=2.774194`，临时产物自动删除。
- 默认配置、固定评估种子 1042 的单次启发式 episode：`mean_hops_accepted=0.166386554622`，`mean_path_length=1.134453781513`，任务完成率为 `0.991666666667`。
- 实测值满足 `mean_path_length = mean_hops_accepted + edge_execution_rate / task_success_rate + 2 * cloud_execution_rate / task_success_rate`。
- 测试前后 `results/` 文件列表一致，未将开发测试产物留在正式结果目录。

## 未解决事项和实现边界

- 这两个数值来自未训练启发式基线，只用于验证指标口径和环境运行，不是 PPO 训练结论。
- 平均跳数与平均传输路径长度不能互换；论文、日志和图表必须使用各自的正式名称。

## 后续建议

- 正式训练和评估同时报告 `mean_hops_accepted`、`mean_path_length`、任务完成率及本地/边缘/云执行比例，并在图表中明确单位和口径。
